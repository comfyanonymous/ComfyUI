import asyncio
import contextvars
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch

import comfy.conds
import comfy.hooks
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.supported_models
from comfy_execution.progress import reset_progress_registry, set_progress_registry
from comfy_execution.utils import CurrentNodeContext, reset_current_client_id, set_current_client_id


FAMILY_ANIMA = "anima"
FAMILY_SD15 = "sd15"
FAMILY_SDXL = "sdxl"
CONTINUOUS_SAMPLER_NODE_FAMILIES = {
    "AnimaContinuousKSampler": FAMILY_ANIMA,
    "SD15ContinuousKSampler": FAMILY_SD15,
    "SDXLContinuousKSampler": FAMILY_SDXL,
}

_COORDINATORS = {}
_VAE_COORDINATORS = {}
_CANCEL_CHECKER = None


def set_cancel_checker(checker):
    global _CANCEL_CHECKER
    _CANCEL_CHECKER = checker


def _is_cancelled(prompt_id):
    return prompt_id is not None and _CANCEL_CHECKER is not None and _CANCEL_CHECKER(prompt_id)


def euler_step(x, denoised, sigma, sigma_next):
    return x + (x - denoised) / sigma * (sigma_next - sigma)


_UNSUPPORTED_CONDITIONING = {
    "additional_models",
    "area",
    "clip_end_percent",
    "clip_start_percent",
    "control",
    "crossattn_controlnet",
    "default",
    "end_percent",
    "gligen",
    "hooks",
    "mask",
    "mask_strength",
    "noise_concat",
    "start_percent",
    "strength",
    "timestep_end",
    "timestep_start",
    "unclip_conditioning",
}


def _validate_conditioning(name, conditions):
    if len(conditions) != 1:
        raise ValueError(f"Continuous batching requires one {name} conditioning entry")
    present = _UNSUPPORTED_CONDITIONING.intersection(conditions[0][1])
    if present:
        raise ValueError("Continuous batching does not support conditioning feature(s): {}".format(", ".join(sorted(present))))


def _value_structure(value):
    if torch.is_tensor(value):
        return ("tensor", tuple(value.shape), value.dtype, value.device)
    if isinstance(value, dict):
        return ("dict", tuple((key, _value_structure(item)) for key, item in sorted(value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_value_structure(item) for item in value))
    return type(value)


def _conditioning_structure(family, conditions):
    value, metadata = conditions[0]
    if family == FAMILY_ANIMA:
        text_ids = metadata.get("t5xxl_ids")
        if torch.is_tensor(text_ids):
            return ("anima_context", max(512, text_ids.shape[-1]), value.shape[-1], value.dtype)
    return (
        _value_structure(value),
        tuple((key, _value_structure(item)) for key, item in sorted(metadata.items())),
    )


def _nested_mapping_entries(mapping):
    if not isinstance(mapping, dict):
        return bool(mapping)
    return any(_nested_mapping_entries(value) for value in mapping.values())


def _wrapper_types(wrappers):
    present = set()
    for wrapper_type, keyed_wrappers in wrappers.items():
        if _nested_mapping_entries(keyed_wrappers):
            present.add(getattr(wrapper_type, "value", wrapper_type))
    return present


def _model_wrapper_types(model_patcher):
    transformer_options = model_patcher.model_options.get("transformer_options", {})
    wrappers = {}
    comfy.patcher_extension.merge_nested_dicts(wrappers, model_patcher.wrappers, copy_dict1=False)
    comfy.patcher_extension.merge_nested_dicts(wrappers, transformer_options.get("wrappers", {}), copy_dict1=False)
    return _wrapper_types(wrappers)


def _validate_model_extensions(family, model_patcher):
    model_options = model_patcher.model_options
    unsupported_options = {
        "context_handler",
        "model_function_wrapper",
        "multigpu_clones",
        "sampler_calc_cond_batch_function",
        "sampler_cfg_function",
        "sampler_post_cfg_function",
        "sampler_pre_cfg_function",
    }
    present = unsupported_options.intersection(model_options)
    if present:
        raise ValueError("Continuous batching does not support model option(s): {}".format(", ".join(sorted(present))))

    if _nested_mapping_entries(model_patcher.callbacks):
        raise ValueError("Continuous batching does not support model callbacks")
    if any(model_patcher.additional_models.values()):
        raise ValueError("Continuous batching does not support additional models")

    transformer_options = model_options.get("transformer_options", {})
    if _nested_mapping_entries(transformer_options.get("callbacks", {})):
        raise ValueError("Continuous batching does not support transformer callbacks")

    present_wrappers = _model_wrapper_types(model_patcher)
    if present_wrappers:
        raise ValueError("Continuous batching does not support wrapper(s): {}".format(", ".join(sorted(present_wrappers))))

    patch_keys = {"patches", "patches_replace"}.intersection(transformer_options)
    if any(_nested_mapping_entries(transformer_options[key]) for key in patch_keys):
        raise ValueError("Continuous batching does not support transformer patches")


def _validate_model_family(family, model_patcher):
    model = model_patcher.model
    if family == FAMILY_ANIMA:
        if not isinstance(model, comfy.model_base.Anima):
            raise ValueError("Anima continuous batching requires an Anima model")
        return

    latent_channels = model.latent_format.latent_channels
    input_channels = getattr(model.diffusion_model, "in_channels", None)
    if family == FAMILY_SD15:
        if type(model) is not comfy.model_base.BaseModel or not isinstance(model.model_config, comfy.supported_models.SD15):
            raise ValueError("SD1.5 continuous batching requires a standard SD1.5 model")
    elif family == FAMILY_SDXL:
        if type(model) not in (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner):
            raise ValueError("SDXL continuous batching requires a standard SDXL or SDXL Refiner model")
    else:
        raise ValueError(f"Unknown continuous batching model family: {family}")

    if input_channels != latent_channels or model.concat_keys:
        raise ValueError("SD continuous batching does not support inpaint or concatenated-input models")


def _processed_conditioning_signature(family, cond):
    expected = {
        FAMILY_ANIMA: {"c_crossattn": comfy.conds.CONDRegular},
        FAMILY_SD15: {"c_crossattn": comfy.conds.CONDCrossAttn},
        FAMILY_SDXL: {"c_crossattn": comfy.conds.CONDCrossAttn, "y": comfy.conds.CONDRegular},
    }[family]
    if cond is None or cond.area is not None or cond.control is not None or cond.patches is not None or cond.hooks is not None:
        raise ValueError("Continuous batching received unsupported processed conditioning")
    if cond.conditioning.keys() != expected.keys():
        present = ", ".join(sorted(cond.conditioning))
        raise ValueError(f"Continuous batching received unsupported processed conditioning keys: {present}")

    signature = []
    for key, expected_type in expected.items():
        value = cond.conditioning[key]
        if type(value) is not expected_type or not torch.is_tensor(value.cond):
            raise ValueError(f"Continuous batching received an unsupported {key} conditioning wrapper")
        signature.append((key, type(value), tuple(value.cond.shape), value.cond.dtype, value.cond.device))
    return tuple(signature)


@dataclass(frozen=True)
class _PreparedConditioning:
    conditioning: dict
    uuid: Any
    signature: tuple


def cfg_combine(cond, uncond, cfg):
    if math.isclose(cfg, 1.0):
        return cond
    return uncond + (cond - uncond) * cfg


def _cfg_branches(cfg, model_options):
    if math.isclose(cfg, 1.0) and not model_options.get("disable_cfg1_optimization", False):
        return (("positive", 0),)
    return (("negative", 1), ("positive", 0))


@dataclass
class ContinuousBatchRequest:
    family: str
    model_patcher: Any
    noise: torch.Tensor
    latent_image: torch.Tensor
    positive: list
    negative: list
    sigmas: torch.Tensor
    callback: Any
    seed: int
    cfg: float
    max_batch_size: int
    admission_delay: float
    prompt_id: str | None = None
    node_id: str | None = None
    client_id: str | None = None
    progress_registry: Any = None
    index: int = 0
    x: torch.Tensor | None = None
    conds: dict | None = None
    output: torch.Tensor | None = None
    prepared: bool = False
    processed_conds: dict | None = field(default=None, init=False)

    def validate(self):
        _validate_model_family(self.family, self.model_patcher)
        if self.model_patcher.is_dynamic():
            raise ValueError("Continuous batching does not support dynamic model patchers")
        if self.latent_image.is_nested or self.noise.is_nested:
            raise ValueError("Continuous batching does not support nested latents")
        if self.latent_image.shape != self.noise.shape or self.latent_image.shape[0] != 1:
            raise ValueError("Continuous batching requires one latent per request")
        if self.sigmas.ndim != 1 or len(self.sigmas) < 2:
            raise ValueError("Continuous batching requires a one-dimensional sigma schedule")
        if self.max_batch_size < 1:
            raise ValueError("Continuous batching max batch size must be positive")
        if not math.isfinite(self.cfg) or self.cfg < 0:
            raise ValueError("Continuous batching CFG must be finite and non-negative")
        _validate_conditioning("positive", self.positive)
        _validate_conditioning("negative", self.negative)
        _validate_model_extensions(self.family, self.model_patcher)

    def key(self):
        self.validate()
        return (
            self.family,
            self.model_key(),
            tuple(self.latent_image.shape[1:]),
            _conditioning_structure(self.family, self.positive),
            _conditioning_structure(self.family, self.negative),
            self.max_batch_size,
            self.admission_delay,
        )

    def model_key(self):
        return (
            id(self.model_patcher.model),
            id(self.model_patcher),
            self.model_patcher.patches_uuid,
            self.model_patcher.load_device,
            self.model_patcher.model_dtype(),
        )

    def clear(self):
        self.model_patcher = None
        self.noise = None
        self.latent_image = None
        self.positive = None
        self.negative = None
        self.sigmas = None
        self.callback = None
        self.progress_registry = None
        self.x = None
        self.conds = None
        self.processed_conds = None
        self.output = None


@dataclass
class _QueuedRequest:
    state: ContinuousBatchRequest
    future: asyncio.Future = field(init=False)
    cancelled: bool = False

    def __post_init__(self):
        self.future = asyncio.get_running_loop().create_future()


class ContinuousBatchSession:
    def __init__(self, model_patcher):
        self.model_patcher = model_patcher
        self.model_options = None
        self.inner_model = None
        self.loaded_models = None
        self.load_conds = None
        self.capacity = 0
        self.orig_hook_mode = None
        self.reload_model = False
        self.open = False

    def open_session(self, states):
        if self.open:
            return
        for state in states:
            state.validate()
        representative = states[0]
        combined_conds = {
            "positive": comfy.sampler_helpers.convert_cond(representative.positive),
            "negative": comfy.sampler_helpers.convert_cond(representative.negative),
        }
        self.model_options = comfy.model_patcher.create_model_options_clone(self.model_patcher.model_options)
        self.model_options.setdefault("transformer_options", {})["sample_sigmas"] = representative.sigmas
        comfy.samplers.preprocess_conds_hooks(combined_conds)
        self.orig_hook_mode = self.model_patcher.hook_mode
        self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.open = True
        try:
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, combined_conds, self.model_options)
            comfy.samplers.filter_registered_hooks_on_conds(combined_conds, self.model_options)
            self.inner_model, _, self.loaded_models = comfy.sampler_helpers.prepare_sampling(
                self.model_patcher,
                [len(states)] + list(representative.noise.shape[1:]),
                combined_conds,
                self.model_options,
            )
            self.load_conds = combined_conds
            self.capacity = len(states)
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.load_device, dtype=self.model_patcher.model_dtype())
            self.model_patcher.pre_run()
            self.reload_model = False
        except BaseException:
            self.close()
            raise

    def request_model_reload(self):
        if self.open:
            self.reload_model = True

    def ensure_model_loaded(self, states):
        required_capacity = max(self.capacity, len(states))
        if not self.reload_model and required_capacity == self.capacity:
            return
        representative = states[0]
        self.inner_model, _, self.loaded_models = comfy.sampler_helpers.prepare_sampling(
            self.model_patcher,
            [required_capacity] + list(representative.noise.shape[1:]),
            self.load_conds,
            self.model_options,
        )
        comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.load_device, dtype=self.model_patcher.model_dtype())
        self.capacity = required_capacity
        self.reload_model = False

    def prepare_request(self, state):
        if state.prepared:
            return
        device = self.model_patcher.load_device
        state.noise = state.noise.to(device=device, dtype=torch.float32)
        state.latent_image = state.latent_image.to(device=device, dtype=torch.float32)
        state.sigmas = state.sigmas.to(device)
        conds = {
            "positive": comfy.sampler_helpers.convert_cond(state.positive),
            "negative": comfy.sampler_helpers.convert_cond(state.negative),
        }
        conds = comfy.samplers.process_conds(
            self.inner_model,
            state.noise,
            conds,
            device,
            state.latent_image,
            None,
            state.seed,
            latent_shapes=[state.latent_image.shape],
        )
        latent_image = state.latent_image
        if torch.count_nonzero(latent_image) > 0:
            latent_image = self.inner_model.process_latent_in(latent_image)
        sigma = float(state.sigmas[0])
        sigma_max = float(self.inner_model.model_sampling.sigma_max)
        max_denoise = math.isclose(sigma_max, sigma, rel_tol=1e-5) or sigma > sigma_max
        state.x = self.inner_model.model_sampling.noise_scaling(state.sigmas[0], state.noise, latent_image, max_denoise)
        sigma = state.sigmas[0].to(state.x).unsqueeze(0)
        processed_conds = {}
        for name in ("positive", "negative"):
            if len(conds[name]) != 1:
                raise ValueError(f"Continuous batching requires one processed {name} conditioning entry")
            processed = comfy.samplers.get_area_and_mult(conds[name][0], state.x, sigma)
            signature = _processed_conditioning_signature(state.family, processed)
            processed_conds[name] = _PreparedConditioning(
                conditioning=processed.conditioning,
                uuid=processed.uuid,
                signature=signature,
            )
        state.latent_image = latent_image
        state.conds = conds
        state.processed_conds = processed_conds
        state.prepared = True

    def predict(self, states):
        if len(states) == 1:
            state = states[0]
            sigma = state.sigmas[state.index].to(state.x).unsqueeze(0)
            model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            model_options.setdefault("transformer_options", {})["sample_sigmas"] = state.sigmas
            return [comfy.samplers.sampling_function(
                self.inner_model,
                state.x,
                sigma,
                state.conds["negative"],
                state.conds["positive"],
                state.cfg,
                model_options=model_options,
                seed=state.seed,
            )]

        state_timesteps = []
        state_branches = []
        entries = []
        for state_index, state in enumerate(states):
            sigma = state.sigmas[state.index].to(state.x)
            state_timesteps.append(sigma)
            branches = _cfg_branches(state.cfg, self.model_options)
            state_branches.append(branches)
            for name, branch in branches:
                entries.append((state_index, name, branch, state.x, sigma, state.processed_conds[name]))

        buckets = []
        for entry in entries:
            for bucket in buckets:
                if (
                    entry[5].signature == bucket[0][5].signature
                    and entry[3].shape == bucket[0][3].shape
                    and comfy.samplers.cond_equal_size(entry[5].conditioning, bucket[0][5].conditioning)
                ):
                    bucket.append(entry)
                    break
            else:
                buckets.append([entry])

        state_timestep = torch.stack(state_timesteps)
        self.model_patcher.prepare_state(state_timestep, self.model_options)
        branch_outputs = [{} for _ in states]
        for bucket in buckets:
            input_x = torch.cat([entry[3] for entry in bucket])
            timestep = torch.stack([entry[4] for entry in bucket])
            conditioning = comfy.samplers.cond_cat([entry[5].conditioning for entry in bucket])
            transformer_options = self.model_patcher.apply_hooks(hooks=None)
            if "transformer_options" in self.model_options:
                transformer_options = comfy.patcher_extension.merge_nested_dicts(
                    transformer_options,
                    self.model_options["transformer_options"],
                    copy_dict1=False,
                )
            transformer_options["cond_or_uncond"] = [entry[2] for entry in bucket]
            transformer_options["uuids"] = [entry[5].uuid for entry in bucket]
            transformer_options["sigmas"] = timestep
            conditioning["transformer_options"] = transformer_options
            outputs = self.inner_model.apply_model(input_x, timestep, **conditioning).split(1)
            for entry, output in zip(bucket, outputs):
                branch_outputs[entry[0]][entry[1]] = output

        predictions = []
        for state, branches, outputs in zip(states, state_branches, branch_outputs):
            if len(branches) == 1:
                predictions.append(outputs["positive"])
            else:
                predictions.append(cfg_combine(outputs["positive"], outputs["negative"], state.cfg))
        return predictions

    @staticmethod
    def run_callback(state, prediction):
        if state.callback is None:
            return
        client_token = set_current_client_id(state.client_id)
        progress_token = set_progress_registry(state.progress_registry) if state.progress_registry is not None else None
        try:
            if state.prompt_id is not None and state.node_id is not None:
                with CurrentNodeContext(state.prompt_id, state.node_id):
                    state.callback(state.index, prediction, state.x, len(state.sigmas) - 1)
            else:
                state.callback(state.index, prediction, state.x, len(state.sigmas) - 1)
        finally:
            if progress_token is not None:
                reset_progress_registry(progress_token)
            reset_current_client_id(client_token)

    def step(self, states):
        with comfy.model_management.cuda_device_context(self.model_patcher.load_device):
            self.open_session(states)
            self.ensure_model_loaded(states)
            for state in states:
                self.prepare_request(state)
            denoised = self.predict(states)
            updates = []
            for state, prediction in zip(states, denoised):
                if prediction.shape != state.x.shape:
                    raise RuntimeError("Continuous batch denoiser returned an invalid shape")
                sigma = state.sigmas[state.index].to(state.x)
                self.run_callback(state, prediction)
                state.x = euler_step(state.x, prediction, sigma, state.sigmas[state.index + 1].to(state.x))
                state.index += 1
                finished = state.index == len(state.sigmas) - 1
                if finished:
                    samples = self.inner_model.model_sampling.inverse_noise_scaling(state.sigmas[-1], state.x)
                    samples = self.inner_model.process_latent_out(samples.to(torch.float32))
                    state.output = samples.to(
                        device=comfy.model_management.intermediate_device(),
                        dtype=comfy.model_management.intermediate_dtype(),
                    )
                updates.append((state, finished))
            return updates

    def close(self):
        if not self.open:
            return
        try:
            comfy.samplers.cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_patcher.cleanup()
            if self.loaded_models is not None:
                comfy.sampler_helpers.cleanup_models({}, self.loaded_models)
        finally:
            self.model_patcher.hook_mode = self.orig_hook_mode
            self.model_patcher.restore_hook_patches()
            self.inner_model = None
            self.loaded_models = None
            self.load_conds = None
            self.capacity = 0
            self.model_options = None
            self.reload_model = False
            self.open = False


class ContinuousBatchCoordinator:
    def __init__(self, model_key, state):
        self.model_key = model_key
        self.session = ContinuousBatchSession(state.model_patcher)
        self.group_key = None
        self.pending = []
        self.active = []
        self.task = None
        self.last_batch_size = None

    async def submit(self, state):
        request = _QueuedRequest(state)
        self.pending.append(request)
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        try:
            return await request.future
        except asyncio.CancelledError:
            request.cancelled = True
            raise

    def _admit(self):
        if not self.pending:
            return
        group_key = self.active[0].state.key() if self.active else self.pending[0].state.key()
        if not self.active:
            if self.group_key is not None and self.group_key != group_key:
                self.session.close()
                self.session = ContinuousBatchSession(self.pending[0].state.model_patcher)
            self.group_key = group_key
        max_batch_size = self.active[0].state.max_batch_size if self.active else self.pending[0].state.max_batch_size
        remaining = []
        admitted = False
        for request in self.pending:
            if request.state.key() != group_key or len(self.active) >= max_batch_size:
                remaining.append(request)
                continue
            if request.cancelled:
                request.state.clear()
            else:
                self.active.append(request)
                admitted = True
        self.pending = remaining
        if admitted and self.session.open:
            self.session.request_model_reload()

    def _drop_cancelled(self):
        kept = []
        for request in self.active:
            if request.cancelled or _is_cancelled(getattr(request.state, "prompt_id", None)):
                request.state.clear()
                if not request.future.done():
                    request.future.set_exception(comfy.model_management.InterruptProcessingException())
            else:
                kept.append(request)
        self.active = kept

    async def _run(self):
        try:
            if self.pending and self.pending[0].state.admission_delay > 0:
                await asyncio.sleep(self.pending[0].state.admission_delay)
            while self.pending or self.active:
                self._admit()
                self._drop_cancelled()
                if not self.active:
                    await asyncio.sleep(0)
                    continue
                current = list(self.active)
                if len(current) != self.last_batch_size:
                    logging.info("Continuous %s batch size: %d", current[0].state.family, len(current))
                    self.last_batch_size = len(current)
                updates = self.session.step([request.state for request in current])
                self.active = []
                finished_any = False
                for request, (state, finished) in zip(current, updates):
                    if request.cancelled:
                        state.clear()
                    elif _is_cancelled(getattr(state, "prompt_id", None)):
                        state.clear()
                        if not request.future.done():
                            request.future.set_exception(comfy.model_management.InterruptProcessingException())
                    elif finished:
                        finished_any = True
                        output = state.output
                        state.clear()
                        if not request.future.done():
                            request.future.set_result(output)
                    else:
                        self.active.append(request)
                if finished_any and self.active:
                    self.session.request_model_reload()
                self._admit()
                self._drop_cancelled()
                if self.pending or self.active:
                    await asyncio.sleep(0)
        except BaseException as error:
            for request in self.active + self.pending:
                request.state.clear()
                if not request.future.done():
                    request.future.set_exception(error)
            self.active.clear()
            self.pending.clear()
        finally:
            self.session.close()
            if _COORDINATORS.get(self.model_key) is self:
                del _COORDINATORS[self.model_key]
            self.task = None


async def sample_euler_continuous(state):
    state.key()
    model_key = state.model_key()
    coordinator = _COORDINATORS.get(model_key)
    if coordinator is None:
        coordinator = ContinuousBatchCoordinator(model_key, state)
        _COORDINATORS[model_key] = coordinator
    return await coordinator.submit(state)


@dataclass
class ContinuousVAEDecodeRequest:
    vae: Any
    samples: torch.Tensor
    max_batch_size: int
    admission_delay: float
    prompt_id: str | None = None
    node_id: str | None = None
    client_id: str | None = None
    progress_registry: Any = None

    def validate(self):
        if type(self.vae) is not comfy.sd.VAE:
            raise ValueError("Continuous VAE decoding requires a core VAE")
        if not torch.is_tensor(self.samples):
            raise ValueError("Continuous VAE decoding requires a tensor latent")
        if self.samples.is_nested:
            raise ValueError("Continuous VAE decoding does not support nested latents")
        if self.samples.layout is not torch.strided:
            raise ValueError("Continuous VAE decoding requires a dense strided latent")
        if self.samples.ndim == 0:
            raise ValueError("Continuous VAE decoding requires a batched latent")
        if self.samples.shape[0] != 1:
            raise ValueError("Continuous VAE decoding requires one latent per request")
        if self.vae.latent_dim == 2:
            if self.samples.ndim != 4:
                raise ValueError("Continuous VAE decoding requires a four-dimensional image latent")
        elif self.vae.latent_dim == 3:
            if self.samples.ndim != 5 or self.samples.shape[2] != 1:
                raise ValueError("Continuous VAE decoding supports only single-frame image latents for three-dimensional VAEs")
        else:
            raise ValueError("Continuous VAE decoding does not support this VAE latent dimensionality")
        if self.max_batch_size < 1:
            raise ValueError("Continuous VAE decoding max batch size must be positive")
        if not math.isfinite(self.admission_delay) or self.admission_delay < 0:
            raise ValueError("Continuous VAE decoding admission delay must be finite and non-negative")

    def key(self):
        self.validate()
        return (
            id(self.vae),
            id(self.vae.patcher),
            self.vae.device,
            self.vae.vae_dtype,
            self.vae.output_device,
            tuple(self.samples.shape[1:]),
            self.samples.dtype,
            self.samples.device,
            self.max_batch_size,
            self.admission_delay,
        )

    def vae_key(self):
        return id(self.vae)

    def clear(self):
        self.vae = None
        self.samples = None
        self.progress_registry = None


@dataclass
class _QueuedVAERequest:
    state: ContinuousVAEDecodeRequest
    future: asyncio.Future = field(init=False)
    cancelled: bool = False

    def __post_init__(self):
        self.future = asyncio.get_running_loop().create_future()


class ContinuousVAECoordinator:
    def __init__(self, vae_key, vae):
        self.vae_key = vae_key
        self.vae = vae
        self.pending = []
        self.task = None
        self.last_batch_size = None

    async def submit(self, state):
        request = _QueuedVAERequest(state)
        self.pending.append(request)
        if self.task is None:
            self.task = contextvars.Context().run(asyncio.create_task, self._run())
        try:
            return await request.future
        except asyncio.CancelledError:
            request.cancelled = True
            raise

    @staticmethod
    def _cancel(request):
        request.state.clear()
        if not request.future.done():
            request.future.set_exception(comfy.model_management.InterruptProcessingException())

    @staticmethod
    def _fail(request, error):
        request.state.clear()
        if not request.future.done():
            request.future.set_exception(error)

    def _drop_cancelled(self):
        remaining = []
        for request in self.pending:
            if request.cancelled or _is_cancelled(request.state.prompt_id):
                self._cancel(request)
            else:
                remaining.append(request)
        self.pending = remaining

    def _take_group(self, group_key, max_batch_size):
        group = []
        remaining = []
        for request in self.pending:
            if request.cancelled or _is_cancelled(request.state.prompt_id):
                self._cancel(request)
            elif request.state.key() == group_key and len(group) < max_batch_size:
                group.append(request)
            else:
                remaining.append(request)
        self.pending = remaining
        return group

    def _decode(self, group):
        representative = group[0].state
        samples = representative.samples if len(group) == 1 else torch.cat([request.state.samples for request in group])
        client_id_token = set_current_client_id(representative.client_id)
        progress_token = set_progress_registry(representative.progress_registry) if representative.progress_registry is not None else None
        try:
            if representative.prompt_id is not None and representative.node_id is not None:
                with CurrentNodeContext(representative.prompt_id, representative.node_id):
                    images = self.vae.decode(samples)
            else:
                images = self.vae.decode(samples)
        finally:
            if progress_token is not None:
                reset_progress_registry(progress_token)
            reset_current_client_id(client_id_token)
        if not torch.is_tensor(images) or images.ndim < 1 or images.shape[0] != len(group):
            raise RuntimeError("Continuous VAE decoder returned an invalid batch shape")
        return images

    def _finish_group(self, group, images):
        for index, request in enumerate(group):
            if request.cancelled or _is_cancelled(request.state.prompt_id):
                self._cancel(request)
                continue
            output = images if len(group) == 1 else images[index:index + 1].clone()
            request.state.clear()
            if not request.future.done():
                request.future.set_result(output)

    async def _run(self):
        error = None
        try:
            while self.pending:
                self._drop_cancelled()
                if not self.pending:
                    continue
                representative = self.pending[0]
                group_key = representative.state.key()
                if representative.state.admission_delay > 0:
                    await asyncio.sleep(representative.state.admission_delay)
                self._drop_cancelled()
                if not any(request is representative for request in self.pending):
                    continue
                group = self._take_group(group_key, representative.state.max_batch_size)
                if not group:
                    continue
                if len(group) != self.last_batch_size:
                    logging.info("Continuous VAE batch size: %d", len(group))
                    self.last_batch_size = len(group)
                try:
                    images = self._decode(group)
                except Exception as decode_error:
                    for request in group:
                        self._fail(request, decode_error)
                    continue
                self._finish_group(group, images)
        except Exception as run_error:
            error = run_error
        finally:
            if error is not None:
                for request in self.pending:
                    self._fail(request, error)
            else:
                for request in self.pending:
                    self._cancel(request)
            self.pending.clear()
            self.vae = None
            if _VAE_COORDINATORS.get(self.vae_key) is self:
                del _VAE_COORDINATORS[self.vae_key]
            self.task = None


async def decode_vae_continuous(state):
    state.key()
    vae_key = state.vae_key()
    coordinator = _VAE_COORDINATORS.get(vae_key)
    if coordinator is None:
        coordinator = ContinuousVAECoordinator(vae_key, state.vae)
        _VAE_COORDINATORS[vae_key] = coordinator
    return await coordinator.submit(state)

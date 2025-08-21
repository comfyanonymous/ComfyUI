from __future__ import annotations
from typing import TYPE_CHECKING, Union
from comfy_api.latest import io, ComfyExtension
import comfy.patcher_extension
import logging
import torch
import comfy.model_patcher
if TYPE_CHECKING:
    from uuid import UUID


def easycache_forward_wrapper(executor, *args, **kwargs):
    # get values from args
    x: torch.Tensor = args[0]
    transformer_options: dict[str] = args[-1]
    easycache: EasyCacheHolder = transformer_options["easycache"]
    sigmas = transformer_options["sigmas"]
    uuids = transformer_options["uuids"]
    if sigmas is not None and easycache.is_past_end_timestep(sigmas):
        return executor(*args, **kwargs)
    # prepare next x_prev
    has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
    next_x_prev = x
    input_change = None
    do_easycache = easycache.should_do_easycache(sigmas)
    if do_easycache:
        # if first cond marked this step for skipping, skip it and use appropriate cached values
        if easycache.skip_current_step:
            return easycache.apply_cache_diff(x, uuids)
        if easycache.initial_step:
            easycache.first_cond_uuid = uuids[0]
            has_first_cond_uuid = easycache.has_first_cond_uuid(uuids)
            easycache.initial_step = False
        if has_first_cond_uuid:
            if easycache.has_x_prev_subsampled():
                input_change = (easycache.subsample(x, uuids, clone=False) - easycache.x_prev_subsampled).flatten().abs().mean()
            if easycache.has_output_prev_norm() and easycache.has_relative_transformation_rate():
                approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
                easycache.cumulative_change_rate += approx_output_change_rate
                if easycache.cumulative_change_rate < easycache.reuse_threshold:
                    if easycache.verbose:
                        logging.info(f"EasyCache [verbose] - skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    # other conds should also skip this step, and instead use their cached values
                    easycache.skip_current_step = True
                    return easycache.apply_cache_diff(x, uuids)
                else:
                    if easycache.verbose:
                        logging.info(f"EasyCache [verbose] - NOT skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    easycache.cumulative_change_rate = 0.0

    output: torch.Tensor = executor(*args, **kwargs)
    if has_first_cond_uuid and easycache.has_output_prev_norm():
        output_change = (easycache.subsample(output, uuids, clone=False) - easycache.output_prev_subsampled).flatten().abs().mean()
        if easycache.verbose:
            output_change_rate = output_change / easycache.output_prev_norm
            easycache.output_change_rates.append(output_change_rate.item())
        if easycache.has_relative_transformation_rate():
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
            easycache.approx_output_change_rates.append(approx_output_change_rate.item())
            if easycache.verbose:
                logging.info(f"EasyCache [verbose] - approx_output_change_rate: {approx_output_change_rate}")
        if input_change is not None:
            easycache.relative_transformation_rate = output_change / input_change
        if easycache.verbose:
            logging.info(f"EasyCache [verbose] - output_change_rate: {output_change_rate}")
    # TODO: allow cache_diff to be offloaded
    easycache.update_cache_diff(output, next_x_prev, uuids)
    if has_first_cond_uuid:
        easycache.x_prev_subsampled = easycache.subsample(next_x_prev, uuids)
        easycache.output_prev_subsampled = easycache.subsample(output, uuids)
        easycache.output_prev_norm = output.flatten().abs().mean()
        if easycache.verbose:
            logging.info(f"EasyCache [verbose] - x_prev_subsampled: {easycache.x_prev_subsampled.shape}")
    return output

def super_easycache_predict_noise_wrapper(executor, *args, **kwargs):
    # get values from args
    x: torch.Tensor = args[0]
    timestep: float = args[1]
    model_options: dict[str] = args[2]
    easycache: SuperEasyCacheHolder = model_options["transformer_options"]["easycache"]
    if easycache.is_past_end_timestep(timestep):
        return executor(*args, **kwargs)
    # prepare next x_prev
    next_x_prev = x
    input_change = None
    do_easycache = easycache.should_do_easycache(timestep)
    if do_easycache:
        if easycache.has_x_prev_subsampled():
            if easycache.has_x_prev_subsampled():
                input_change = (easycache.subsample(x, clone=False) - easycache.x_prev_subsampled).flatten().abs().mean()
            if easycache.has_output_prev_norm() and easycache.has_relative_transformation_rate():
                approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
                easycache.cumulative_change_rate += approx_output_change_rate
                if easycache.cumulative_change_rate < easycache.reuse_threshold:
                    if easycache.verbose:
                        logging.info(f"EasyCache [verbose] - skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    # other conds should also skip this step, and instead use their cached values
                    easycache.skip_current_step = True
                    return easycache.apply_cache_diff(x)
                else:
                    if easycache.verbose:
                        logging.info(f"EasyCache [verbose] - NOT skipping step; cumulative_change_rate: {easycache.cumulative_change_rate}, reuse_threshold: {easycache.reuse_threshold}")
                    easycache.cumulative_change_rate = 0.0
    output: torch.Tensor = executor(*args, **kwargs)
    if easycache.has_output_prev_norm():
        output_change = (easycache.subsample(output, clone=False) - easycache.output_prev_subsampled).flatten().abs().mean()
        if easycache.verbose:
            output_change_rate = output_change / easycache.output_prev_norm
            easycache.output_change_rates.append(output_change_rate.item())
        if easycache.has_relative_transformation_rate():
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
            easycache.approx_output_change_rates.append(approx_output_change_rate.item())
            if easycache.verbose:
                logging.info(f"EasyCache [verbose] - approx_output_change_rate: {approx_output_change_rate}")
        if input_change is not None:
            easycache.relative_transformation_rate = output_change / input_change
        if easycache.verbose:
            logging.info(f"EasyCache [verbose] - output_change_rate: {output_change_rate}")
    # TODO: allow cache_diff to be offloaded
    easycache.update_cache_diff(output, next_x_prev)
    easycache.x_prev_subsampled = easycache.subsample(next_x_prev)
    easycache.output_prev_subsampled = easycache.subsample(output)
    easycache.output_prev_norm = output.flatten().abs().mean()
    if easycache.verbose:
        logging.info(f"EasyCache [verbose] - x_prev_subsampled: {easycache.x_prev_subsampled.shape}")
    return output

def easycache_calc_cond_batch_wrapper(executor, *args, **kwargs):
    model_options = args[-1]
    easycache: EasyCacheHolder = model_options["transformer_options"]["easycache"]
    easycache.skip_current_step = False
    # TODO: check if first_cond_uuid is active at this timestep; otherwise, EasyCache needs to be partially reset
    return executor(*args, **kwargs)

def easycache_sample_wrapper(executor, *args, **kwargs):
    """
    This OUTER_SAMPLE wrapper makes sure easycache is prepped for current run, and all memory usage is cleared at the end.
    """
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        # clone and prepare timesteps
        guider.model_options["transformer_options"]["easycache"] = guider.model_options["transformer_options"]["easycache"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        return executor(*args, **kwargs)
    finally:
        easycache: Union[EasyCacheHolder, SuperEasyCacheHolder] = guider.model_options['transformer_options']['easycache']
        output_change_rates = easycache.output_change_rates
        approx_output_change_rates = easycache.approx_output_change_rates
        if easycache.verbose:
            logging.info(f"{easycache.name} [verbose] - output_change_rates {len(output_change_rates)}: {output_change_rates}")
            logging.info(f"{easycache.name} [verbose] - approx_output_change_rates {len(approx_output_change_rates)}: {approx_output_change_rates}")
        total_steps = len(args[3])-1
        logging.info(f"{easycache.name} - skipped {easycache.total_steps_skipped}/{total_steps} steps ({total_steps/(total_steps-easycache.total_steps_skipped):.2f}x speedup).")
        easycache.reset()
        guider.model_options = orig_model_options


class EasyCacheHolder:
    def __init__(self, reuse_threshold: float, start_percent: float, end_percent: float, subsample_factor: int, offload_cache_diff: bool, verbose: bool=False):
        self.name = "EasyCache"
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.offload_cache_diff = offload_cache_diff
        self.verbose = verbose
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.skip_current_step = False
        # cache values
        self.first_cond_uuid = None
        self.x_prev_subsampled: torch.Tensor = None
        self.output_prev_subsampled: torch.Tensor = None
        self.output_prev_norm: torch.Tensor = None
        self.uuid_cache_diffs: dict[UUID, torch.Tensor] = {}
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.total_steps_skipped = 0

    def is_past_end_timestep(self, timestep: float) -> bool:
        return not (timestep[0] > self.end_t).item()

    def should_do_easycache(self, timestep: float) -> bool:
        return (timestep[0] <= self.start_t).item()

    def has_x_prev_subsampled(self) -> bool:
        return self.x_prev_subsampled is not None

    def has_output_prev_subsampled(self) -> bool:
        return self.output_prev_subsampled is not None

    def has_output_prev_norm(self) -> bool:
        return self.output_prev_norm is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def subsample(self, x: torch.Tensor, uuids: list[UUID], clone: bool = True) -> torch.Tensor:
        batch_offset = x.shape[0] // len(uuids)
        uuid_idx = uuids.index(self.first_cond_uuid)
        if self.subsample_factor > 1:
            to_return = x[uuid_idx*batch_offset:(uuid_idx+1)*batch_offset, ..., ::self.subsample_factor, ::self.subsample_factor]
            if clone:
                return to_return.clone()
            return to_return
        to_return = x[uuid_idx*batch_offset:(uuid_idx+1)*batch_offset, ...]
        if clone:
            return to_return.clone()
        return to_return

    def apply_cache_diff(self, x: torch.Tensor, uuids: list[UUID]):
        if self.first_cond_uuid in uuids:
            self.total_steps_skipped += 1
        batch_offset = x.shape[0] // len(uuids)
        for i, uuid in enumerate(uuids):
            x[i*batch_offset:(i+1)*batch_offset, ...] += self.uuid_cache_diffs[uuid].to(x.device)
        return x

    def update_cache_diff(self, output: torch.Tensor, x: torch.Tensor, uuids: list[UUID]):
        diff = output - x
        batch_offset = diff.shape[0] // len(uuids)
        for i, uuid in enumerate(uuids):
            self.uuid_cache_diffs[uuid] = diff[i*batch_offset:(i+1)*batch_offset, ...]

    def has_first_cond_uuid(self, uuids: list[UUID]) -> bool:
        return self.first_cond_uuid in uuids

    def reset(self):
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.skip_current_step = False
        self.output_change_rates = []
        self.first_cond_uuid = None
        del self.x_prev_subsampled
        self.x_prev_subsampled = None
        del self.output_prev_subsampled
        self.output_prev_subsampled = None
        del self.output_prev_norm
        self.output_prev_norm = None
        del self.uuid_cache_diffs
        self.uuid_cache_diffs = {}
        self.total_steps_skipped = 0
        return self

    def clone(self):
        return EasyCacheHolder(self.reuse_threshold, self.start_percent, self.end_percent, self.subsample_factor, self.offload_cache_diff, self.verbose)


class EasyCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EasyCache",
            display_name="EasyCache",
            description="Native EasyCache implementation.",
            category="advanced/debug/model",
            inputs=[
                io.Model.Input("model", tooltip="The model to add EasyCache to."),
                io.Float.Input("reuse_threshold", min=0.0, default=0.2, max=3.0, step=0.01, tooltip="The threshold for reusing cached steps."),
                io.Float.Input("start_percent", min=0.0, default=0.15, max=1.0, step=0.01, tooltip="The relative sampling step to begin use of EasyCache."),
                io.Float.Input("end_percent", min=0.0, default=0.95, max=1.0, step=0.01, tooltip="The relative sampling step to end use of EasyCache."),
                io.Int.Input("subsample_factor", min=1, default=8, max=128, step=1, tooltip="The factor to subsample latents to cache by."),
                io.Boolean.Input("verbose", default=False, tooltip="Whether to log verbose information."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with EasyCache."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, reuse_threshold: float, start_percent: float, end_percent: float, subsample_factor: int, verbose: bool) -> io.NodeOutput:
        model = model.clone()
        model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(reuse_threshold, start_percent, end_percent, subsample_factor, offload_cache_diff=False, verbose=verbose)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, "easycache", easycache_calc_cond_batch_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
        return io.NodeOutput(model)


class SuperEasyCacheHolder:
    def __init__(self, reuse_threshold: float, start_percent: float, end_percent: float, subsample_factor: int, offload_cache_diff: bool, verbose: bool=False):
        self.name = "SuperEasyCache"
        self.reuse_threshold = reuse_threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.subsample_factor = subsample_factor
        self.offload_cache_diff = offload_cache_diff
        self.verbose = verbose
        # timestep values
        self.start_t = 0.0
        self.end_t = 0.0
        # control values
        self.relative_transformation_rate: float = None
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        # cache values
        self.x_prev_subsampled: torch.Tensor = None
        self.output_prev_subsampled: torch.Tensor = None
        self.output_prev_norm: torch.Tensor = None
        self.cache_diff: torch.Tensor = None
        self.output_change_rates = []
        self.approx_output_change_rates = []
        self.total_steps_skipped = 0

    def has_cache_diff(self) -> bool:
        return self.cache_diff is not None

    def is_past_end_timestep(self, timestep: float) -> bool:
        return not (timestep[0] > self.end_t).item()

    def should_do_easycache(self, timestep: float) -> bool:
        return (timestep[0] <= self.start_t).item()

    def has_x_prev_subsampled(self) -> bool:
        return self.x_prev_subsampled is not None

    def has_output_prev_subsampled(self) -> bool:
        return self.output_prev_subsampled is not None

    def has_output_prev_norm(self) -> bool:
        return self.output_prev_norm is not None

    def has_relative_transformation_rate(self) -> bool:
        return self.relative_transformation_rate is not None

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def subsample(self, x: torch.Tensor, clone: bool = True) -> torch.Tensor:
        if self.subsample_factor > 1:
            to_return = x[..., ::self.subsample_factor, ::self.subsample_factor]
            if clone:
                return to_return.clone()
            return to_return
        if clone:
            return x.clone()
        return x

    def apply_cache_diff(self, x: torch.Tensor):
        self.total_steps_skipped += 1
        return x + self.cache_diff.to(x.device)

    def update_cache_diff(self, output: torch.Tensor, x: torch.Tensor):
        self.cache_diff = output - x

    def reset(self):
        self.relative_transformation_rate = 0.0
        self.cumulative_change_rate = 0.0
        self.initial_step = True
        self.output_change_rates = []
        self.approx_output_change_rates = []
        del self.cache_diff
        self.cache_diff = None
        self.total_steps_skipped = 0
        return self

    def clone(self):
        return SuperEasyCacheHolder(self.reuse_threshold, self.start_percent, self.end_percent, self.subsample_factor, self.offload_cache_diff, self.verbose)

class SuperEasyCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SuperEasyCache",
            display_name="Super EasyCache",
            description="Native SuperEasyCache implementation.",
            category="advanced/debug/model",
            inputs=[
                io.Model.Input("model", tooltip="The model to add SuperEasyCache to."),
                io.Float.Input("reuse_threshold", min=0.0, default=0.2, max=3.0, step=0.01, tooltip="The threshold for reusing cached steps."),
                io.Float.Input("start_percent", min=0.0, default=0.15, max=1.0, step=0.01, tooltip="The relative sampling step to begin use of EasyCache."),
                io.Float.Input("end_percent", min=0.0, default=0.95, max=1.0, step=0.01, tooltip="The relative sampling step to end use of EasyCache."),
                io.Int.Input("subsample_factor", min=1, default=8, max=128, step=1, tooltip="The factor to subsample latents to cache by."),
                io.Boolean.Input("verbose", default=False, tooltip="Whether to log verbose information."),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with SuperEasyCache."),
            ],
        )

    @classmethod
    def execute(cls, model: io.Model.Type, reuse_threshold: float, start_percent: float, end_percent: float, subsample_factor: int, verbose: bool) -> io.NodeOutput:
        model = model.clone()
        model.model_options["transformer_options"]["easycache"] = SuperEasyCacheHolder(reuse_threshold, start_percent, end_percent, subsample_factor, offload_cache_diff=False, verbose=verbose)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
        model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, "easycache", super_easycache_predict_noise_wrapper)
        return io.NodeOutput(model)


class EasyCacheExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            EasyCacheNode,
            SuperEasyCacheNode,
        ]

def comfy_entrypoint():
    return EasyCacheExtension()

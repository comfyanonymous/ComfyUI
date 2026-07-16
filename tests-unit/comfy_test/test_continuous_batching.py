import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import comfy.conds
import comfy.model_base
import comfy.patcher_extension
import comfy.supported_models
from comfy.continuous_batching import (
    FAMILY_ANIMA,
    FAMILY_SD15,
    FAMILY_SDXL,
    ContinuousBatchCoordinator,
    ContinuousBatchSession,
    _batch_cfg_predictions,
    _batch_euler_updates,
    _cfg_branches,
    _conditioning_structure,
    _PreparedConditioning,
    _processed_conditioning_signature,
    _validate_conditioning,
    _validate_model_extensions,
    _validate_model_family,
    cfg_combine,
    euler_step,
)
from comfy_execution.graph import DynamicPrompt
from comfy_execution.progress import ProgressRegistry, get_progress_state, reset_progress_state
from comfy_execution.utils import get_current_client_id, get_executing_context, reset_current_client_id, set_current_client_id


class FakeState:
    def __init__(self, name, steps, max_batch_size=2, admission_delay=0.0):
        self.name = name
        self.steps = steps
        self.max_batch_size = max_batch_size
        self.admission_delay = admission_delay
        self.model_patcher = object()
        self.family = "test"
        self.output = None
        self.cleared = False

    def clear(self):
        self.cleared = True

    def key(self):
        return "group"


class FakeSession:
    def __init__(self, fail=False, on_first_step=None):
        self.batches = []
        self.closed = 0
        self.open = False
        self.fail = fail
        self.on_first_step = on_first_step
        self.reload_requests = 0

    def step(self, states):
        self.open = True
        self.batches.append([state.name for state in states])
        if len(self.batches) == 1 and self.on_first_step is not None:
            self.on_first_step()
        if self.fail:
            raise RuntimeError("denoiser failed")
        updates = []
        for state in states:
            state.steps -= 1
            finished = state.steps == 0
            if finished:
                state.output = state.name + " output"
            updates.append((state, finished))
        return updates

    def close(self):
        self.closed += 1
        self.open = False

    def request_model_reload(self):
        self.reload_requests += 1


def _model_patcher(model):
    return SimpleNamespace(model=model)


def _bare_model(model_type, config_type, in_channels=4, latent_channels=4, concat_keys=()):
    model = object.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.model_config = object.__new__(config_type)
    model.diffusion_model = SimpleNamespace(in_channels=in_channels)
    model.latent_format = SimpleNamespace(latent_channels=latent_channels)
    model.concat_keys = concat_keys
    return model


def _extension_patcher(model_options=None, callbacks=None, additional_models=None, wrappers=None):
    return SimpleNamespace(
        model_options=model_options or {},
        callbacks=callbacks or {},
        additional_models=additional_models or {},
        wrappers=wrappers or {},
    )


def _processed_cond(conditioning):
    return SimpleNamespace(area=None, control=None, patches=None, hooks=None, conditioning=conditioning)


def _batch_cond(x, length, marker, uuid):
    cond = _processed_cond({"c_crossattn": comfy.conds.CONDCrossAttn(torch.full((1, length, 4), marker))})
    cond.input_x = x
    cond.uuid = uuid
    return cond


def _batch_state(sigma, cfg, negative_marker, positive_marker):
    x = torch.zeros(1, 4, 2, 2)
    negative = _batch_cond(x, 77, negative_marker, f"negative-{negative_marker}")
    positive = _batch_cond(x, 154, positive_marker, f"positive-{positive_marker}")
    return SimpleNamespace(
        family=FAMILY_SD15,
        x=x,
        sigmas=torch.tensor([sigma, 0.0]),
        index=0,
        cfg=cfg,
        conds={
            "negative": [negative],
            "positive": [positive],
        },
        processed_conds={
            "negative": _PreparedConditioning(negative.conditioning, negative.uuid, _processed_conditioning_signature(FAMILY_SD15, negative)),
            "positive": _PreparedConditioning(positive.conditioning, positive.uuid, _processed_conditioning_signature(FAMILY_SD15, positive)),
        },
    )


class _RecordingPatcher:
    def __init__(self):
        self.prepared_sigmas = []

    def prepare_state(self, sigmas, model_options):
        self.prepared_sigmas.append(sigmas.clone())

    def apply_hooks(self, hooks):
        return {}


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def apply_model(self, input_x, timestep, **conditioning):
        transformer_options = conditioning["transformer_options"]
        crossattn = conditioning["c_crossattn"]
        self.calls.append((crossattn.shape[1], timestep.clone(), transformer_options))
        markers = crossattn[:, 0, 0].reshape(-1, 1, 1, 1)
        return torch.ones_like(input_x) * markers


def test_euler_and_cfg_match_reference_formulas():
    x = torch.tensor([[[4.0, 2.0]]])
    denoised = torch.tensor([[[1.0, 0.5]]])
    sigma = torch.tensor(2.0)
    sigma_next = torch.tensor(0.75)
    assert torch.equal(euler_step(x, denoised, sigma, sigma_next), x + (x - denoised) / sigma * (sigma_next - sigma))

    cond = torch.tensor([3.0])
    uncond = torch.tensor([1.0])
    assert torch.equal(cfg_combine(cond, uncond, 5.0), torch.tensor([11.0]))
    assert cfg_combine(cond, uncond, 1.0) is cond
    assert _cfg_branches(1.0, {}) == (("positive", 0),)
    assert _cfg_branches(1.0, {"disable_cfg1_optimization": True}) == (("negative", 1), ("positive", 0))
    assert _cfg_branches(5.0, {}) == (("negative", 1), ("positive", 0))


def test_batched_cfg_and_euler_match_per_request_math_for_different_schedules(monkeypatch):
    calls = {"cat": 0, "stack": 0, "addcmul": []}
    original_cat = torch.cat
    original_stack = torch.stack
    original_addcmul = torch.addcmul

    def cat(*args, **kwargs):
        calls["cat"] += 1
        return original_cat(*args, **kwargs)

    def stack(*args, **kwargs):
        calls["stack"] += 1
        return original_stack(*args, **kwargs)

    def addcmul(input, tensor1, tensor2, **kwargs):
        calls["addcmul"].append((tensor2.shape, tensor2.dtype, tensor2.device))
        return original_addcmul(input, tensor1, tensor2, **kwargs)

    monkeypatch.setattr(torch, "cat", cat)
    monkeypatch.setattr(torch, "stack", stack)
    monkeypatch.setattr(torch, "addcmul", addcmul)

    states = [
        SimpleNamespace(x=torch.tensor([[[[4.0, 2.0]]]]), sigmas=torch.tensor([2.0, 0.5, 0.0]), index=0, cfg=2.0),
        SimpleNamespace(x=torch.tensor([[[[3.0, 6.0]]]]), sigmas=torch.tensor([3.0, 1.5, 0.25, 0.0]), index=1, cfg=3.5),
    ]
    branches = [(("negative", 1), ("positive", 0))] * 2
    outputs = [
        {"negative": torch.tensor([[[[1.0, 0.5]]]]), "positive": torch.tensor([[[[2.0, 1.5]]]])},
        {"negative": torch.tensor([[[[0.5, 1.0]]]]), "positive": torch.tensor([[[[1.5, 2.0]]]])},
    ]

    predictions = _batch_cfg_predictions(states, branches, outputs)
    expected_predictions = [cfg_combine(output["positive"], output["negative"], state.cfg) for state, output in zip(states, outputs)]
    updates = _batch_euler_updates(states, predictions)
    expected_updates = [
        euler_step(state.x, prediction, state.sigmas[state.index], state.sigmas[state.index + 1])
        for state, prediction in zip(states, expected_predictions)
    ]

    for actual, expected in zip(predictions, expected_predictions):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(updates, expected_updates):
        torch.testing.assert_close(actual, expected)
    assert calls == {
        "cat": 4,
        "stack": 2,
        "addcmul": [
            (torch.Size([2, 1, 1, 1]), torch.float32, torch.device("cpu")),
            (torch.Size([2, 1, 1, 1]), torch.float32, torch.device("cpu")),
        ],
    }


def test_batched_cfg_preserves_cfg1_positive_output_identity():
    positive = torch.ones(1, 1, 1, 1)
    states = [SimpleNamespace(cfg=1.0), SimpleNamespace(cfg=2.0)]
    branches = [(("negative", 1), ("positive", 0))] * 2
    outputs = [
        {"negative": torch.zeros_like(positive), "positive": positive},
        {"negative": torch.zeros_like(positive), "positive": torch.full_like(positive, 2.0)},
    ]

    predictions = _batch_cfg_predictions(states, branches, outputs)

    assert predictions[0] is positive
    assert torch.equal(predictions[1], torch.full_like(positive, 4.0))


def test_multi_step_runs_callbacks_before_one_batched_euler_update(monkeypatch):
    events = []
    session = ContinuousBatchSession(SimpleNamespace(load_device=torch.device("cpu")))
    states = []
    predictions = []
    for index, sigma_next in enumerate((0.5, 0.25)):
        x = torch.full((1, 1, 1, 1), 2.0 + index)
        states.append(SimpleNamespace(
            x=x,
            sigmas=torch.tensor([1.0 + index, sigma_next, 0.0]),
            index=0,
            callback=lambda *args, index=index: events.append(f"callback-{index}"),
            client_id=None,
            progress_registry=None,
            prompt_id=None,
            node_id=None,
        ))
        predictions.append(torch.ones_like(x))

    monkeypatch.setattr("comfy.continuous_batching.comfy.model_management.cuda_device_context", lambda device: nullcontext())
    monkeypatch.setattr(session, "open_session", lambda requests: None)
    monkeypatch.setattr(session, "ensure_model_loaded", lambda requests: None)
    monkeypatch.setattr(session, "prepare_request", lambda request: None)
    monkeypatch.setattr(session, "predict", lambda requests: predictions)
    original_batched_euler = _batch_euler_updates

    def batched_euler(requests, denoised):
        events.append("batched-euler")
        return original_batched_euler(requests, denoised)

    monkeypatch.setattr("comfy.continuous_batching._batch_euler_updates", batched_euler)

    updates = session.step(states)

    assert events == ["callback-0", "callback-1", "batched-euler"]
    assert updates == [(states[0], False), (states[1], False)]
    assert [state.index for state in states] == [1, 1]


def test_single_request_prediction_uses_standard_sampling_function(monkeypatch):
    expected = torch.ones(1, 2)
    seen = []

    def sampling_function(model, x, sigma, negative, positive, cfg, model_options, seed):
        seen.append((model, x, sigma, negative, positive, cfg, model_options, seed))
        return expected

    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.sampling_function", sampling_function)
    session = ContinuousBatchSession(object())
    session.inner_model = "inner-model"
    session.model_options = {"transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.0])}}
    state = SimpleNamespace(
        x=torch.zeros(1, 2),
        sigmas=torch.tensor([2.0, 0.0]),
        index=0,
        conds={"negative": ["negative"], "positive": ["positive"]},
        cfg=5.0,
        seed=42,
    )

    assert session.predict([state]) == [expected]
    assert seen[0][0] == "inner-model"
    assert torch.equal(seen[0][2], torch.tensor([2.0]))
    assert seen[0][3:6] == (["negative"], ["positive"], 5.0)
    assert torch.equal(seen[0][6]["transformer_options"]["sample_sigmas"], state.sigmas)
    assert torch.equal(session.model_options["transformer_options"]["sample_sigmas"], torch.tensor([1.0, 0.0]))
    assert seen[0][7] == 42


def test_multi_prediction_buckets_positive_154_and_negative_77_for_two_requests(monkeypatch):
    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.get_area_and_mult", lambda *args: pytest.fail("predict reprocessed conditioning"))
    patcher = _RecordingPatcher()
    model = _RecordingModel()
    session = ContinuousBatchSession(patcher)
    session.inner_model = model
    session.model_options = {}
    states = [
        _batch_state(2.0, 2.0, 1.0, 3.0),
        _batch_state(1.0, 3.0, 10.0, 20.0),
    ]

    session.predict(states)

    assert [call[0] for call in model.calls] == [77, 154]
    assert torch.equal(patcher.prepared_sigmas[0], torch.tensor([2.0, 1.0]))
    for _, sigmas, transformer_options in model.calls:
        assert torch.equal(sigmas, torch.tensor([2.0, 1.0]))
        assert torch.equal(transformer_options["sigmas"], sigmas)
    assert model.calls[0][2]["cond_or_uncond"] == [1, 1]
    assert model.calls[1][2]["cond_or_uncond"] == [0, 0]
    assert model.calls[0][2]["uuids"] == ["negative-1.0", "negative-10.0"]
    assert model.calls[1][2]["uuids"] == ["positive-3.0", "positive-20.0"]


def test_multi_prediction_remaps_bucket_outputs_before_cfg():
    session = ContinuousBatchSession(_RecordingPatcher())
    session.inner_model = _RecordingModel()
    session.model_options = {}
    states = [
        _batch_state(2.0, 2.0, 1.0, 3.0),
        _batch_state(1.0, 3.0, 10.0, 20.0),
    ]

    predictions = session.predict(states)

    assert torch.equal(predictions[0], torch.full_like(states[0].x, 5.0))
    assert torch.equal(predictions[1], torch.full_like(states[1].x, 40.0))


def test_prepare_request_processes_conditioning_once_across_predict_steps(monkeypatch):
    get_area_calls = []

    def get_area_and_mult(cond, *args):
        get_area_calls.append(cond.uuid)
        return cond

    monkeypatch.setattr("comfy.continuous_batching.comfy.sampler_helpers.convert_cond", lambda cond: cond)
    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.process_conds", lambda model, noise, conds, *args, **kwargs: conds)
    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.get_area_and_mult", get_area_and_mult)

    patcher = _RecordingPatcher()
    patcher.load_device = torch.device("cpu")
    model = _RecordingModel()
    model.model_sampling = SimpleNamespace(
        sigma_max=torch.tensor(2.0),
        noise_scaling=lambda sigma, noise, latent, max_denoise: noise,
    )
    session = ContinuousBatchSession(patcher)
    session.inner_model = model
    session.model_options = {}

    def state(negative_marker, positive_marker):
        x = torch.zeros(1, 4, 2, 2)
        negative = _batch_cond(x, 77, negative_marker, f"negative-{negative_marker}")
        positive = _batch_cond(x, 154, positive_marker, f"positive-{positive_marker}")
        return SimpleNamespace(
            family=FAMILY_SD15,
            noise=x.clone(),
            latent_image=x.clone(),
            sigmas=torch.tensor([2.0, 1.0, 0.0]),
            positive=[positive],
            negative=[negative],
            seed=1,
            cfg=2.0,
            index=0,
            prepared=False,
            processed_conds=None,
        )

    states = [state(1.0, 3.0), state(10.0, 20.0)]
    for request in states:
        session.prepare_request(request)

    assert get_area_calls == ["positive-3.0", "negative-1.0", "positive-20.0", "negative-10.0"]
    first = session.predict(states)
    for request in states:
        request.index = 1
    second = session.predict(states)

    assert get_area_calls == ["positive-3.0", "negative-1.0", "positive-20.0", "negative-10.0"]
    assert torch.equal(first[0], torch.full_like(states[0].x, 5.0))
    assert torch.equal(first[1], torch.full_like(states[1].x, 30.0))
    assert torch.equal(second[0], first[0])
    assert torch.equal(second[1], first[1])


def test_model_family_validation_accepts_only_plain_sd_contracts():
    sd15 = _bare_model(comfy.model_base.BaseModel, comfy.supported_models.SD15)
    _validate_model_family(FAMILY_SD15, _model_patcher(sd15))

    sd20 = _bare_model(comfy.model_base.BaseModel, comfy.supported_models.SD20)
    with pytest.raises(ValueError, match="standard SD1.5"):
        _validate_model_family(FAMILY_SD15, _model_patcher(sd20))

    inpaint = _bare_model(comfy.model_base.BaseModel, comfy.supported_models.SD15, in_channels=9)
    with pytest.raises(ValueError, match="inpaint or concatenated"):
        _validate_model_family(FAMILY_SD15, _model_patcher(inpaint))

    sdxl = _bare_model(comfy.model_base.SDXL, comfy.supported_models.SDXL)
    refiner = _bare_model(comfy.model_base.SDXLRefiner, comfy.supported_models.SDXLRefiner)
    _validate_model_family(FAMILY_SDXL, _model_patcher(sdxl))
    _validate_model_family(FAMILY_SDXL, _model_patcher(refiner))

    ip2p = _bare_model(comfy.model_base.SDXL_instructpix2pix, comfy.supported_models.SDXL_instructpix2pix, in_channels=8)
    with pytest.raises(ValueError, match="standard SDXL"):
        _validate_model_family(FAMILY_SDXL, _model_patcher(ip2p))


def test_model_extension_validation_is_conservative_for_sd_and_anima():
    for family in (FAMILY_ANIMA, FAMILY_SD15, FAMILY_SDXL):
        _validate_model_extensions(family, _extension_patcher())
    with pytest.raises(ValueError, match="model callbacks"):
        _validate_model_extensions(FAMILY_ANIMA, _extension_patcher(callbacks={"event": {None: [object()]}}))
    with pytest.raises(ValueError, match="additional models"):
        _validate_model_extensions(FAMILY_SD15, _extension_patcher(additional_models={"control": [object()]}))


@pytest.mark.parametrize("family", [FAMILY_ANIMA, FAMILY_SD15, FAMILY_SDXL])
@pytest.mark.parametrize("patch_key", ["patches", "patches_replace"])
def test_model_extension_validation_rejects_transformer_patches_for_every_family(family, patch_key):
    with pytest.raises(ValueError, match="transformer patches"):
        _validate_model_extensions(family, _extension_patcher(model_options={"transformer_options": {patch_key: {"attn": [object()]}}}))


def test_model_extension_validation_allows_anima_attention_backend_override():
    _validate_model_extensions(FAMILY_ANIMA, _extension_patcher(model_options={"transformer_options": {"optimized_attention_override": object()}}))


@pytest.mark.parametrize("family", [FAMILY_ANIMA, FAMILY_SD15, FAMILY_SDXL])
@pytest.mark.parametrize("wrapper_type", [comfy.patcher_extension.WrappersMP.APPLY_MODEL, comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL])
def test_model_extension_validation_rejects_wrappers_for_every_family(family, wrapper_type):
    wrappers = {wrapper_type: {None: [object()]}}
    with pytest.raises(ValueError, match="wrapper"):
        _validate_model_extensions(family, _extension_patcher(wrappers=wrappers))


def test_conditioning_contracts_reject_unsupported_features_and_wrappers():
    raw = [[torch.zeros(1, 77, 768), {}]]
    _validate_conditioning("positive", raw)
    with pytest.raises(ValueError, match="one positive"):
        _validate_conditioning("positive", raw + raw)
    with pytest.raises(ValueError, match="control"):
        _validate_conditioning("positive", [[raw[0][0], {"control": object()}]])

    sd15 = _processed_cond({"c_crossattn": comfy.conds.CONDCrossAttn(torch.zeros(1, 77, 768))})
    _processed_conditioning_signature(FAMILY_SD15, sd15)
    with pytest.raises(ValueError, match="unsupported c_crossattn"):
        _processed_conditioning_signature(FAMILY_SD15, _processed_cond({"c_crossattn": comfy.conds.CONDRegular(torch.zeros(1, 77, 768))}))

    sdxl = _processed_cond({
        "c_crossattn": comfy.conds.CONDCrossAttn(torch.zeros(1, 77, 2048)),
        "y": comfy.conds.CONDRegular(torch.zeros(1, 2816)),
    })
    _processed_conditioning_signature(FAMILY_SDXL, sdxl)
    with pytest.raises(ValueError, match="conditioning keys"):
        _processed_conditioning_signature(FAMILY_SDXL, sd15)


def test_single_request_rejects_unsupported_processed_conditioning_during_prepare(monkeypatch):
    bad_condition = _processed_cond({"c_crossattn": comfy.conds.CONDRegular(torch.zeros(1, 77, 768))})
    monkeypatch.setattr("comfy.continuous_batching.comfy.sampler_helpers.convert_cond", lambda cond: cond)
    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.process_conds", lambda *args, **kwargs: {"positive": [bad_condition], "negative": [bad_condition]})
    monkeypatch.setattr("comfy.continuous_batching.comfy.samplers.get_area_and_mult", lambda cond, *args: cond)

    model_sampling = SimpleNamespace(
        sigma_max=torch.tensor(1.0),
        noise_scaling=lambda sigma, noise, latent, max_denoise: noise,
    )
    session = ContinuousBatchSession(SimpleNamespace(load_device=torch.device("cpu")))
    session.inner_model = SimpleNamespace(model_sampling=model_sampling)
    state = SimpleNamespace(
        family=FAMILY_SD15,
        noise=torch.zeros(1, 4, 2, 2),
        latent_image=torch.zeros(1, 4, 2, 2),
        sigmas=torch.tensor([1.0, 0.0]),
        positive=[object()],
        negative=[object()],
        seed=1,
        prepared=False,
    )

    with pytest.raises(ValueError, match="unsupported c_crossattn conditioning wrapper"):
        session.prepare_request(state)
    assert not state.prepared


def test_anima_conditioning_structure_groups_padding_compatible_prompts():
    short = [[torch.zeros(1, 12, 1024), {"t5xxl_ids": torch.zeros(1, 77)}]]
    longer = [[torch.zeros(1, 40, 1024), {"t5xxl_ids": torch.zeros(1, 200)}]]
    long = [[torch.zeros(1, 12, 1024), {"t5xxl_ids": torch.zeros(1, 513)}]]
    assert _conditioning_structure(FAMILY_ANIMA, short) == _conditioning_structure(FAMILY_ANIMA, longer)
    assert _conditioning_structure(FAMILY_ANIMA, short) != _conditioning_structure(FAMILY_ANIMA, long)


def test_callback_uses_request_progress_and_routing_context():
    request_registry = ProgressRegistry("request", DynamicPrompt({}))
    reset_progress_state("outer", DynamicPrompt({}))
    client_token = set_current_client_id("outer-client")
    seen = []
    state = SimpleNamespace(
        callback=lambda *args: seen.append((get_executing_context(), get_current_client_id(), get_progress_state())),
        client_id="request-client",
        progress_registry=request_registry,
        prompt_id="request",
        node_id="sampler",
        index=0,
        x=torch.zeros(1),
        sigmas=torch.tensor([1.0, 0.0]),
    )
    try:
        ContinuousBatchSession.run_callback(state, torch.zeros(1))
        assert seen[0][0].prompt_id == "request"
        assert seen[0][0].node_id == "sampler"
        assert seen[0][1] == "request-client"
        assert seen[0][2] is request_registry
        assert get_current_client_id() == "outer-client"
        assert get_progress_state().prompt_id == "outer"
    finally:
        reset_current_client_id(client_token)


def test_admits_at_step_boundaries_and_retires_finished_requests():
    async def run():
        first = FakeState("first", 3)
        coordinator = ContinuousBatchCoordinator("key", first)
        second = FakeState("second", 1)
        second_tasks = []
        session = FakeSession(on_first_step=lambda: second_tasks.append(asyncio.create_task(coordinator.submit(second))))
        coordinator.session = session
        first_task = asyncio.create_task(coordinator.submit(first))
        for _ in range(100):
            if second_tasks:
                break
            if first_task.done():
                await first_task
            await asyncio.sleep(0)
        assert second_tasks
        assert await asyncio.gather(first_task, second_tasks[0]) == ["first output", "second output"]
        assert session.batches[0] == ["first"]
        assert ["first", "second"] in session.batches
        assert session.batches[-1] == ["first"]
        assert first.cleared and second.cleared
        assert session.closed == 1

    asyncio.run(run())


def test_failure_clears_requests_and_finishing_request_reloads_survivor():
    async def failure():
        first = FakeState("first", 1)
        second = FakeState("second", 1)
        coordinator = ContinuousBatchCoordinator("key", first)
        session = FakeSession(fail=True)
        coordinator.session = session
        results = await asyncio.gather(coordinator.submit(first), coordinator.submit(second), return_exceptions=True)
        assert all(isinstance(result, RuntimeError) for result in results)
        assert first.cleared and second.cleared
        assert session.closed == 1

    async def residency():
        fast = FakeState("fast", 1)
        slow = FakeState("slow", 2)
        coordinator = ContinuousBatchCoordinator("key", fast)
        session = FakeSession()
        coordinator.session = session
        assert await asyncio.gather(coordinator.submit(fast), coordinator.submit(slow)) == ["fast output", "slow output"]
        assert session.batches == [["fast", "slow"], ["slow"]]
        assert session.reload_requests == 1

    asyncio.run(failure())
    asyncio.run(residency())

"""Regression tests for SeedVR2 conditioning model resolution and RoPE
frequency cast.

Pin two behaviors:

  1. ``_resolve_seedvr2_diffusion_model`` returns the inner diffusion-model
     for the expected ``model.model.diffusion_model`` shape and fails loud
     with a ``RuntimeError`` whose message begins with
     ``_SEEDVR2_INVALID_MODEL_MSG_PREFIX`` for any other shape, including
     the four distinct missing-vs-None subcases of the chain.
  2. ``_apply_rope_freqs_float32_cast`` is idempotent **per-tensor by
     dtype check**, NOT per-instance by sentinel attribute. Every call
     walks the diffusion-model module tree and invokes ``.to(float32)``
     only on tensors whose dtype is not already ``float32``. A cache-by-
     attribute (sentinel) approach is rejected because the sentinel
     would survive ComfyUI's dynamic model unload/reload cycle while
     ``rope.freqs`` itself is restored to the archived dtype, so the
     next call would short-circuit and leave RoPE running in fp16/bf16
     — the exact failure this helper is supposed to prevent. The dtype
     check is self-correcting against any weight-restore lifecycle
     event.

Import isolation: ``comfy.model_management`` is stubbed via direct
``sys.modules`` assignment so importing ``comfy_extras.nodes_seedvr`` does
not trigger GPU/server-side initialization. ``patch.dict`` is intentionally
NOT used here because its snapshot/restore semantics evict transitively
imported third-party modules (e.g. ``torchvision``) on exit, which causes
``torch``'s global op-library Meta-key registrations to double-register on
re-import. Module-level cached import + scoped restore of the four mocked
entries avoids that hazard. See ``_import_nodes_seedvr_isolated``.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


_SENTINEL = object()


def _import_nodes_seedvr_isolated():
    """Stub ``comfy.model_management``, import (or reuse a cached import of)
    ``comfy_extras.nodes_seedvr``, and return ``(module, restore)``.

    ``restore()`` snapshots and restores three in-process import-state
    surfaces:

      1. ``sys.modules["comfy.model_management"]`` — the stubbed module.
      2. ``sys.modules["comfy_extras.nodes_seedvr"]`` — the imported test
         target. If we leave this in ``sys.modules`` after the test, a
         later test importing the real ``comfy_extras.nodes_seedvr`` will
         get our stubbed-``comfy.model_management`` cached version, which
         does not re-resolve against the real ``comfy.model_management``.
      3. ``comfy_extras.nodes_seedvr`` package attribute on the
         ``comfy_extras`` package, mirroring the existing
         ``comfy.model_management`` attribute restore.

    All three are restored verbatim if previously set; deleted on exit
    if previously unset. No global state leaks into later tests.
    """
    prior_comfy_mm = sys.modules.get("comfy.model_management", _SENTINEL)
    prior_comfy_mm_attr = _SENTINEL
    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        prior_comfy_mm_attr = getattr(comfy_pkg, "model_management", _SENTINEL)
    prior_nodes_seedvr_module = sys.modules.get(
        "comfy_extras.nodes_seedvr", _SENTINEL,
    )
    prior_nodes_seedvr_attr = _SENTINEL
    comfy_extras_pkg = sys.modules.get("comfy_extras")
    if comfy_extras_pkg is not None:
        prior_nodes_seedvr_attr = getattr(
            comfy_extras_pkg, "nodes_seedvr", _SENTINEL,
        )

    # ``comfy_extras.nodes_seedvr`` imports ``comfy.sample`` (added in PR
    # #59) which pulls in the full samplers/k_diffusion/model_patcher
    # transitive chain. That chain re-imports ``comfy.model_management``
    # and calls feature-detection predicates like ``xformers_enabled()``
    # in module-init code (``comfy/ldm/modules/attention.py:18``); a bare
    # ``MagicMock()`` returns truthy for those calls and triggers a real
    # ``import xformers`` that fails in the test environment. Pin the
    # boolean-returning predicates to ``False`` so the import chain
    # follows the no-extension path.
    # Configure stub so every ``..._enabled[_*]()`` predicate returns
    # False. The transitive import chain through ``comfy.sample`` → ...
    # invokes several feature-detection predicates at module-init time
    # (``comfy/ldm/modules/attention.py`` ``xformers_enabled()``,
    # ``comfy/ldm/modules/diffusionmodules/model.py``
    # ``xformers_enabled_vae()``, etc.). A bare ``MagicMock()`` returns
    # truthy auto-attrs, which triggers real ``import xformers`` calls
    # that fail in the test environment.
    mock_mm = MagicMock()
    mock_mm.xformers_enabled.return_value = False
    mock_mm.xformers_enabled_vae.return_value = False
    mock_mm.pytorch_attention_enabled.return_value = False
    mock_mm.pytorch_attention_enabled_vae.return_value = False
    mock_mm.sage_attention_enabled.return_value = False
    mock_mm.flash_attention_enabled.return_value = False
    torch_version_parts = torch.version.__version__.split(".")
    mock_mm.torch_version_numeric = (
        int(torch_version_parts[0]),
        int(torch_version_parts[1]),
    )
    mock_mm.WINDOWS = False
    mock_mm.is_intel_xpu.return_value = False
    sys.modules["comfy.model_management"] = mock_mm
    # The transitive import chain reaches code paths that do
    # ``comfy.model_management.<attr>`` (attribute access on the comfy
    # package, not a fresh import). Setting only ``sys.modules`` is not
    # enough — also bind the stub as the package attribute. If the
    # ``comfy`` package isn't imported yet at stub-time (cold first run),
    # importing it now is safe and idempotent.
    if comfy_pkg is None:
        import comfy as _comfy_pkg  # noqa: F401
        comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        setattr(comfy_pkg, "model_management", mock_mm)
    if "comfy_extras.nodes_seedvr" in sys.modules:
        nodes_seedvr = sys.modules["comfy_extras.nodes_seedvr"]
    else:
        nodes_seedvr = importlib.import_module("comfy_extras.nodes_seedvr")

    def _restore():
        # 1. comfy.model_management sys.modules entry
        if prior_comfy_mm is _SENTINEL:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = prior_comfy_mm
        # 2. comfy.model_management package attribute on comfy
        comfy_pkg_now = sys.modules.get("comfy")
        if comfy_pkg_now is not None:
            if prior_comfy_mm_attr is _SENTINEL:
                if hasattr(comfy_pkg_now, "model_management"):
                    delattr(comfy_pkg_now, "model_management")
            else:
                setattr(comfy_pkg_now, "model_management", prior_comfy_mm_attr)
        # 3. comfy_extras.nodes_seedvr sys.modules entry
        if prior_nodes_seedvr_module is _SENTINEL:
            sys.modules.pop("comfy_extras.nodes_seedvr", None)
        else:
            sys.modules["comfy_extras.nodes_seedvr"] = prior_nodes_seedvr_module
        # 4. comfy_extras.nodes_seedvr package attribute on comfy_extras
        comfy_extras_pkg_now = sys.modules.get("comfy_extras")
        if comfy_extras_pkg_now is not None:
            if prior_nodes_seedvr_attr is _SENTINEL:
                if hasattr(comfy_extras_pkg_now, "nodes_seedvr"):
                    delattr(comfy_extras_pkg_now, "nodes_seedvr")
            else:
                setattr(
                    comfy_extras_pkg_now, "nodes_seedvr",
                    prior_nodes_seedvr_attr,
                )

    return nodes_seedvr, _restore


class _Rope(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(4))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = _Rope()


class _DiffusionModel(nn.Module):
    def __init__(
        self,
        n_blocks=3,
        zero_conditioning=False,
        conditioning_dtype=torch.float32,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        if zero_conditioning:
            # Simulates a numz-format DiT-only file loaded via UNETLoader:
            # ``register_buffer`` zero-init at ``comfy/ldm/seedvr/model.py``
            # leaves the buffers at zero when ``load_state_dict`` cannot
            # find ``positive_conditioning`` / ``negative_conditioning``
            # keys in the state_dict. The fail-loud guard at
            # ``SeedVR2Conditioning.execute`` distinguishes this from a
            # properly-baked file by ``abs().sum() == 0`` on both buffers.
            self.register_buffer(
                "positive_conditioning",
                torch.zeros((2, 4), dtype=conditioning_dtype),
            )
            self.register_buffer(
                "negative_conditioning",
                torch.zeros((3, 4), dtype=conditioning_dtype),
            )
        else:
            self.register_buffer(
                "positive_conditioning",
                torch.ones((2, 4), dtype=conditioning_dtype),
            )
            self.register_buffer(
                "negative_conditioning",
                torch.zeros((3, 4), dtype=conditioning_dtype),
            )


class _ModelInner:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_resolve_seedvr2_diffusion_model_returns_inner_when_valid():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        resolved = nodes_seedvr._resolve_seedvr2_diffusion_model(patcher)
        assert resolved is diffusion_model
    finally:
        restore()


def test_seedvr2_conditioning_schema_exposes_model_passthrough_output():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        schema = nodes_seedvr.SeedVR2Conditioning.define_schema()
        assert [input_item.id for input_item in schema.inputs] == [
            "model",
            "vae_conditioning",
        ]
        assert schema.inputs[1].display_name == "LATENT"
        assert [output.display_name for output in schema.outputs] == [
            "model",
            "positive",
            "negative",
            "latent",
        ]
    finally:
        restore()


def test_seedvr2_conditioning_returns_packed_input_latent_deterministically():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        patcher = _ModelPatcher(diffusion_model)
        samples = torch.arange(1, 25, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
        vae_conditioning = {"samples": samples}

        _, first_positive, first_negative, first_latent = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )
        _, second_positive, second_negative, second_latent = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher,
                vae_conditioning,
            )
        )

        expected_latent = samples.reshape(1, 6, 2, 2)
        channel_last = samples.movedim(1, -1).contiguous()
        expected_condition = torch.cat(
            [
                channel_last,
                torch.ones((*channel_last.shape[:-1], 1)),
            ],
            dim=-1,
        ).movedim(-1, 1).reshape(1, 9, 2, 2)

        assert torch.equal(first_latent["samples"], expected_latent)
        assert torch.equal(second_latent["samples"], expected_latent)
        assert torch.equal(
            first_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_positive[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            first_negative[0][1]["condition"],
            expected_condition,
        )
        assert torch.equal(
            second_negative[0][1]["condition"],
            expected_condition,
        )
    finally:
        restore()


def test_resolve_seedvr2_diffusion_model_raises_runtime_error_with_specific_prefix():
    """Pin all four failure modes of the resolver chain to the same error
    prefix and to message text that distinguishes 'attribute missing'
    from 'attribute present but None'. The four modes:

      mode 1: input has no 'model' attribute
      mode 2: input.model is None
      mode 3: 'model.model' has no 'diffusion_model' attribute
      mode 4: 'model.model.diffusion_model' is None
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        # Mode 1: model has no 'model' attribute at all.
        class _NoModelAttr:
            pass

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoModelAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'model' attribute" in msg

        # Mode 2: model.model exists but is None (must not be conflated
        # with "no 'model' attribute").
        class _ModelIsNone:
            def __init__(self):
                self.model = None

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_ModelIsNone())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "input.model is None" in msg

        # Mode 3: model.model exists, has no 'diffusion_model' attribute.
        class _NoDiffusionAttr:
            def __init__(self):
                self.model = object()

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_NoDiffusionAttr())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "no 'diffusion_model' attribute" in msg

        # Mode 4: model.model.diffusion_model exists but is None (must not
        # be conflated with "no 'diffusion_model' attribute").
        class _DiffusionIsNoneInner:
            def __init__(self):
                self.diffusion_model = None

        class _DiffusionIsNone:
            def __init__(self):
                self.model = _DiffusionIsNoneInner()

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr._resolve_seedvr2_diffusion_model(_DiffusionIsNone())
        msg = str(excinfo.value)
        assert msg.startswith(nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX)
        assert "'model.model.diffusion_model' is None" in msg
    finally:
        restore()


def test_apply_rope_freqs_float32_cast_idempotent_on_unchanged_dtype():
    """Calling the helper twice on a model whose rope.freqs is already
    float32 must NOT mutate the tensor identity or contents — the dtype
    check on every nested module short-circuits the .to() call when the
    tensor is already in float32.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()

        # Starting dtype is non-float32 so the first call has work to do.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        first_call_data_ids = []
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32
                first_call_data_ids.append(id(module.rope.freqs.data))

        # Second call on the same already-float32 model: every per-tensor
        # dtype check sees float32 and skips the .to() call. Tensor data
        # identity must be preserved (no re-allocation).
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module, prior_id in zip(
            (m for m in diffusion_model.modules()
             if hasattr(m, "rope") and hasattr(m.rope, "freqs")),
            first_call_data_ids,
            strict=True,
        ):
            assert module.rope.freqs.data.dtype == torch.float32
            assert id(module.rope.freqs.data) == prior_id, (
                "Already-float32 rope.freqs must not be re-allocated on "
                "subsequent calls; the per-tensor dtype check must skip the "
                ".to(float32) call when the tensor is already in float32."
            )
    finally:
        restore()


def test_apply_rope_freqs_float32_cast_recovers_after_dtype_reset():
    """After a model unload/reload that restores rope.freqs from an
    archived non-float32 dtype, the next call must re-cast to float32.
    A bool-sentinel cache approach would short-circuit here and leave
    RoPE running in fp16/bf16.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel()
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # First call casts to float32.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32

        # Simulate a Comfy dynamic unload/reload that restores rope.freqs
        # to the archived (non-float32) dtype.
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                module.rope.freqs.data = module.rope.freqs.data.to(torch.float64)

        # Second call must detect the dtype regression and re-cast.
        nodes_seedvr._apply_rope_freqs_float32_cast(diffusion_model)
        for module in diffusion_model.modules():
            if hasattr(module, "rope") and hasattr(module.rope, "freqs"):
                assert module.rope.freqs.data.dtype == torch.float32, (
                    "After a model unload/reload that resets rope.freqs to "
                    "non-float32, the next _apply_rope_freqs_float32_cast "
                    "call MUST re-cast to float32. A bool-sentinel cache "
                    "would have short-circuited here."
                )
    finally:
        restore()


# ---------------------------------------------------------------------------
# Fail-loud guard: zero-valued conditioning buffers
# ---------------------------------------------------------------------------


def test_seedvr2_conditioning_fails_loud_on_zero_buffers():
    """A SeedVR2 model whose ``positive_conditioning`` AND
    ``negative_conditioning`` buffers are both zero-valued is an
    unrecoverable load state — a numz-format DiT-only ``.safetensors``
    file was loaded via ``UNETLoader`` without the SeedVR2 conditioning
    keys baked in. ``SeedVR2Conditioning.execute`` must raise
    ``RuntimeError`` carrying the standard SeedVR2 invalid-model prefix
    instead of letting the diffusion sampler run on null prompt
    conditioning (which silently produces wrong output).
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel(zero_conditioning=True)
        patcher = _ModelPatcher(diffusion_model)
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )

        message = str(excinfo.value)
        assert message.startswith(
            nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX
        ), (
            "Fail-loud message must use the standard "
            "_SEEDVR2_INVALID_MODEL_MSG_PREFIX so callers/log scrapers "
            f"can match it. Got: {message!r}"
        )
        assert "positive_conditioning" in message
        assert "negative_conditioning" in message
    finally:
        restore()


def test_seedvr2_conditioning_fails_loud_on_fp8_zero_buffers():
    """The zero-buffer sentinel must reduce fp8 conditioning tensors
    without hitting PyTorch's unsupported float8 reductions.
    """
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        pytest.skip("torch build does not expose float8_e4m3fn")

    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel(
            zero_conditioning=True,
            conditioning_dtype=fp8_dtype,
        )
        patcher = _ModelPatcher(diffusion_model)
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )

        message = str(excinfo.value)
        assert message.startswith(
            nodes_seedvr._SEEDVR2_INVALID_MODEL_MSG_PREFIX
        )
        assert "zero-valued" in message
    finally:
        restore()


def test_seedvr2_conditioning_does_not_fire_on_partial_zero_buffers():
    """The guard checks BOTH buffers together: a model with zero
    ``negative_conditioning`` but non-zero ``positive_conditioning``
    (the existing baseline mock fixture) must NOT trigger the fail-loud
    path. This pins the AND-gating semantic and prevents a future
    regression to OR-gating from rejecting valid bundled checkpoints
    where one buffer happens to be all-zeros.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        # Baseline _DiffusionModel has positive=ones, negative=zeros.
        diffusion_model = _DiffusionModel(zero_conditioning=False)
        patcher = _ModelPatcher(diffusion_model)
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        # Should not raise.
        passthrough_model, positive, negative, latent = (
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )
        )
        assert positive[0][0].shape == (1, 2, 4)
        assert negative[0][0].shape == (1, 3, 4)
        assert passthrough_model is patcher
    finally:
        restore()


def test_seedvr2_conditioning_fail_loud_never_exposes_safetensors_path():
    """The fail-loud message must not expose local model paths from
    ``cached_patcher_init``. Public runtime errors should describe the
    invalid SeedVR2 contract without making filesystem paths part of the
    public behavior contract.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel(zero_conditioning=True)
        patcher = _ModelPatcher(diffusion_model)
        # Mimic the ``cached_patcher_init`` shape comfy.sd attaches.
        patcher.cached_patcher_init = (
            object(),  # function reference
            ("/some/models/diffusion_models/seedvr2_ema_7b_fp16.safetensors",),
        )
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )

        message = str(excinfo.value)
        assert "/some/models/diffusion_models" not in message
        assert "seedvr2_ema_7b_fp16.safetensors" not in message
        assert "Source file:" not in message
        assert "positive_conditioning" in message
        assert "negative_conditioning" in message
    finally:
        restore()


def test_seedvr2_conditioning_fail_loud_falls_back_when_path_unavailable():
    """When ``cached_patcher_init`` is missing or its tuple does not
    contain a ``.safetensors`` path, the fail-loud message still
    delivers the actionable diagnostic without leaking ``None`` or
    raising during message formatting.
    """
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        diffusion_model = _DiffusionModel(zero_conditioning=True)
        patcher = _ModelPatcher(diffusion_model)
        # No cached_patcher_init set on the patcher.
        vae_conditioning = {"samples": torch.zeros((1, 2, 1, 1, 1))}

        with pytest.raises(RuntimeError) as excinfo:
            nodes_seedvr.SeedVR2Conditioning.execute(
                patcher, vae_conditioning,
            )
        message = str(excinfo.value)
        assert "Source file:" not in message  # no empty path leak
        assert "Re-bake" in message  # actionable guidance still present
        assert "bf16 keys" not in message
    finally:
        restore()

"""Consolidated SeedVR2 conditioning and refactor regression tests.

Merges the prior test_seedvr2_refactor_nodes.py and
test_seedvr_conditioning_hardening.py modules. Refactor tests use the
top-level comfy_extras.nodes_seedvr import; conditioning-hardening tests
use _import_nodes_seedvr_isolated() for sys.modules isolation when
mocking comfy.model_management.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True


_SENTINEL = object()
_TARGETS = (
    ("comfy.model_management", "comfy"),
    ("comfy_extras.nodes_seedvr", "comfy_extras"),
)


def _import_nodes_seedvr_isolated():
    """Import comfy_extras.nodes_seedvr with comfy.model_management mocked."""
    priors = []
    for mod_name, parent_name in _TARGETS:
        prior_mod = sys.modules.get(mod_name, _SENTINEL)
        parent = sys.modules.get(parent_name)
        attr = mod_name.split(".")[-1]
        prior_attr = (
            getattr(parent, attr, _SENTINEL) if parent is not None else _SENTINEL
        )
        priors.append((mod_name, parent_name, attr, prior_mod, prior_attr))

    mock_mm = MagicMock()
    for fn in (
        "xformers_enabled", "xformers_enabled_vae",
        "pytorch_attention_enabled", "pytorch_attention_enabled_vae",
        "sage_attention_enabled", "flash_attention_enabled",
        "is_intel_xpu",
    ):
        getattr(mock_mm, fn).return_value = False
    tv = torch.version.__version__.split(".")
    mock_mm.torch_version_numeric = (int(tv[0]), int(tv[1]))
    mock_mm.WINDOWS = False
    sys.modules["comfy.model_management"] = mock_mm
    if sys.modules.get("comfy") is None:
        import comfy as _comfy_pkg  # noqa: F401
    comfy_pkg = sys.modules.get("comfy")
    if comfy_pkg is not None:
        setattr(comfy_pkg, "model_management", mock_mm)
    nodes_seedvr = sys.modules.get("comfy_extras.nodes_seedvr") or (
        importlib.import_module("comfy_extras.nodes_seedvr")
    )

    def _restore():
        for mod_name, parent_name, attr, prior_mod, prior_attr in priors:
            if prior_mod is _SENTINEL:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = prior_mod
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            if prior_attr is _SENTINEL:
                if hasattr(parent, attr):
                    delattr(parent, attr)
            else:
                setattr(parent, attr, prior_attr)

    return nodes_seedvr, _restore


class _Rope(nn.Module):
    """Minimal RoPE stub exposing a `freqs` parameter."""
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.zeros(4))


class _Block(nn.Module):
    """Minimal transformer block stub holding a `_Rope`."""
    def __init__(self):
        super().__init__()
        self.rope = _Rope()


class _DiffusionModel(nn.Module):
    """Stub diffusion model with N blocks and pos/neg conditioning buffers."""
    def __init__(self, n_blocks=3, zero_conditioning=False, conditioning_dtype=torch.float32):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])
        pos = torch.zeros if zero_conditioning else torch.ones
        self.register_buffer("positive_conditioning", pos((2, 4), dtype=conditioning_dtype))
        self.register_buffer("negative_conditioning", torch.zeros((3, 4), dtype=conditioning_dtype))


class _ModelInner:
    """Inner model wrapper exposing `.diffusion_model`."""
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


class _ModelPatcher:
    """ModelPatcher stub exposing `.model._ModelInner`."""
    def __init__(self, diffusion_model):
        self.model = _ModelInner(diffusion_model)


def test_seedvr2_conditioning_schema_exposes_model_passthrough_output():
    nodes_seedvr, restore = _import_nodes_seedvr_isolated()
    try:
        schema = nodes_seedvr.SeedVR2Conditioning.define_schema()
        assert [input_item.id for input_item in schema.inputs] == [
            "model",
            "vae_conditioning",
        ]
        assert schema.inputs[1].display_name == "latent"
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


def test_seedvr2_conditioning_fails_loud_on_zero_buffers():
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

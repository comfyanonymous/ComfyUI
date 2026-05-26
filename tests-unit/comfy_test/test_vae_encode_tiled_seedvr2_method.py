"""Unit tests for ``VAE.encode_tiled_seedvr2``: existence with the
SeedVR2 tile-shape signature and delegation through
``comfy.ldm.seedvr.vae.tiled_vae(..., encode=True)`` with one call per
spatial tile.

Mirrors the decode-side method-existence + delegation contract for
``VAE.decode_tiled_seedvr2``; CPU-only via mocks and a
``VideoAutoencoderKLWrapper.__new__`` wrapper stub (no weights, no
GPU).
"""

import inspect
from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402
import nodes as nodes_mod  # noqa: E402


def _make_minimal_seedvr2_vae():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper

    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = 8

    vae.vae_output_dtype = lambda: torch.float32
    vae.process_input = lambda x: x
    return vae


def test_method_exists_with_seedvr2_signature():
    assert hasattr(sd_mod.VAE, "encode_tiled_seedvr2"), (
        "VAE.encode_tiled_seedvr2 must be defined on the VAE class."
    )
    sig = inspect.signature(sd_mod.VAE.encode_tiled_seedvr2)
    params = list(sig.parameters)
    for required in ("self", "pixel_samples", "tile_x", "tile_y",
                     "overlap", "tile_t", "overlap_t"):
        assert required in params, (
            f"VAE.encode_tiled_seedvr2 missing required parameter "
            f"{required!r}; got parameters {params}."
        )


def test_vae_encode_tiled_allows_zero_temporal_controls_and_passes_zero_through():
    input_types = nodes_mod.VAEEncodeTiled.INPUT_TYPES()["required"]
    assert input_types["temporal_size"][1]["min"] == 0
    assert input_types["temporal_overlap"][1]["min"] == 0
    assert "SeedVR2 allows 0" in input_types["temporal_size"][1]["tooltip"]

    class _EncodeRecorder:
        def __init__(self):
            self.calls = []

        def encode_tiled(self, pixels, **kwargs):
            self.calls.append({"shape": tuple(pixels.shape), **kwargs})
            return torch.zeros(1, 16, 1, 8, 8)

    recorder = _EncodeRecorder()
    node = nodes_mod.VAEEncodeTiled()

    output = node.encode(
        recorder,
        torch.zeros(1, 64, 64, 3),
        tile_size=256,
        overlap=64,
        temporal_size=0,
        temporal_overlap=8,
    )

    assert recorder.calls == [
        {
            "shape": (1, 64, 64, 3),
            "tile_x": 256,
            "tile_y": 256,
            "overlap": 64,
            "tile_t": 0,
            "overlap_t": 0,
        }
    ]
    assert torch.equal(output[0]["samples"], torch.zeros(1, 16, 1, 8, 8))


def test_method_routes_through_tiled_vae_encode_true():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))

    tiled_vae_mock = MagicMock(return_value=torch.zeros((1, 16, 2, 8, 8)))

    with patch.object(seedvr_vae_mod, "tiled_vae", tiled_vae_mock):
        vae.encode_tiled_seedvr2(pixel_samples)

    assert tiled_vae_mock.call_count >= 1, (
        f"Expected encode_tiled_seedvr2 to delegate to tiled_vae at "
        f"least once; got {tiled_vae_mock.call_count} calls."
    )
    for call in tiled_vae_mock.call_args_list:
        assert call.kwargs.get("encode") is True, (
            f"Every tiled_vae delegation from encode_tiled_seedvr2 must "
            f"pass encode=True; got kwargs={call.kwargs!r}."
        )


def test_method_sets_wrapper_device_before_tiled_vae():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))
    assert not hasattr(vae.first_stage_model, "device")

    def _assert_device_initialized(*args, **kwargs):
        vae_model = args[1]
        assert vae_model.device == vae.device
        return torch.zeros((1, 16, 2, 8, 8))

    with patch.object(seedvr_vae_mod, "tiled_vae",
                      MagicMock(side_effect=_assert_device_initialized)):
        vae.encode_tiled_seedvr2(pixel_samples)


def test_method_honors_explicit_tile_parameters_over_stale_wrapper_args():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))
    vae.first_stage_model.tiled_args = {
        "tile_size": (17, 19),
        "tile_overlap": (3, 5),
        "temporal_size": 7,
        "temporal_overlap": 2,
        "preserved": "value",
    }

    tiled_vae_mock = MagicMock(return_value=torch.zeros((1, 16, 2, 8, 8)))

    with patch.object(seedvr_vae_mod, "tiled_vae", tiled_vae_mock):
        vae.encode_tiled_seedvr2(
            pixel_samples,
            tile_x=96,
            tile_y=80,
            overlap=12,
            tile_t=11,
            overlap_t=4,
        )

    assert tiled_vae_mock.call_args.kwargs["tile_size"] == (80, 96)
    assert tiled_vae_mock.call_args.kwargs["tile_overlap"] == (12, 12)
    assert tiled_vae_mock.call_args.kwargs["temporal_size"] == 11
    assert tiled_vae_mock.call_args.kwargs["temporal_overlap"] == 4
    assert vae.first_stage_model.tiled_args["preserved"] == "value"


def test_method_uses_explicit_defaults_when_call_omits_tile_parameters():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))
    vae.first_stage_model.tiled_args = {
        "tile_size": (128, 160),
        "tile_overlap": (16, 24),
        "temporal_size": 9,
        "temporal_overlap": 1,
    }

    tiled_vae_mock = MagicMock(return_value=torch.zeros((1, 16, 2, 8, 8)))

    with patch.object(seedvr_vae_mod, "tiled_vae", tiled_vae_mock):
        vae.encode_tiled_seedvr2(pixel_samples)

    assert tiled_vae_mock.call_args.kwargs["tile_size"] == (512, 512)
    assert tiled_vae_mock.call_args.kwargs["tile_overlap"] == (64, 64)
    assert tiled_vae_mock.call_args.kwargs["temporal_size"] == 9999
    assert tiled_vae_mock.call_args.kwargs["temporal_overlap"] == 0
    assert vae.first_stage_model.tiled_args == {
        "tile_size": (128, 160),
        "tile_overlap": (16, 24),
        "temporal_size": 9,
        "temporal_overlap": 1,
    }


def test_method_clamps_overlap_below_tile_size():
    vae = _make_minimal_seedvr2_vae()
    pixel_samples = torch.zeros((1, 3, 8, 64, 64))

    tiled_vae_mock = MagicMock(return_value=torch.zeros((1, 16, 2, 8, 8)))

    with patch.object(seedvr_vae_mod, "tiled_vae", tiled_vae_mock):
        vae.encode_tiled_seedvr2(
            pixel_samples,
            tile_x=64,
            tile_y=48,
            overlap=96,
        )

    assert tiled_vae_mock.call_args.kwargs["tile_overlap"] == (40, 56)

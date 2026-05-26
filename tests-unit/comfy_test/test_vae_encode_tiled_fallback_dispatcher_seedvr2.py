"""Unit tests for the ``VAE.encode`` OOM-fallback dispatcher routing of
SeedVR2 vs non-SeedVR2 3D inputs.

Mirrors the decode-side dispatcher contract in
``test_vae_decode_tiled_dispatcher_seedvr2_4d.py``: the two candidate
methods (``encode_tiled_seedvr2``, ``encode_tiled_3d``) are patched on
the ``VAE`` class, the regular encode is forced to OOM via a stub, and
the test asserts the dispatcher selects the SeedVR2-aware tiler when
``first_stage_model`` is a ``VideoAutoencoderKLWrapper`` while
preserving the generic 3D tiler for non-SeedVR2 inputs.
"""

from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402


def _populate_common_vae_attrs(vae):
    vae.patcher = MagicMock()
    vae.patcher.get_free_memory = MagicMock(return_value=8 * 1024 * 1024 * 1024)
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = True
    vae.extra_1d_channel = None
    vae.upscale_ratio = 8
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None
    vae.not_video = False
    vae.crop_input = False
    vae.pad_channel_value = None

    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_encode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_encode = lambda *a, **k: 1


def _make_seedvr2_vae():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper
    _populate_common_vae_attrs(vae)
    return vae


def _make_non_seedvr2_vae():
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()
    _populate_common_vae_attrs(vae)
    return vae


def _force_regular_encode_oom(*args, **kwargs):
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def test_seedvr2_3d_routes_to_encode_tiled_seedvr2_on_oom():
    vae = _make_seedvr2_vae()
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "encode",
                      side_effect=_force_regular_encode_oom), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected encode_tiled_seedvr2 to be called once for a SeedVR2 3D "
        f"input under OOM fallback; got {seedvr2_call.call_count} calls."
    )
    assert generic_call.call_count == 0, (
        f"encode_tiled_3d must NOT be called for a SeedVR2 input; got "
        f"{generic_call.call_count} calls."
    )


def test_seedvr2_oom_fallback_uses_explicit_seedvr2_tile_defaults():
    vae = _make_seedvr2_vae()
    vae.first_stage_model.tiled_args = {
        "tile_size": (128, 128),
        "tile_overlap": (32, 32),
        "temporal_size": 12,
        "temporal_overlap": 4,
    }
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "encode",
                      side_effect=_force_regular_encode_oom), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True):
        vae.encode(pixel_samples)

    assert seedvr2_call.call_count == 1
    assert seedvr2_call.call_args.kwargs == {
        "tile_x": 256,
        "tile_y": 256,
        "overlap": 64,
    }


def test_oom_fallback_dispatcher_breakdown():
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    seedvr2_vae = _make_seedvr2_vae()
    non_seedvr2_vae = _make_non_seedvr2_vae()
    non_seedvr2_vae.first_stage_model.encode = MagicMock(
        side_effect=_force_regular_encode_oom
    )

    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "encode",
                      side_effect=_force_regular_encode_oom), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        seedvr2_vae.encode(pixel_samples)
        non_seedvr2_vae.encode(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected encode_tiled_seedvr2 called once across SeedVR2 + "
        f"non-SeedVR2 OOM fallbacks; got {seedvr2_call.call_count}."
    )
    assert generic_call.call_count == 1, (
        f"Expected encode_tiled_3d called once across SeedVR2 + non-SeedVR2 "
        f"OOM fallbacks; got {generic_call.call_count}."
    )


def test_non_seedvr2_encode_tiled_3d_default_overlap_is_concrete():
    vae = _make_non_seedvr2_vae()
    vae.downscale_ratio = (lambda a: max(1, a // 4), 8, 8)
    vae.upscale_ratio = (lambda a: a * 4, 8, 8)
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    pixel_samples = torch.zeros((1, 8, 64, 64, 3))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode_tiled(pixel_samples)

    assert generic_call.call_args.kwargs["overlap"] == (1, 64, 64)

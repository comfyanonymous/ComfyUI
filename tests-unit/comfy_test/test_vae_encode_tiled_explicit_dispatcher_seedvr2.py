"""Unit tests for the explicit ``VAE.encode_tiled`` dispatcher routing of
SeedVR2 vs non-SeedVR2 3D inputs.

Mirrors the decode-side dispatcher contract in
``test_vae_decode_tiled_dispatcher_seedvr2_4d.py`` and the encode OOM
fallback contract in ``test_vae_encode_tiled_fallback_dispatcher_seedvr2.py``:
the two candidate methods (``encode_tiled_seedvr2``, ``encode_tiled_3d``)
are patched on the ``VAE`` class, ``encode_tiled`` is invoked directly,
and the test asserts the dispatcher selects the SeedVR2-aware tiler when
``first_stage_model`` is a ``VideoAutoencoderKLWrapper`` while preserving
the generic 3D tiler for non-SeedVR2 inputs.
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
    vae.upscale_ratio = [lambda x: x]
    vae.upscale_index_formula = None
    vae.output_channels = 3
    vae.latent_channels = 16
    vae.latent_dim = 3
    vae.downscale_ratio = [lambda x: x]
    vae.downscale_index_formula = None
    vae.not_video = False
    vae.crop_input = False
    vae.pad_channel_value = None

    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_encode = lambda: 8
    vae.vae_encode_crop_pixels = lambda x: x
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


def test_explicit_encode_tiled_seedvr2_3d_routes_to_seedvr2_tiler():
    vae = _make_seedvr2_vae()
    pixel_samples = torch.zeros((1, 64, 64, 3))

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        vae.encode_tiled(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected encode_tiled_seedvr2 to be called once for a SeedVR2 3D "
        f"input via explicit encode_tiled; got {seedvr2_call.call_count} calls."
    )
    assert generic_call.call_count == 0, (
        f"encode_tiled_3d must NOT be called for a SeedVR2 input via explicit "
        f"encode_tiled; got {generic_call.call_count} calls."
    )


def test_explicit_encode_tiled_dispatcher_breakdown():
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))
    generic_call = MagicMock(return_value=torch.zeros(1, 16, 2, 8, 8))

    seedvr2_vae = _make_seedvr2_vae()
    non_seedvr2_vae = _make_non_seedvr2_vae()

    pixel_samples = torch.zeros((1, 64, 64, 3))

    with patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.VAE, "encode_tiled_seedvr2", seedvr2_call,
                      create=True), \
         patch.object(sd_mod.VAE, "encode_tiled_3d", generic_call):
        seedvr2_vae.encode_tiled(pixel_samples)
        non_seedvr2_vae.encode_tiled(pixel_samples)

    assert seedvr2_call.call_count == 1, (
        f"Expected encode_tiled_seedvr2 called once across SeedVR2 + "
        f"non-SeedVR2 explicit encode_tiled calls; got "
        f"{seedvr2_call.call_count}."
    )
    assert generic_call.call_count == 1, (
        f"Expected encode_tiled_3d called once across SeedVR2 + non-SeedVR2 "
        f"explicit encode_tiled calls; got {generic_call.call_count}."
    )

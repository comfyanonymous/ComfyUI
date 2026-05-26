"""Unit test for the ``VAE.decode`` tiled-fallback dispatcher routing of
SeedVR2 latents in their 4D collapsed form ``(B, 16*T, H, W)``.

Regression: the dispatcher branch at ``comfy/sd.py``'s
``VAE.decode -> if do_tile: ... elif dims == 2`` previously routed
``ndim == 4`` SeedVR2 latents to the generic ``decode_tiled_``, whose
``tiled_scale`` mask broadcast does not understand the
``(16, T)`` channel-time collapse and crashed with
``"The size of tensor a (1024) must match the size of tensor b (256)
at non-singleton dimension 4"``.

Post-fix: when the wrapped model is a
``comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper`` and the input is 4D,
the dispatcher must route to ``decode_tiled_seedvr2`` instead. This
test verifies the dispatcher selection without invoking the actual VAE
math (which would require real model weights and a GPU): the two
candidate methods are patched, the regular decode is forced to OOM via
a stub, and the test asserts that ``decode_tiled_seedvr2`` is called
exactly once (and ``decode_tiled_`` zero times) for a 4D SeedVR2
input.
"""

from unittest.mock import MagicMock, patch

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.seedvr.vae as seedvr_vae_mod  # noqa: E402
import comfy.sd as sd_mod  # noqa: E402


def _make_minimal_seedvr2_vae():
    """Construct a ``comfy.sd.VAE`` instance whose ``first_stage_model``
    is a real ``VideoAutoencoderKLWrapper`` (built via ``__new__`` to
    skip weight allocation), with the VAE's other attributes stubbed
    to the minimum that ``VAE.decode``'s regular-decode setup path
    requires before the OOM forced fallback.
    """
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    wrapper = seedvr_vae_mod.VideoAutoencoderKLWrapper.__new__(
        seedvr_vae_mod.VideoAutoencoderKLWrapper
    )
    vae.first_stage_model = wrapper

    # Minimum surface that ``VAE.decode`` touches before tiled fallback:
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
    vae.latent_dim = 3  # SeedVR2 is a 3D-temporal latent format (T, H, W)
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None

    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.process_input = lambda x: x
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    return vae


def _force_regular_decode_oom(*args, **kwargs):
    """Stub ``first_stage_model.decode`` to raise an OOM-shaped error
    so ``VAE.decode``'s ``except`` branch sets ``do_tile = True`` and
    falls into the tiled-fallback dispatcher.
    """
    raise torch.cuda.OutOfMemoryError("forced OOM for dispatcher test")


def test_4d_seedvr2_latent_routes_to_decode_tiled_seedvr2():
    vae = _make_minimal_seedvr2_vae()
    samples_4d = torch.zeros(1, 16 * 3, 8, 8)  # (B, 16*T, H, W), T=3

    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(seedvr_vae_mod.VideoAutoencoderKLWrapper, "decode",
                      side_effect=_force_regular_decode_oom), \
         patch.object(sd_mod.VAE, "decode_tiled_seedvr2", seedvr2_call), \
         patch.object(sd_mod.VAE, "decode_tiled_", generic_call):
        vae.decode(samples_4d)

    assert seedvr2_call.call_count == 1, (
        f"Expected decode_tiled_seedvr2 to be called once for a 4D SeedVR2 "
        f"latent under tiled fallback; got {seedvr2_call.call_count} calls."
    )
    assert generic_call.call_count == 0, (
        f"decode_tiled_ must NOT be called for a 4D SeedVR2 latent; got "
        f"{generic_call.call_count} calls. Pre-fix dispatcher would route "
        f"to this method and crash inside tiled_scale's mask broadcast."
    )


def test_4d_non_seedvr2_latent_still_routes_to_generic_decode_tiled():
    """The dispatcher fix must NOT affect non-SeedVR2 4D latents: any
    other VAE whose ``first_stage_model`` is not a
    ``VideoAutoencoderKLWrapper`` continues to route to the generic
    ``decode_tiled_``.
    """
    vae = sd_mod.VAE.__new__(sd_mod.VAE)
    vae.first_stage_model = MagicMock()  # NOT a VideoAutoencoderKLWrapper

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
    vae.latent_channels = 4
    vae.latent_dim = 2
    vae.downscale_ratio = 8
    vae.downscale_index_formula = None
    vae.vae_output_dtype = lambda: torch.float32
    vae.spacial_compression_decode = lambda: 8
    vae.process_output = lambda x: x
    vae.throw_exception_if_invalid = lambda: None
    vae.memory_used_decode = lambda *a, **k: 1
    vae.first_stage_model.decode = MagicMock(
        side_effect=_force_regular_decode_oom
    )

    samples_4d = torch.zeros(1, 4, 8, 8)
    generic_call = MagicMock(return_value=torch.zeros(1, 3, 64, 64))
    seedvr2_call = MagicMock(return_value=torch.zeros(1, 3, 9, 64, 64))

    with patch.object(sd_mod.model_management, "raise_non_oom",
                      lambda e: None), \
         patch.object(sd_mod.model_management, "load_models_gpu",
                      lambda *a, **k: None), \
         patch.object(sd_mod.model_management, "soft_empty_cache",
                      lambda: None), \
         patch.object(sd_mod.VAE, "decode_tiled_seedvr2", seedvr2_call), \
         patch.object(sd_mod.VAE, "decode_tiled_", generic_call):
        vae.decode(samples_4d)

    assert generic_call.call_count == 1, (
        f"Expected decode_tiled_ to be called once for a non-SeedVR2 4D "
        f"latent; got {generic_call.call_count} calls."
    )
    assert seedvr2_call.call_count == 0, (
        f"decode_tiled_seedvr2 must NOT be called for non-SeedVR2 latents; "
        f"got {seedvr2_call.call_count} calls."
    )

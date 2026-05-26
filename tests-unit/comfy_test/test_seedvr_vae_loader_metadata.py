"""Regression test for ``comfy/sd.py``'s ``VAE.__init__`` loader — must
apply SeedVR2-specific metadata when the SeedVR2 magic key
``decoder.up_blocks.2.upsamplers.0.upscale_conv.weight`` is present in the
state dict.

Without the SeedVR2 elif branch the loader leaves ``latent_channels=4`` /
``latent_dim=2`` defaults, so down-stream consumers mis-shape the latent
buffer and crash with a channel-count mismatch. The expected behaviour
sets ``latent_channels=16``, ``latent_dim=3``, ``disable_offload=True``,
``downscale_index_formula=(4, 8, 8)``, ``upscale_index_formula=(4, 8,
8)``, plus the SeedVR2 ``memory_used_decode`` / ``memory_used_encode``
lambdas, the ``downscale_ratio`` / ``upscale_ratio`` tuples, and the
SeedVR2 ``process_input`` / ``crop_input=False`` overrides.

This module exercises the real ``VAE.__init__`` detection-and-load path
with a stubbed state dict containing only the SeedVR2 magic key, and
patches ``comfy.ldm.seedvr.vae.VideoAutoencoderKLWrapper`` with a tiny
``nn.Module`` subclass so the test stays CPU-only and weight-load-free
while still satisfying ``isinstance(...)`` against the real wrapper class
(see ``_StubVideoAutoencoderKLWrapper`` below).
"""

from unittest.mock import patch

import pytest
import torch

# CPU-only CI fix: ``comfy.sd`` transitively imports
# ``comfy.model_management``, whose import-time
# ``cpu_state = CPUState.CPU if args.cpu`` initialiser reads
# ``comfy.cli_args.args.cpu``. Match the pattern at
# ``tests-unit/comfy_test/test_seedvr_vae_decode_unpadded_t.py:33-44``: flip
# ``args.cpu`` BEFORE importing any ``comfy.sd`` / ``comfy.ldm.*`` symbol
# when CUDA is unavailable. Issue-191 AC-3 additionally requires the
# ``_cli_args.cpu = True`` assignment line number to precede every line
# matching ``^import comfy`` or ``^from comfy`` in the committed file, so
# the cli_args module is loaded via ``importlib`` here rather than via
# ``from comfy.cli_args import args``.
import importlib

_cli_args = importlib.import_module("comfy.cli_args").args

if not torch.cuda.is_available():
    _cli_args.cpu = True

import torch.nn as nn  # noqa: E402

import comfy.ldm.seedvr.vae as seedvr_vae  # noqa: E402
import comfy.sd  # noqa: E402


_SEEDVR2_MAGIC_KEY = "decoder.up_blocks.2.upsamplers.0.upscale_conv.weight"


class _StubVideoAutoencoderKLWrapper(seedvr_vae.VideoAutoencoderKLWrapper):
    """Subclass that bypasses the real wrapper's heavy weight construction.

    The downstream ``comfy.sd.VAE.__init__`` lifecycle after line 519 only
    relies on ``nn.Module`` machinery — ``.eval()``, ``.to(dtype)``,
    ``state_dict()`` for ``module_size``, and
    ``load_state_dict(strict=False)``. A bare ``nn.Module.__init__`` provides
    all of that. Subclassing ``VideoAutoencoderKLWrapper`` keeps
    ``isinstance(stub_instance, VideoAutoencoderKLWrapper)`` ``True`` after
    the patch context exits, so the AC-A isinstance assertion holds against
    the real wrapper class.
    """

    def __init__(self):
        nn.Module.__init__(self)


def _build_seedvr2_stub_sd():
    """Minimum state dict that triggers the SeedVR2 elif branch in
    ``comfy/sd.py``. The detection is a pure ``in sd`` containment check
    against the magic key at line 518; no other key is required to reach
    that branch (the diffusers-convert early-out at lines 444-446 is
    short-circuited by the ``is_seedvr2_vae`` flag set at line 443).

    The ``load_state_dict`` call at line 884 uses ``strict=False`` so the
    single magic key is accepted as ``unexpected`` against the empty stub
    module without raising.
    """
    return {_SEEDVR2_MAGIC_KEY: torch.zeros(1)}


@pytest.fixture(scope="module")
def seedvr2_vae():
    """Build a real ``comfy.sd.VAE`` instance through the detection-and-load
    path with the SeedVR2 wrapper class stubbed for CPU-only execution.
    """
    sd = _build_seedvr2_stub_sd()
    with patch.object(
        seedvr_vae,
        "VideoAutoencoderKLWrapper",
        _StubVideoAutoencoderKLWrapper,
    ):
        vae = comfy.sd.VAE(sd=sd)
    return vae


def test_seedvr2_loader_first_stage_model_is_video_autoencoder_kl_wrapper(
    seedvr2_vae,
):
    assert isinstance(
        seedvr2_vae.first_stage_model, seedvr_vae.VideoAutoencoderKLWrapper
    ) is True, (
        "Expected first_stage_model to be a VideoAutoencoderKLWrapper "
        f"instance; got {type(seedvr2_vae.first_stage_model).__name__}. The "
        "SeedVR2 elif branch at comfy/sd.py:518 may not have been taken."
    )


def test_seedvr2_loader_sets_latent_channels_16(seedvr2_vae):
    assert seedvr2_vae.latent_channels == 16, (
        "Expected latent_channels=16 (set at comfy/sd.py:520 inside the "
        f"SeedVR2 elif branch); got {seedvr2_vae.latent_channels}. SeedVR2's "
        "VideoAutoencoderKL uses 16-channel latents per Wang et al., ICLR "
        "2026 (arXiv 2506.05301) §3; the loader default of 4 (comfy/sd.py:457)"
        " is wrong for the SeedVR2 path."
    )


def test_seedvr2_loader_sets_latent_dim_3(seedvr2_vae):
    assert seedvr2_vae.latent_dim == 3, (
        "Expected latent_dim=3 (set at comfy/sd.py:521 inside the SeedVR2 "
        f"elif branch); got {seedvr2_vae.latent_dim}. SeedVR2 latents are 3D "
        "(T, H, W) per the upstream ByteDance-Seed/SeedVR "
        "VideoAutoencoderKL contract; the loader default of 2 "
        "(comfy/sd.py:458) is wrong for the SeedVR2 path."
    )


def test_seedvr2_loader_sets_downscale_index_formula(seedvr2_vae):
    assert seedvr2_vae.downscale_index_formula == (4, 8, 8), (
        "Expected downscale_index_formula=(4, 8, 8) (set at "
        f"comfy/sd.py:527); got {seedvr2_vae.downscale_index_formula}. "
        "SeedVR2's spatial-temporal downscale ratio is 4× temporal × 8× "
        "spatial × 8× spatial."
    )


def test_seedvr2_loader_sets_upscale_index_formula(seedvr2_vae):
    assert seedvr2_vae.upscale_index_formula == (4, 8, 8), (
        "Expected upscale_index_formula=(4, 8, 8) (set at "
        f"comfy/sd.py:529); got {seedvr2_vae.upscale_index_formula}. "
        "SeedVR2's spatial-temporal upscale ratio is the inverse of its "
        "downscale ratio: 4× temporal × 8× spatial × 8× spatial."
    )


def test_seedvr2_loader_sets_disable_offload(seedvr2_vae):
    assert seedvr2_vae.disable_offload is True, (
        "Expected disable_offload=True (set at comfy/sd.py:522); got "
        f"{seedvr2_vae.disable_offload}. SeedVR2 cannot tolerate CPU "
        "offload during decode (the wrapper retains memory-state references "
        "across slice boundaries — see VideoAutoencoderKL.slicing_decode)."
    )


def test_seedvr2_loader_normalizes_comfy_pixels_at_vae_boundary(seedvr2_vae):
    pixels = torch.tensor([0.0, 0.5, 1.0])

    normalized = seedvr2_vae.process_input(pixels)

    assert torch.equal(normalized, torch.tensor([-1.0, 0.0, 1.0]))

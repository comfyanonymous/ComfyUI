"""Unit tests for ``comfy_extras.nodes_seedvr.SeedVR2ProgressiveSampler``."""

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.sample  # noqa: E402
import comfy_extras.nodes_seedvr as nodes_seedvr_mod  # noqa: E402
from comfy_extras.nodes_seedvr import SeedVR2ProgressiveSampler  # noqa: E402

_LAT_C = 16
_COND_C = 17


def _make_inputs(B: int = 1, T: int = 5, H: int = 8, W: int = 8):
    """Build minimal SeedVR2-shaped sampling inputs."""
    samples_5d = torch.arange(
        B * _LAT_C * T * H * W, dtype=torch.float32
    ).reshape(B, _LAT_C, T, H, W)
    samples = samples_5d.reshape(B, _LAT_C * T, H, W).contiguous()

    cond_5d = torch.arange(
        B * _COND_C * T * H * W, dtype=torch.float32
    ).reshape(B, _COND_C, T, H, W) + 10000.0
    cond = cond_5d.reshape(B, _COND_C * T, H, W).contiguous()

    text_pos = torch.zeros(1, 4, 32)
    text_neg = torch.zeros(1, 4, 32)
    positive = [[text_pos, {"condition": cond.clone()}]]
    negative = [[text_neg, {"condition": cond.clone()}]]
    latent_image = {"samples": samples}
    return latent_image, positive, negative, samples_5d, cond_5d


def _identity_fix_empty(model, latent_image, downscale_ratio_spacial=None):
    return latent_image


def _fingerprinted_prepare_noise(latent_image, seed, batch_inds=None):
    """Return a tensor whose values encode ``(seed, position)``."""
    base = torch.arange(
        latent_image.numel(), dtype=torch.float32
    ).reshape(latent_image.shape)
    return base + float(seed) * 1e6


def test_progressive_sampler_schema_exposes_manual_default_auto_chunking():
    schema = SeedVR2ProgressiveSampler.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert inputs["chunking_mode"].options == ["manual", "auto"]
    assert inputs["chunking_mode"].default == "manual"


def test_auto_chunking_walks_two_three_four_chunk_ladder():
    """Auto mode must walk 2-, 3-, then 4-chunk geometries on OOM."""
    latent, pos, neg, _, _ = _make_inputs(T=17)
    calls = []

    def _oom_until_four_chunks(model, noise, steps, cfg, sampler_name,
                               scheduler, positive, negative,
                               latent_image, denoise=1.0,
                               noise_mask=None, seed=None):
        calls.append(tuple(latent_image.shape))
        if latent_image.shape[1] > _LAT_C * 5:
            raise torch.cuda.OutOfMemoryError("chunk too large")
        return latent_image.clone()

    with patch.object(comfy.sample, "sample",
                      side_effect=_oom_until_four_chunks), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise), \
         patch.object(nodes_seedvr_mod.comfy.model_management,
                      "soft_empty_cache") as soft_empty:
        out = SeedVR2ProgressiveSampler.execute(
            model=None, seed=0, steps=2, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            positive=pos, negative=neg, latent=latent,
            denoise=1.0, frames_per_chunk=65, temporal_overlap=0,
            chunking_mode="auto",
        )

    assert calls[:4] == [
        (1, _LAT_C * 17, 8, 8),
        (1, _LAT_C * 9, 8, 8),
        (1, _LAT_C * 6, 8, 8),
        (1, _LAT_C * 5, 8, 8),
    ]
    assert torch.equal(out.result[0]["samples"], latent["samples"])
    assert soft_empty.call_count == 3


@pytest.mark.parametrize("bad_chunk", [0, -1, 2])
def test_t3_invalid_frames_per_chunk_raises_value_error(bad_chunk):
    """``frames_per_chunk`` violating 4n+1 (or <1) must raise ``ValueError`` before any model invocation."""
    latent, pos, neg, _, _ = _make_inputs(T=5)

    sampler_called = {"n": 0}

    def _should_not_be_called(*args, **kwargs):
        sampler_called["n"] += 1
        return torch.zeros(1)

    with patch.object(comfy.sample, "sample",
                      side_effect=_should_not_be_called), \
         patch.object(comfy.sample, "fix_empty_latent_channels",
                      side_effect=_identity_fix_empty), \
         patch.object(comfy.sample, "prepare_noise",
                      side_effect=_fingerprinted_prepare_noise):
        with pytest.raises(ValueError) as excinfo:
            SeedVR2ProgressiveSampler.execute(
                model=None, seed=0, steps=2, cfg=1.0,
                sampler_name="euler", scheduler="simple",
                positive=pos, negative=neg, latent=latent,
                denoise=1.0, frames_per_chunk=bad_chunk, temporal_overlap=0,
            )
    assert str(bad_chunk) in str(excinfo.value)
    assert sampler_called["n"] == 0

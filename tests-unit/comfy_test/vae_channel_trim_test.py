import logging

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.sd import VAE  # noqa: E402


def make_vae():
    vae = VAE(sd={})
    vae.crop_input = False
    return vae


def test_extra_channels_are_trimmed_and_warned_once(caplog):
    vae = make_vae()
    pixels = torch.zeros(1, 16, 16, 4)

    with caplog.at_level(logging.WARNING):
        first = vae.vae_encode_crop_pixels(pixels)
        second = vae.vae_encode_crop_pixels(pixels)

    assert first.shape == (1, 16, 16, 3)
    assert second.shape == (1, 16, 16, 3)

    trim_warnings = [r for r in caplog.records if "VAE encode" in r.getMessage()]
    assert len(trim_warnings) == 1
    assert "SplitImageWithAlpha" in trim_warnings[0].getMessage()


def test_matching_channels_do_not_warn(caplog):
    vae = make_vae()

    with caplog.at_level(logging.WARNING):
        out = vae.vae_encode_crop_pixels(torch.zeros(1, 16, 16, 3))

    assert out.shape == (1, 16, 16, 3)
    assert not [r for r in caplog.records if "VAE encode" in r.getMessage()]

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.latent_formats
import comfy.sample


class _Model:
    def __init__(self, latent_format):
        self._latent_format = latent_format

    def get_model_object(self, name):
        assert name == "latent_format"
        return self._latent_format


def test_seedvr2_latent_format_uses_16_channels_without_3d_empty_latent_expansion():
    latent_format = comfy.latent_formats.SeedVR2()
    latent_image = torch.zeros(1, 1, 4, 5)

    fixed = comfy.sample.fix_empty_latent_channels(_Model(latent_format), latent_image)

    assert latent_format.latent_channels == 16
    assert latent_format.latent_dimensions == 2
    assert fixed.shape == (1, 16, 4, 5)


def test_seedvr2_empty_collapsed_latent_preserves_temporal_channel_multiples():
    latent_format = comfy.latent_formats.SeedVR2()
    latent_image = torch.zeros(1, 48, 4, 5)

    fixed = comfy.sample.fix_empty_latent_channels(_Model(latent_format), latent_image)

    assert latent_format.preserve_empty_channel_multiples is True
    assert fixed.shape == latent_image.shape
    assert fixed.data_ptr() == latent_image.data_ptr()

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.latent_formats import SD3
from comfy_extras.nodes_custom_sampler import AddNoise


class _Identity:
    def __call__(self, value):
        return value


class _ModelSampling:
    def noise_scaling(self, sigma, noise, latent_image):
        return noise * sigma


class _Model:
    def __init__(self):
        self.objects = {
            "latent_format": SD3(),
            "model_sampling": _ModelSampling(),
            "process_latent_in": _Identity(),
            "process_latent_out": _Identity(),
        }

    def get_model_object(self, name):
        return self.objects[name]


class _OnesNoise:
    def generate_noise(self, latent):
        return torch.ones_like(latent["samples"])


def test_add_noise_expands_an_empty_latent_to_the_model_channel_count():
    latent = {"samples": torch.zeros(1, 4, 2, 2)}

    result = AddNoise.add_noise(_Model(), _OnesNoise(), torch.tensor([1.0, 0.0]), latent)

    assert result[0]["samples"].shape == (1, 16, 2, 2)


def test_add_noise_preserves_the_channels_of_a_non_empty_latent():
    latent = {"samples": torch.ones(1, 4, 2, 2)}

    result = AddNoise.add_noise(_Model(), _OnesNoise(), torch.tensor([1.0, 0.0]), latent)

    assert result[0]["samples"].shape == (1, 4, 2, 2)

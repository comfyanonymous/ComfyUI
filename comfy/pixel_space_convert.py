import torch


# "Fake" VAE that converts from IMAGE B, H, W, C and values on the scale of 0..1
# to LATENT B, C, H, W and values on the scale of -1..1.
class PixelspaceConversionVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_space_vae = torch.nn.Parameter(torch.tensor(1.0))

    def encode(self, pixels: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return pixels

    def decode(self, samples: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return samples


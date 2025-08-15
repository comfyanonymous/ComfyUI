from abc import ABC, abstractmethod
from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def latent_to_pixel_coords(
    latent_coords: Tensor, scale_factors: Tuple[int, int, int], causal_fix: bool = False
) -> Tensor:
    """
    Converts latent coordinates to pixel coordinates by scaling them according to the VAE's
    configuration.
    Args:
        latent_coords (Tensor): A tensor of shape [batch_size, 3, num_latents]
        containing the latent corner coordinates of each token.
        scale_factors (Tuple[int, int, int]): The scale factors of the VAE's latent space.
        causal_fix (bool): Whether to take into account the different temporal scale
            of the first frame. Default = False for backwards compatibility.
    Returns:
        Tensor: A tensor of pixel coordinates corresponding to the input latent coordinates.
    """
    pixel_coords = (
        latent_coords
        * torch.tensor(scale_factors, device=latent_coords.device)[None, :, None]
    )
    if causal_fix:
        # Fix temporal scale for first frame to 1 due to causality
        pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


class Patchifier(ABC):
    def __init__(self, patch_size: int):
        super().__init__()
        self._patch_size = (1, patch_size, patch_size)

    @abstractmethod
    def patchify(
        self, latents: Tensor, frame_rates: Tensor, scale_grid: bool
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        pass

    @property
    def patch_size(self):
        return self._patch_size

    def get_latent_coords(
        self, latent_num_frames, latent_height, latent_width, batch_size, device
    ):
        """
        Return a tensor of shape [batch_size, 3, num_patches] containing the
            top-left corner latent coordinates of each latent patch.
        The tensor is repeated for each batch element.
        """
        latent_sample_coords = torch.meshgrid(
            torch.arange(0, latent_num_frames, self._patch_size[0], device=device),
            torch.arange(0, latent_height, self._patch_size[1], device=device),
            torch.arange(0, latent_width, self._patch_size[2], device=device),
            indexing="ij",
        )
        latent_sample_coords = torch.stack(latent_sample_coords, dim=0)
        latent_coords = latent_sample_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        latent_coords = rearrange(
            latent_coords, "b c f h w -> b c (f h w)", b=batch_size
        )
        return latent_coords


class SymmetricPatchifier(Patchifier):
    def patchify(
        self,
        latents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b, _, f, h, w = latents.shape
        latent_coords = self.get_latent_coords(f, h, w, b, latents.device)
        latents = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents, latent_coords

    def unpatchify(
        self,
        latents: Tensor,
        output_height: int,
        output_width: int,
        output_num_frames: int,
        out_channels: int,
    ) -> Tuple[Tensor, Tensor]:
        output_height = output_height // self._patch_size[1]
        output_width = output_width // self._patch_size[2]
        latents = rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q) ",
            f=output_num_frames,
            h=output_height,
            w=output_width,
            p=self._patch_size[1],
            q=self._patch_size[2],
        )
        return latents

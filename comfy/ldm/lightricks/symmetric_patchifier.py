from abc import ABC, abstractmethod
from typing import Tuple

import torch
from einops import rearrange
from torch import Tensor


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    elif dims_to_append == 0:
        return x
    return x[(...,) + (None,) * dims_to_append]


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

    def get_grid(
        self, orig_num_frames, orig_height, orig_width, batch_size, scale_grid, device
    ):
        f = orig_num_frames // self._patch_size[0]
        h = orig_height // self._patch_size[1]
        w = orig_width // self._patch_size[2]
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_f = torch.arange(f, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if scale_grid is not None:
            for i in range(3):
                if isinstance(scale_grid[i], Tensor):
                    scale = append_dims(scale_grid[i], grid.ndim - 1)
                else:
                    scale = scale_grid[i]
                grid[:, i, ...] = grid[:, i, ...] * scale * self._patch_size[i]

        grid = rearrange(grid, "b c f h w -> b c (f h w)", b=batch_size)
        return grid


class SymmetricPatchifier(Patchifier):
    def patchify(
        self,
        latents: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        latents = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self._patch_size[0],
            p2=self._patch_size[1],
            p3=self._patch_size[2],
        )
        return latents

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

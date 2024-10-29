#original code from https://github.com/genmoai/models under apache 2.0 license

# import functools
import math

import torch


def centers(start: float, stop, num, dtype=None, device=None):
    """linspace through bin centers.

    Args:
        start (float): Start of the range.
        stop (float): End of the range.
        num (int): Number of points.
        dtype (torch.dtype): Data type of the points.
        device (torch.device): Device of the points.

    Returns:
        centers (Tensor): Centers of the bins. Shape: (num,).
    """
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


# @functools.lru_cache(maxsize=1)
def create_position_matrix(
    T: int,
    pH: int,
    pW: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    target_area: float = 36864,
):
    """
    Args:
        T: int - Temporal dimension
        pH: int - Height dimension after patchify
        pW: int - Width dimension after patchify

    Returns:
        pos: [T * pH * pW, 3] - position matrix
    """
    # Create 1D tensors for each dimension
    t = torch.arange(T, dtype=dtype)

    # Positionally interpolate to area 36864.
    # (3072x3072 frame with 16x16 patches = 192x192 latents).
    # This automatically scales rope positions when the resolution changes.
    # We use a large target area so the model is more sensitive
    # to changes in the learned pos_frequencies matrix.
    scale = math.sqrt(target_area / (pW * pH))
    w = centers(-pW * scale / 2, pW * scale / 2, pW)
    h = centers(-pH * scale / 2, pH * scale / 2, pH)

    # Use meshgrid to create 3D grids
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

    # Stack and reshape the grids.
    pos = torch.stack([grid_t, grid_h, grid_w], dim=-1)  # [T, pH, pW, 3]
    pos = pos.view(-1, 3)  # [T * pH * pW, 3]
    pos = pos.to(dtype=dtype, device=device)

    return pos


def compute_mixed_rotation(
    freqs: torch.Tensor,
    pos: torch.Tensor,
):
    """
    Project each 3-dim position into per-head, per-head-dim 1D frequencies.

    Args:
        freqs: [3, num_heads, num_freqs] - learned rotation frequency (for t, row, col) for each head position
        pos: [N, 3] - position of each token
        num_heads: int

    Returns:
        freqs_cos: [N, num_heads, num_freqs] - cosine components
        freqs_sin: [N, num_heads, num_freqs] - sine components
    """
    assert freqs.ndim == 3
    freqs_sum = torch.einsum("Nd,dhf->Nhf", pos.to(freqs), freqs)
    freqs_cos = torch.cos(freqs_sum)
    freqs_sin = torch.sin(freqs_sum)
    return freqs_cos, freqs_sin

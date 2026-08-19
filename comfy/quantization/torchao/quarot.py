"""
QuaRot: Group-wise Hadamard rotation for INT quantization quality.

Spreads activation outliers across channels via orthogonal Hadamard matrices.
Based on QuaRot (2024). Pure PyTorch — scipy optional for large matrices.
"""

import torch

try:
    from scipy.linalg import hadamard as _scipy_hadamard
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

_HADAMARD_CACHE: dict = {}
_HADAMARD_MAX_SIZE = 64  # Maximum number of cached (size, device, dtype) entries


def build_hadamard(
    size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build normalized orthogonal Hadamard matrix (size must be power of 2).
    Returns H such that H @ H^T = I.
    Uses scipy if available, otherwise pure-PyTorch Sylvester construction.
    """
    if size & (size - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if _SCIPY_AVAILABLE:
        import numpy as np
        H_np = _scipy_hadamard(size)
        H = torch.tensor(H_np, dtype=dtype, device=device) / (size ** 0.5)
    else:
        # Pure-PyTorch Sylvester construction
        H = torch.tensor([[1.0]], dtype=dtype, device=device)
        n = 1
        while n < size:
            top = torch.cat([H, H], dim=1)
            bottom = torch.cat([H, -H], dim=1)
            H = torch.cat([top, bottom], dim=0)
            n *= 2
        H = H / (size ** 0.5)

    # Enforce cache size cap (evict oldest entry)
    if len(_HADAMARD_CACHE) >= _HADAMARD_MAX_SIZE:
        _HADAMARD_CACHE.pop(next(iter(_HADAMARD_CACHE)))

    _HADAMARD_CACHE[cache_key] = H
    return H


def rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Rotate weight matrix offline: W_rot = W @ H_block^T.

    For Linear(in, out) with weight (out_f, in_f):
    each row is split into groups of group_size and rotated by H^T.

    Args:
        weight:     (out_features, in_features)
        H:          Normalized Hadamard matrix (group_size, group_size)
        group_size: Group size for block-diagonal rotation

    Returns:
        Rotated weight, same shape.
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size
    W_grouped = weight.view(out_f, n_groups, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)

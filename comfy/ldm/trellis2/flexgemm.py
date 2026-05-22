from typing import Optional, Tuple
import torch

import comfy.model_management

UINT32_SENTINEL = 0xFFFFFFFF


def compute_kernel_offsets(Kw, Kh, Kd, Dw, Dh, Dd, device):
    """Kernel spatial offsets in the same order as the CUDA/Triton kernels."""
    offsets = []
    for vx in range(Kw):
        for vy in range(Kh):
            for vz in range(Kd):
                offsets.append((vx * Dw, vy * Dh, vz * Dd))
    return torch.tensor(offsets, device=device, dtype=torch.int32)


class TorchHashMap:
    """Sorted-array hashmap backed by torch.searchsorted."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor, default_value: int):
        device = keys.device
        self.sorted_keys, order = torch.sort(keys.to(torch.long))
        self.sorted_vals = values.to(torch.long)[order]
        self.default_value = torch.tensor(default_value, dtype=torch.long, device=device)
        self._n = self.sorted_keys.numel()

    # Chunk size for lookup_flat, caps each transient to ~CHUNK rows.
    _LOOKUP_CHUNK = 1 << 23   # 8M rows ≈ 64 MB per int64 temp

    def lookup_flat(self, flat_keys: torch.Tensor) -> torch.Tensor:
        N = flat_keys.shape[0]
        out = torch.full((N,), -1, device=flat_keys.device, dtype=torch.int32)
        if self._n == 0 or N == 0:
            return out
        for s in range(0, N, self._LOOKUP_CHUNK):
            e = min(s + self._LOOKUP_CHUNK, N)
            flat_chunk = flat_keys[s:e].to(torch.long)
            idx = torch.searchsorted(self.sorted_keys, flat_chunk)
            in_range = idx < self._n
            idx.clamp_(max=self._n - 1)  # reuse idx as the "safe" index
            found = in_range & (self.sorted_keys[idx] == flat_chunk)
            if found.any():
                found_idx = found.nonzero(as_tuple=True)[0]
                out[s + found_idx] = self.sorted_vals[idx[found_idx]].to(torch.int32)
        return out


def build_submanifold_neighbor_map(
    hashmap,
    coords: torch.Tensor,
    W, H, D,
    Kw, Kh, Kd,
    Dw, Dh, Dd,
):
    device = coords.device
    M = coords.shape[0]
    V = Kw * Kh * Kd
    half_V = V // 2 + 1
    INVALID = -1

    # int32 neighbour map: 4 bytes/elem vs 8 bytes for int64
    neighbor = torch.full((M, V), INVALID, device=device, dtype=torch.int32)

    b = coords[:, 0].long()
    x = coords[:, 1].long()
    y = coords[:, 2].long()
    z = coords[:, 3].long()

    offsets = compute_kernel_offsets(Kw, Kh, Kd, Dw, Dh, Dd, device)

    ox = x - (Kw // 2) * Dw
    oy = y - (Kh // 2) * Dh
    oz = z - (Kd // 2) * Dd

    for v in range(half_V):
        if v == half_V - 1:
            # Center voxel always maps to itself
            neighbor[:, v] = torch.arange(M, device=device, dtype=torch.int32)
            continue

        dx, dy, dz = offsets[v]

        kx = ox + dx
        ky = oy + dy
        kz = oz + dz

        valid = (
            (kx >= 0) & (kx < W) &
            (ky >= 0) & (ky < H) &
            (kz >= 0) & (kz < D)
        )

        flat = (
            b[valid] * (W * H * D) +
            kx[valid] * (H * D) +
            ky[valid] * D +
            kz[valid]
        )

        if flat.numel() > 0:
            found = hashmap.lookup_flat(flat)
            idx_in_M = torch.where(valid)[0]
            neighbor[idx_in_M, v] = found.to(torch.int32)

            # BUG FIX: old code used  found != hashmap.default_value  which
            # compared int32 -1 against int64 4294967295 → always True.
            # We now explicitly check for valid indices.
            valid_found_mask = found >= 0
            if valid_found_mask.any():
                src_points = idx_in_M[valid_found_mask]
                dst_points = found[valid_found_mask].long()
                neighbor[dst_points, V - 1 - v] = src_points.to(torch.int32)

    return neighbor

def get_recommended_chunk_mem(
    device=None,
    safety_fraction: float = 0.2,
    min_gb: float = 0.25,
    max_gb: float = 2.0,
):
    """Pick a chunk-memory budget (in GB) for sparse conv batching."""
    free_gb = comfy.model_management.get_free_memory(device) / (1024 ** 3)
    return max(min_gb, min(free_gb * safety_fraction, max_gb))

def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    neighbor_cache: Optional[torch.Tensor],
    dilation: tuple,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if feats.shape[0] == 0:
        Co = weight.shape[0]
        return torch.empty((0, Co), device=feats.device, dtype=feats.dtype), None

    W, H, D = shape

    Co, Kw, Kh, Kd, Ci = weight.shape
    V = Kw * Kh * Kd
    device = feats.device

    if neighbor_cache is None:
        b_stride = W * H * D
        x_stride = H * D
        y_stride = D
        z_stride = 1

        flat_keys = (coords[:, 0].long() * b_stride +
                     coords[:, 1].long() * x_stride +
                     coords[:, 2].long() * y_stride +
                     coords[:, 3].long() * z_stride)
        vals = torch.arange(coords.shape[0], dtype=torch.int32, device=device)
        hashmap = TorchHashMap(flat_keys, vals, UINT32_SENTINEL)

        neighbor = build_submanifold_neighbor_map(
            hashmap, coords, W, H, D, Kw, Kh, Kd,
            dilation[0], dilation[1], dilation[2]
        )
    else:
        neighbor = neighbor_cache

    N_pts = feats.shape[0]
    sentinel = -1

    weight_T = weight.view(Co, V * Ci).T

    output = torch.empty(N_pts, Co, device=device, dtype=feats.dtype)

    # Chunk size from memory budget. The dominant peak is `gathered`, of shape (chunk, V, Ci) in feats.dtype.
    max_chunk_mem_gb = get_recommended_chunk_mem(device)
    mem_per_row = V * Ci * feats.element_size()
    max_chunk_mem = max_chunk_mem_gb * (1024 ** 3)
    chunk_size = max(1, int(max_chunk_mem / mem_per_row))
    chunk_size = min(chunk_size, N_pts)

    for start in range(0, N_pts, chunk_size):
        end = min(start + chunk_size, N_pts)
        actual_chunk = end - start

        chunk_neighbor = neighbor[start:end]
        chunk_valid = chunk_neighbor != sentinel
        # clamp(-1 -> 0) keeps invalid indices in-range so the gather is safe
        chunk_idx = chunk_neighbor.clamp(min=0)

        # (chunk, V, Ci) gather, then in-place zero of invalid neighbors.
        gathered = feats[chunk_idx]
        gathered.mul_(chunk_valid.unsqueeze(-1))

        # GEMM (chunk, V*Ci) @ (V*Ci, Co) -> (chunk, Co), written to output[start:end].
        gathered_flat = gathered.view(actual_chunk, V * Ci)
        torch.matmul(gathered_flat, weight_T, out=output[start:end])

    if bias is not None:
        output += bias.unsqueeze(0).to(output.dtype)

    return output, neighbor

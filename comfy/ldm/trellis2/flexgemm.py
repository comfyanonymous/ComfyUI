# will contain every cuda -> pytorch operation

from typing import Optional, Tuple
import torch

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

    def lookup_flat(self, flat_keys: torch.Tensor) -> torch.Tensor:
        flat = flat_keys.to(torch.long)
        if self._n == 0:
            return torch.full((flat.shape[0],), -1, device=flat.device, dtype=torch.int32)
        idx = torch.searchsorted(self.sorted_keys, flat)
        idx_safe = torch.clamp(idx, max=self._n - 1)
        found = (idx < self._n) & (self.sorted_keys[idx_safe] == flat)
        out = torch.full((flat.shape[0],), -1, device=flat.device, dtype=torch.int32)
        if found.any():
            out[found] = self.sorted_vals[idx_safe[found]].to(torch.int32)
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
    safety_fraction: float = 0.4,
    min_gb: float = 0.25,
    max_gb: float = 8.0,
):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if device.type == 'cuda':
        try:
            idx = device.index if device.index is not None else 0
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            free_gb = free_bytes / (1024 ** 3)
            total_gb = total_bytes / (1024 ** 3)

            recommended = free_gb * safety_fraction
            result = max(min_gb, min(recommended, max_gb))
            return result

        except Exception:
            try:
                idx = device.index if device.index is not None else 0
                total_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            except Exception:
                total_gb = 16.0

            if total_gb < 12:
                result = 0.5
            elif total_gb < 16:
                result = 0.75
            elif total_gb < 24:
                result = 1.0
            elif total_gb < 32:
                result = 2.0
            elif total_gb < 48:
                result = 4.0
            else:
                result = 6.0
            return result

    else:
        try:
            import psutil
            avail_gb = psutil.virtual_memory().available / (1024 ** 3)
            recommended = avail_gb * safety_fraction
            result = max(min_gb, min(recommended, max_gb))
            return result
        except ImportError:
            return min_gb

def sparse_submanifold_conv3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape: tuple,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    neighbor_cache: Optional[torch.Tensor],
    dilation: tuple,
    max_chunk_mem_gb: float = 6.0,
    accumulate_f32: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

    if feats.shape[0] == 0:
        Co = weight.shape[0]
        return torch.empty((0, Co), device=feats.device, dtype=feats.dtype), None

    if len(shape) == 5:
        _, _, W, H, D = shape
    else:
        W, H, D = shape

    Co, Kw, Kh, Kd, Ci = weight.shape
    V = Kw * Kh * Kd
    device = feats.device
    sentinel = -1
    max_chunk_mem_gb = get_recommended_chunk_mem(device)

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

    if accumulate_f32:
        weight_T = weight.view(Co, V * Ci).to(torch.float32).T.contiguous()
        output = torch.zeros(N_pts, Co, device=device, dtype=torch.float32)
    else:
        weight_T = weight.view(Co, V * Ci).to(feats.dtype).T.contiguous()
        output = torch.zeros(N_pts, Co, device=device, dtype=feats.dtype)

    # ------------------------------------------------------------------
    # Chunk size from memory budget
    # ------------------------------------------------------------------
    bytes_per_elem = 4 if accumulate_f32 else feats.element_size()
    mem_per_row = V * Ci * bytes_per_elem
    max_chunk_mem = max_chunk_mem_gb * (1024 ** 3)
    chunk_size = max(1, int(max_chunk_mem / mem_per_row))
    chunk_size = min(chunk_size, N_pts)

    # ------------------------------------------------------------------
    # Chunked forward pass
    #   Each iteration:
    #     1. gather   (chunk, V, Ci)     – memory bound
    #     2. mask     zero invalids       – in-place, no extra alloc
    #     3. reshape  (chunk, V*Ci)
    #     4. GEMM     (chunk, V*Ci) @ (V*Ci, Co) → (chunk, Co)  – cuBLAS
    #        written directly into output slice via out= argument
    # ------------------------------------------------------------------
    for start in range(0, N_pts, chunk_size):
        end = min(start + chunk_size, N_pts)
        actual_chunk = end - start

        # (chunk, V) int32
        chunk_neighbor = neighbor[start:end]
        chunk_valid = chunk_neighbor != sentinel

        # Clamp sentinel -1 → 0 for safe indexing.  No clone of the full map.
        chunk_idx = chunk_neighbor.clamp(min=0).long()

        # Gather: (chunk, V, Ci).  Memory-bound, single index_select.
        gathered = feats[chunk_idx]

        # Zero invalid neighbours in-place.  gathered is a fresh tensor from
        # advanced indexing, so in-place mutation is safe.
        gathered.mul_(chunk_valid.unsqueeze(-1))

        # Reshape to (chunk, V*Ci)
        gathered_flat = gathered.view(actual_chunk, V * Ci)
        if accumulate_f32:
            gathered_flat = gathered_flat.to(torch.float32)

        # Single GEMM call per chunk, written directly into output.
        # This avoids allocating a temporary (chunk, Co) tensor.
        torch.matmul(gathered_flat, weight_T, out=output[start:end])

    if accumulate_f32:
        output = output.to(feats.dtype)

    if bias is not None:
        output = output + bias.unsqueeze(0).to(output.dtype)

    return output, neighbor

class Mesh:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.vertex_attrs = vertex_attrs

    @property
    def device(self):
        return self.vertices.device

    def to(self, device, non_blocking=False):
        return Mesh(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.vertex_attrs.to(device, non_blocking=non_blocking) if self.vertex_attrs is not None else None,
        )

    def cuda(self, non_blocking=False):
        return self.to('cuda', non_blocking=non_blocking)

    def cpu(self):
        return self.to('cpu')

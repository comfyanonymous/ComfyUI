# will contain every cuda -> pytorch operation

import math
import torch
from typing import Dict, Callable

NO_TRITION = False
try:
    allow_tf32 = torch.cuda.is_tf32_supported()
except Exception:
    allow_tf32 = False
try:
    import triton
    import triton.language as tl
    heuristics = {
        'valid_kernel': lambda args: args['valid_kernel'](args['B1']),
        'valid_kernel_seg': lambda args: args['valid_kernel_seg'](args['B1']),
    }

    #@triton_autotune(
    #    configs=config.autotune_config,
    #    key=['LOGN', 'Ci', 'Co', 'V', 'allow_tf32'],
    #)
    @triton.heuristics(heuristics)
    @triton.jit
    def sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel(
        input,
        weight,
        bias,
        neighbor,
        sorted_idx,
        output,
        # Tensor dimensions
        N, LOGN, Ci, Co, V: tl.constexpr,
        # Meta-parameters
        B1: tl.constexpr,   # Block size for N dimension
        B2: tl.constexpr,   # Block size for Co dimension
        BK: tl.constexpr,   # Block size for K dimension (V * Ci)
        allow_tf32: tl.constexpr,  # Allow TF32 precision for matmuls
        # Huristic parameters
        valid_kernel,
        valid_kernel_seg,
    ):

        block_id = tl.program_id(axis=0)
        block_dim_co = tl.cdiv(Co, B2)
        block_id_co = block_id % block_dim_co
        block_id_n = block_id // block_dim_co

        # Create pointers for submatrices of A and B.
        num_k = tl.cdiv(Ci, BK)  # Number of blocks in K dimension
        valid_kernel_start = tl.load(valid_kernel_seg + block_id_n)
        valid_kernel_seglen = tl.load(valid_kernel_seg + block_id_n + 1) - valid_kernel_start
        offset_n = block_id_n * B1 + tl.arange(0, B1)
        n_mask = offset_n < N
        offset_sorted_n = tl.load(sorted_idx + offset_n, mask=n_mask, other=0)  # (B1,)
        offset_co = (block_id_co * B2 + tl.arange(0, B2)) % Co                  # (B2,)
        offset_k = tl.arange(0, BK)                                             # (BK,)

        # Create a block of the output matrix C.
        accumulator = tl.zeros((B1, B2), dtype=tl.float32)

        # Iterate along V*Ci dimension.
        for k in range(num_k * valid_kernel_seglen):
            v = k // num_k
            bk = k % num_k
            v = tl.load(valid_kernel + valid_kernel_start + v)
            # Calculate pointers to input matrix.
            neighbor_offset_n = tl.load(neighbor + offset_sorted_n * V + v)                             # (B1,)
            input_ptr = input + bk * BK + (neighbor_offset_n[:, None].to(tl.int64) * Ci + offset_k[None, :])         # (B1, BK)
            # Calculate pointers to weight matrix.
            weight_ptr = weight + v * Ci + bk * BK + (offset_co[None, :] * V * Ci + offset_k[:, None])  # (BK, B2)
            # Load the next block of input and weight.
            neigh_mask = neighbor_offset_n != 0xffffffff
            k_mask = offset_k < Ci - bk * BK
            input_block = tl.load(input_ptr, mask=neigh_mask[:, None] & k_mask[None, :], other=0.0)
            weight_block = tl.load(weight_ptr, mask=k_mask[:, None], other=0.0)
            # Accumulate along the K dimension.
            accumulator = tl.dot(input_block, weight_block, accumulator,
                                input_precision='tf32' if allow_tf32 else 'ieee')                      # (B1, B2)
        c = accumulator.to(input.type.element_ty)

        # add bias
        if bias is not None:
            bias_block = tl.load(bias + offset_co)
            c += bias_block[None, :]

        # Write back the block of the output matrix with masks.
        out_offset_n = offset_sorted_n
        out_offset_co = block_id_co * B2 + tl.arange(0, B2)
        out_ptr = output + (out_offset_n[:, None] * Co + out_offset_co[None, :])
        out_mask = n_mask[:, None] & (out_offset_co[None, :] < Co)
        tl.store(out_ptr, c, mask=out_mask)
    def sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        neighbor: torch.Tensor,
        sorted_idx: torch.Tensor,
        valid_kernel: Callable[[int], torch.Tensor],
        valid_kernel_seg: Callable[[int], torch.Tensor],
    ) -> torch.Tensor:
        N, Ci, Co, V = neighbor.shape[0], input.shape[1], weight.shape[0], weight.shape[1]
        LOGN = int(math.log2(N))
        output = torch.empty((N, Co), device=input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(Co, META['B2']) * triton.cdiv(N, META['B1']),)
        sparse_submanifold_conv_fwd_masked_implicit_gemm_kernel[grid](
            input, weight, bias, neighbor, sorted_idx, output,
            N, LOGN, Ci, Co, V,
            B1=128,
            B2=64,
            BK=32,
            valid_kernel=valid_kernel,
            valid_kernel_seg=valid_kernel_seg,
            allow_tf32=allow_tf32,
        )
        return output
except:
    NO_TRITION = True

def compute_kernel_offsets(Kw, Kh, Kd, Dw, Dh, Dd, device):
    # offsets in same order as CUDA kernel
    offsets = []
    for vx in range(Kw):
        for vy in range(Kh):
            for vz in range(Kd):
                offsets.append((
                    vx * Dw,
                    vy * Dh,
                    vz * Dd
                ))
    return torch.tensor(offsets, device=device)

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

    INVALID = hashmap.default_value

    neighbor = torch.full((M, V), INVALID, device=device, dtype=torch.long)

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
            neighbor[:, v] = torch.arange(M, device=device)
            continue

        dx, dy, dz = offsets[v]

        kx = ox + dx
        ky = oy + dy
        kz = oz + dz

        # Check spatial bounds
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
            neighbor[idx_in_M, v] = found

            valid_found_mask = (found != INVALID)
            if valid_found_mask.any():
                src_points = idx_in_M[valid_found_mask]
                dst_points = found[valid_found_mask]
                neighbor[dst_points, V - 1 - v] = src_points

    return neighbor

class TorchHashMap:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, default_value: int):
        device = keys.device
        # use long for searchsorted
        self.sorted_keys, order = torch.sort(keys.long())
        self.sorted_vals = values.long()[order]
        self.default_value = torch.tensor(default_value, dtype=torch.long, device=device)
        self._n = self.sorted_keys.numel()

    def lookup_flat(self, flat_keys: torch.Tensor) -> torch.Tensor:
        flat = flat_keys.long()
        idx = torch.searchsorted(self.sorted_keys, flat)
        found = (idx < self._n) & (self.sorted_keys[idx] == flat)
        out = torch.full((flat.shape[0],), self.default_value, device=flat.device, dtype=self.sorted_vals.dtype)
        if found.any():
            out[found] = self.sorted_vals[idx[found]]
        return out


UINT32_SENTINEL = 0xFFFFFFFF

def neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor_map):
    device = neighbor_map.device
    N, V = neighbor_map.shape


    neigh = neighbor_map.to(torch.long)
    sentinel = torch.tensor(UINT32_SENTINEL, dtype=torch.long, device=device)


    neigh_map_T = neigh.t().reshape(-1)

    neigh_mask_T = (neigh_map_T != sentinel).to(torch.int32)

    mask = (neigh != sentinel).to(torch.long)

    powers = (1 << torch.arange(V, dtype=torch.long, device=device))

    gray_long = (mask * powers).sum(dim=1)

    gray_code = gray_long.to(torch.int32)

    binary_long = gray_long.clone()
    for v in range(1, V):
        binary_long ^= (gray_long >> v)
    binary_code = binary_long.to(torch.int32)

    sorted_idx = torch.argsort(binary_code)

    prefix_sum_neighbor_mask = torch.cumsum(neigh_mask_T.to(torch.int32), dim=0)  # (V*N,)

    total_valid_signal = int(prefix_sum_neighbor_mask[-1].item()) if prefix_sum_neighbor_mask.numel() > 0 else 0

    if total_valid_signal > 0:
        valid_signal_i = torch.empty((total_valid_signal,), dtype=torch.long, device=device)
        valid_signal_o = torch.empty((total_valid_signal,), dtype=torch.long, device=device)

        pos = torch.nonzero(neigh_mask_T, as_tuple=True)[0]

        to = (prefix_sum_neighbor_mask[pos] - 1).to(torch.long)

        valid_signal_i[to] = (pos % N).to(torch.long)

        valid_signal_o[to] = neigh_map_T[pos].to(torch.long)
    else:
        valid_signal_i = torch.empty((0,), dtype=torch.long, device=device)
        valid_signal_o = torch.empty((0,), dtype=torch.long, device=device)

    seg = torch.empty((V + 1,), dtype=torch.long, device=device)
    seg[0] = 0
    if V > 0:
        idxs = (torch.arange(1, V + 1, device=device, dtype=torch.long) * N) - 1
        seg[1:] = prefix_sum_neighbor_mask[idxs].to(torch.long)
    else:
        pass

    return gray_code, sorted_idx, valid_signal_i, valid_signal_o, seg

def _popcount_int32_tensor(x: torch.Tensor) -> torch.Tensor:

    x = x.to(torch.int64)

    m1 = torch.tensor(0x5555555555555555, dtype=torch.int64, device=x.device)
    m2 = torch.tensor(0x3333333333333333, dtype=torch.int64, device=x.device)
    m4 = torch.tensor(0x0F0F0F0F0F0F0F0F, dtype=torch.int64, device=x.device)
    h01 = torch.tensor(0x0101010101010101, dtype=torch.int64, device=x.device)

    x = x - ((x >> 1) & m1)
    x = (x & m2) + ((x >> 2) & m2)
    x = (x + (x >> 4)) & m4
    x = (x * h01) >> 56
    return x.to(torch.int32)


def neighbor_map_post_process_for_masked_implicit_gemm_2(
    gray_code: torch.Tensor,    # [N], int32-like (non-negative)
    sorted_idx: torch.Tensor,   # [N], long (indexing into gray_code)
    block_size: int
):
    device = gray_code.device
    N = gray_code.numel()

    # num of blocks (same as CUDA)
    num_blocks = (N + block_size - 1) // block_size

    # Ensure dtypes
    gray_long = gray_code.to(torch.int64)       # safer to OR in 64-bit then cast
    sorted_idx = sorted_idx.to(torch.long)

    # 1) Group gray_code by blocks and compute OR across each block
    # pad the last block with zeros if necessary so we can reshape
    pad = num_blocks * block_size - N
    if pad > 0:
        pad_vals = torch.zeros((pad,), dtype=torch.int64, device=device)
        gray_padded = torch.cat([gray_long[sorted_idx], pad_vals], dim=0)
    else:
        gray_padded = gray_long[sorted_idx]

    # reshape to (num_blocks, block_size) and compute bitwise_or across dim=1
    gray_blocks = gray_padded.view(num_blocks, block_size)       # each row = block entries
    # reduce with bitwise_or
    reduced_code = gray_blocks[:, 0].clone()
    for i in range(1, block_size):
        reduced_code |= gray_blocks[:, i]
    reduced_code = reduced_code.to(torch.int32)  # match CUDA int32

    # 2) compute seglen (popcount per reduced_code) and seg (prefix sum)
    seglen_counts = _popcount_int32_tensor(reduced_code.to(torch.int64)).to(torch.int32)  # [num_blocks]
    # seg: length num_blocks+1, seg[0]=0, seg[i+1]=cumsum(seglen_counts) up to i
    seg = torch.empty((num_blocks + 1,), dtype=torch.int32, device=device)
    seg[0] = 0
    if num_blocks > 0:
        seg[1:] = torch.cumsum(seglen_counts, dim=0)

    total = int(seg[-1].item())

    # 3) scatter — produce valid_kernel_idx as concatenated ascending set-bit positions for each reduced_code row
    if total == 0:
        valid_kernel_idx = torch.empty((0,), dtype=torch.int32, device=device)
        return valid_kernel_idx, seg

    max_val = int(reduced_code.max().item())
    V = max_val.bit_length() if max_val > 0 else 0
    # If you know V externally, pass it instead or set here explicitly.

    if V == 0:
        # no bits set anywhere
        valid_kernel_idx = torch.empty((0,), dtype=torch.int32, device=device)
        return valid_kernel_idx, seg

    # build mask of shape (num_blocks, V): True where bit is set
    bit_pos = torch.arange(0, V, dtype=torch.int64, device=device)  # [V]
    # shifted = reduced_code[:, None] >> bit_pos[None, :]
    shifted = reduced_code.to(torch.int64).unsqueeze(1) >> bit_pos.unsqueeze(0)
    bits = (shifted & 1).to(torch.bool)  # (num_blocks, V)

    positions = bit_pos.unsqueeze(0).expand(num_blocks, V)

    valid_positions = positions[bits]
    valid_kernel_idx = valid_positions.to(torch.int32).contiguous()

    return valid_kernel_idx, seg


def sparse_submanifold_conv3d(feats, coords, shape, weight, bias, neighbor_cache, dilation):
    if len(shape) == 5:
        N, C, W, H, D = shape
    else:
        W, H, D = shape

    Co, Kw, Kh, Kd, Ci = weight.shape

    b_stride = W * H * D
    x_stride = H * D
    y_stride = D
    z_stride = 1

    flat_keys = (coords[:, 0].long() * b_stride +
                 coords[:, 1].long() * x_stride +
                 coords[:, 2].long() * y_stride +
                 coords[:, 3].long() * z_stride)

    vals = torch.arange(coords.shape[0], dtype=torch.int32, device=coords.device)

    hashmap = TorchHashMap(flat_keys, vals, 0xFFFFFFFF)

    if neighbor_cache is None:
        neighbor = build_submanifold_neighbor_map(
            hashmap, coords, W, H, D, Kw, Kh, Kd,
            dilation[0], dilation[1], dilation[2]
        )
    else:
        neighbor = neighbor_cache

    block_size = 128

    gray_code, sorted_idx, valid_signal_i, valid_signal_o, valid_signal_seg = \
        neighbor_map_post_process_for_masked_implicit_gemm_1(neighbor)

    valid_kernel, valid_kernel_seg = \
        neighbor_map_post_process_for_masked_implicit_gemm_2(gray_code, sorted_idx, block_size)

    valid_kernel_fn = lambda b_size: valid_kernel
    valid_kernel_seg_fn = lambda b_size: valid_kernel_seg

    weight_flat = weight.contiguous().view(Co, -1, Ci)

    out = sparse_submanifold_conv_fwd_masked_implicit_gemm_splitk(
        feats,
        weight_flat,
        bias,
        neighbor,
        sorted_idx,
        valid_kernel_fn,
        valid_kernel_seg_fn
    )

    return out, neighbor

class Voxel:
    def __init__(
            self,
            origin: list,
            voxel_size: float,
            coords: torch.Tensor = None,
            attrs: torch.Tensor = None,
            layout: Dict = {},
            device: torch.device = 'cuda'
        ):
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.layout = layout
        self.device = device

    @property
    def position(self):
        return (self.coords + 0.5) * self.voxel_size + self.origin[None, :]

    def split_attrs(self):
        return {
            k: self.attrs[:, self.layout[k]]
            for k in self.layout
        }

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

class MeshWithVoxel(Mesh, Voxel):
    def __init__(self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        origin: list,
        voxel_size: float,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        voxel_shape: torch.Size,
        layout: Dict = {},
    ):
        self.vertices = vertices.float()
        self.faces = faces.int()
        self.origin = torch.tensor(origin, dtype=torch.float32, device=self.device)
        self.voxel_size = voxel_size
        self.coords = coords
        self.attrs = attrs
        self.voxel_shape = voxel_shape
        self.layout = layout

    def to(self, device, non_blocking=False):
        return MeshWithVoxel(
            self.vertices.to(device, non_blocking=non_blocking),
            self.faces.to(device, non_blocking=non_blocking),
            self.origin.tolist(),
            self.voxel_size,
            self.coords.to(device, non_blocking=non_blocking),
            self.attrs.to(device, non_blocking=non_blocking),
            self.voxel_shape,
            self.layout,
        )

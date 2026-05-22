import math
from typing import Tuple, Union

import torch

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.trellis2.vae import VarLenTensor


def _build_cu_seqlens(seqlen, device):
    """Cumulative-offset tensor from a list/tensor of per-item sequence lengths."""
    if isinstance(seqlen, torch.Tensor):
        return torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=seqlen.device), torch.cumsum(seqlen, dim=0).int()],
            dim=0,
        ).to(device)
    return torch.cat(
        [torch.tensor([0]), torch.cumsum(torch.tensor(seqlen), dim=0)],
    ).int().to(device)


def _layout_to_feats(t):
    """Returns (sparse_template_or_None, seqlen, feats[T, H, C])."""
    if isinstance(t, VarLenTensor):
        seqlen = [t.layout[i].stop - t.layout[i].start for i in range(t.shape[0])]
        return t, seqlen, t.feats
    N, L = t.shape[:2]
    return None, [L] * N, t.reshape(-1, *t.shape[-2:])


def dense_attention(q, k, v, **kwargs):
    """q, k, v: [B, L, H, C]. Permutes for comfy's [B, H, L, C] convention."""
    heads = q.shape[2]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True, **kwargs)
    return out.permute(0, 2, 1, 3)


def sparse_attention(q, k, v, **kwargs):
    """
    Varlen attention for SparseTensor inputs. Each of q, k, v may be a VarLenTensor
    (sparse) or dense [B, L, H, C]. Output type matches q. Backend dispatch lives
    in comfy.ldm.modules.attention.optimized_attention; we just build cu_seqlens
    from the layouts.
    """
    s, q_seqlen, q_feats = _layout_to_feats(q)
    _, kv_seqlen, k_feats = _layout_to_feats(k)
    _, _, v_feats = _layout_to_feats(v)
    heads = q_feats.shape[1]

    device = q_feats.device
    cu_seqlens_q = _build_cu_seqlens(q_seqlen, device)
    cu_seqlens_kv = _build_cu_seqlens(kv_seqlen, device)

    out = optimized_attention(
        q_feats, k_feats, v_feats, heads,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max(q_seqlen), max_seqlen_kv=max(kv_seqlen),
        skip_reshape=True, skip_output_reshape=True,
        **kwargs,
    )

    if s is not None:
        return s.replace(out)
    N, L = q.shape[:2]
    return out.reshape(N, L, heads, -1)


def sparse_windowed_self_attention(qkv, window_size: int, shift_window: Tuple[int, int, int] = (0, 0, 0)):
    """Windowed sparse self-attention. Partitions voxels into windows via spatial
    sort, then runs varlen attention with one sequence per non-empty window."""
    cache_name = f'windowed_attention_{window_size}_{shift_window}'
    cache = qkv.get_spatial_cache(cache_name)
    if cache is None:
        cache = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(cache_name, cache)
    fwd_indices, bwd_indices, seq_lens = cache

    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]
    q, k, v = qkv_feats.unbind(dim=1)       # each [M, H, C]
    heads = q.shape[1]
    device = q.device

    cu_seqlens = _build_cu_seqlens(seq_lens, device)
    max_seqlen = int(seq_lens.max())

    out = optimized_attention(
        q, k, v, heads,
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_kv=max_seqlen,
        skip_reshape=True, skip_output_reshape=True,
    )
    out = out[bwd_indices]
    return qkv.replace(out)


def calc_window_partition(
    tensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
):
    """Returns (fwd_indices, bwd_indices, seq_lens) for window partitioning."""
    DIM = tensor.coords.shape[1] - 1
    shift_window = (shift_window,) * DIM if isinstance(shift_window, int) else shift_window
    window_size = (window_size,) * DIM if isinstance(window_size, int) else window_size
    shifted_coords = tensor.coords.clone().detach()
    shifted_coords[:, 1:] += torch.tensor(shift_window, device=tensor.device, dtype=torch.int32).unsqueeze(0)

    MAX_COORDS = [i + j for i, j in zip(tensor.spatial_shape, shift_window)]
    NUM_WINDOWS = [math.ceil((mc + 1) / ws) for mc, ws in zip(MAX_COORDS, window_size)]
    OFFSET = torch.cumprod(torch.tensor([1] + NUM_WINDOWS[::-1]), dim=0).tolist()[::-1]

    shifted_coords[:, 1:] //= torch.tensor(window_size, device=tensor.device, dtype=torch.int32).unsqueeze(0)
    shifted_indices = (shifted_coords * torch.tensor(OFFSET, device=tensor.device, dtype=torch.int32).unsqueeze(0)).sum(dim=1)
    fwd_indices = torch.argsort(shifted_indices)
    bwd_indices = torch.empty_like(fwd_indices)
    bwd_indices[fwd_indices] = torch.arange(fwd_indices.shape[0], device=tensor.device)
    seq_lens = torch.bincount(shifted_indices)
    seq_lens = seq_lens[seq_lens != 0]

    return fwd_indices, bwd_indices, seq_lens

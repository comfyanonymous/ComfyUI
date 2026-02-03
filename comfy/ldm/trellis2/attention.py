import torch
import math
from comfy.ldm.modules.attention import optimized_attention
from typing import Tuple, Union, List
from vae import VarLenTensor

FLASH_ATTN_3_AVA = True
try:
    import flash_attn_interface as flash_attn_3
except:
    FLASH_ATTN_3_AVA = False

# TODO repalce with optimized attention
def scaled_dot_product_attention(*args, **kwargs):
    num_all_args = len(args) + len(kwargs)

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']

    if optimized_attention.__name__ == 'attention_xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = xops.memory_efficient_attention(q, k, v)
    elif optimized_attention.__name__ == 'attention_flash' and not FLASH_ATTN_3_AVA:
        if 'flash_attn' not in globals():
            import flash_attn
        if num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
    elif optimized_attention.__name__ == 'attention_flash': # TODO
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
            if num_all_args == 2:
                k, v = kv.unbind(dim=2)
                out = flash_attn_3.flash_attn_func(q, k, v)
            elif num_all_args == 3:
                out = flash_attn_3.flash_attn_func(q, k, v)
    elif optimized_attention.__name__ == 'attention_pytorch':
        if 'sdpa' not in globals():
            from torch.nn.functional import scaled_dot_product_attention as sdpa
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        out = sdpa(q, k, v)         # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    elif optimized_attention.__name__ == 'attention_basic':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.shape[2] # TODO
        out = optimized_attention(q, k, v)

    return out

def sparse_windowed_scaled_dot_product_self_attention(
    qkv,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
):

    serialization_spatial_cache_name = f'windowed_attention_{window_size}_{shift_window}'
    serialization_spatial_cache = qkv.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, attn_func_args))
    else:
        fwd_indices, bwd_indices, seq_lens, attn_func_args = serialization_spatial_cache

    qkv_feats = qkv.feats[fwd_indices]      # [M, 3, H, C]

    if optimized_attention.__name__ == 'attention_xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        q, k, v = qkv_feats.unbind(dim=1)
        q = q.unsqueeze(0)                                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]             # [M, H, C]
    elif optimized_attention.__name__ == 'attention_flash':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]

    out = out[bwd_indices]      # [T, H, C]

    return qkv.replace(out)

def calc_window_partition(
    tensor,
    window_size: Union[int, Tuple[int, ...]],
    shift_window: Union[int, Tuple[int, ...]] = 0,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:

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
    mask = seq_lens != 0
    seq_lens = seq_lens[mask]

    if optimized_attention.__name__ == 'attention_xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        attn_func_args = {
            'attn_bias': xops.fmha.BlockDiagonalMask.from_seqlens(seq_lens)
        }
    elif optimized_attention.__name__ == 'attention_flash':
        attn_func_args = {
            'cu_seqlens': torch.cat([torch.tensor([0], device=tensor.device), torch.cumsum(seq_lens, dim=0)], dim=0).int(),
            'max_seqlen': torch.max(seq_lens)
        }

    return fwd_indices, bwd_indices, seq_lens, attn_func_args


def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        device = q.device

        if isinstance(q, VarLenTensor):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, VarLenTensor):
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        device = q.device

        if isinstance(q, VarLenTensor):
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, VarLenTensor):
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if optimized_attention.__name__ == 'attention_xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif optimized_attention.__name__ == 'attention_flash':
        if 'flash_attn' not in globals():
            import flash_attn
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif optimized_attention.__name__  == 'flash_attn_3': # TODO
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
            cu_seqlens_kv = cu_seqlens_q.clone()
            max_q_seqlen = max_kv_seqlen = max(q_seqlen)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        elif num_all_args == 3:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
            max_q_seqlen = max(q_seqlen)
            max_kv_seqlen = max(kv_seqlen)
        out = flash_attn_3.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_q_seqlen, max_kv_seqlen)

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)

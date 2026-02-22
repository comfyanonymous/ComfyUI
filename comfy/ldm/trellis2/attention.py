import torch
import math
from comfy.ldm.modules.attention import optimized_attention
from typing import Tuple, Union, List
from comfy.ldm.trellis2.vae import VarLenTensor
import comfy.ops


# replica of the seedvr2 code
def var_attn_arg(kwargs):
    cu_seqlens_q = kwargs.get("cu_seqlens_q", None)
    max_seqlen_q = kwargs.get("max_seqlen_q", None)
    cu_seqlens_k = kwargs.get("cu_seqlens_kv", cu_seqlens_q)
    max_seqlen_k = kwargs.get("max_kv_seqlen", max_seqlen_q)
    assert cu_seqlens_q is not None, "cu_seqlens_q shouldn't be None when var_length is True"
    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    var_length = True
    if var_length:
        cu_seqlens_q, cu_seqlens_k, _, _ = var_attn_arg(kwargs)
        if not skip_reshape:
            # assumes 2D q, k,v [total_tokens, embed_dim]
            total_tokens, embed_dim = q.shape
            head_dim = embed_dim // heads
            q = q.view(total_tokens, heads, head_dim)
            k = k.view(k.shape[0], heads, head_dim)
            v = v.view(v.shape[0], heads, head_dim)

        b = q.size(0)
        dim_head = q.shape[-1]
        q = torch.nested.nested_tensor_from_jagged(q, offsets=cu_seqlens_q.long())
        k = torch.nested.nested_tensor_from_jagged(k, offsets=cu_seqlens_k.long())
        v = torch.nested.nested_tensor_from_jagged(v, offsets=cu_seqlens_k.long())

        mask = None
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = comfy.ops.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
    if var_length:
        return out.contiguous().transpose(1, 2).values()
    if not skip_output_reshape:
        out = (
            out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        )
    return out


# TODO repalce with optimized attention
def scaled_dot_product_attention(*args, **kwargs):
    num_all_args = len(args) + len(kwargs)

    q = None
    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']

    if q is not None:
        heads = q
    else:
        heads = qkv
    heads = heads.shape[2]

    if optimized_attention.__name__ == 'attention_xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        #out = xops.memory_efficient_attention(q, k, v)
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)
    elif optimized_attention.__name__ == 'attention_flash':
        if num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)
    elif optimized_attention.__name__ == 'attention_pytorch':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    elif optimized_attention.__name__ == 'attention_basic':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)

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
    heads = qkv_feats.shape[2]

    if optimized_attention.__name__ == 'attention_xformers':
        q, k, v = qkv_feats.unbind(dim=1)
        q = q.unsqueeze(0)                                                              # [1, M, H, C]
        k = k.unsqueeze(0)                                                              # [1, M, H, C]
        v = v.unsqueeze(0)                                                              # [1, M, H, C]
        #out = xops.memory_efficient_attention(q, k, v, **attn_func_args)[0]             # [M, H, C]
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)
    elif optimized_attention.__name__ == 'attention_flash':
        if 'flash_attn' not in globals():
            import flash_attn
        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)  # [M, H, C]
    else:
        out = optimized_attention(q, k, v, heads, skip_output_reshape=True, skip_reshape=True)

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
    q=None
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

    # TODO: change
    if q is not None:
        heads = q
    else:
        heads = qkv
    heads = heads.shape[2]
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

    elif optimized_attention.__name__ == "attention_pytorch":
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        else:
            cu_seqlens_kv = cu_seqlens_q
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        out = attention_pytorch(q, k, v, heads=heads,cu_seqlens_q=cu_seqlens_q,
                                cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max(q_seqlen), max_kv_seqlen=max(kv_seqlen),
                                skip_reshape=True, skip_output_reshape=True)

    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)

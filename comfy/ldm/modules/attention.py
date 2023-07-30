from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any

from .diffusionmodules.util import checkpoint
from .sub_quadratic_attention import efficient_dot_product_attention

from comfy import model_management
import comfy.ops

if model_management.xformers_enabled():
    import xformers
    import xformers.ops

from comfy.cli_args import args
# CrossAttn precision handling
if args.dont_upcast_attention:
    print("disabling upcasting of attention")
    _ATTN_PRECISION = "fp16"
else:
    _ATTN_PRECISION = "fp32"


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None):
        super().__init__()
        self.proj = comfy.ops.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            comfy.ops.Linear(dim, inner_dim, dtype=dtype, device=device),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim, dtype=dtype, device=device)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            comfy.ops.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels, dtype=None, device=None):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttentionBirchSan(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = comfy.ops.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            comfy.ops.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, value=None, mask=None):
        h = self.heads

        query = self.to_q(x)
        context = default(context, x)
        key = self.to_k(context)
        if value is not None:
            value = self.to_v(value)
        else:
            value = self.to_v(context)

        del context, x

        query = query.unflatten(-1, (self.heads, -1)).transpose(1,2).flatten(end_dim=1)
        key_t = key.transpose(1,2).unflatten(1, (self.heads, -1)).flatten(end_dim=1)
        del key
        value = value.unflatten(-1, (self.heads, -1)).transpose(1,2).flatten(end_dim=1)

        dtype = query.dtype
        upcast_attention = _ATTN_PRECISION =="fp32" and query.dtype != torch.float32
        if upcast_attention:
            bytes_per_token = torch.finfo(torch.float32).bits//8
        else:
            bytes_per_token = torch.finfo(query.dtype).bits//8
        batch_x_heads, q_tokens, _ = query.shape
        _, _, k_tokens = key_t.shape
        qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

        mem_free_total, mem_free_torch = model_management.get_free_memory(query.device, True)

        chunk_threshold_bytes = mem_free_torch * 0.5 #Using only this seems to work better on AMD

        kv_chunk_size_min = None

        #not sure at all about the math here
        #TODO: tweak this
        if mem_free_total > 8192 * 1024 * 1024 * 1.3:
            query_chunk_size_x = 1024 * 4
        elif mem_free_total > 4096 * 1024 * 1024 * 1.3:
            query_chunk_size_x = 1024 * 2
        else:
            query_chunk_size_x = 1024
        kv_chunk_size_min_x = None
        kv_chunk_size_x = (int((chunk_threshold_bytes // (batch_x_heads * bytes_per_token * query_chunk_size_x)) * 2.0) // 1024) * 1024
        if kv_chunk_size_x < 1024:
            kv_chunk_size_x = None

        if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
            # the big matmul fits into our memory limit; do everything in 1 chunk,
            # i.e. send it down the unchunked fast-path
            query_chunk_size = q_tokens
            kv_chunk_size = k_tokens
        else:
            query_chunk_size = query_chunk_size_x
            kv_chunk_size = kv_chunk_size_x
            kv_chunk_size_min = kv_chunk_size_min_x

        hidden_states = efficient_dot_product_attention(
            query,
            key_t,
            value,
            query_chunk_size=query_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min=kv_chunk_size_min,
            use_checkpoint=self.training,
            upcast_attention=upcast_attention,
        )

        hidden_states = hidden_states.to(dtype)

        hidden_states = hidden_states.unflatten(0, (-1, self.heads)).transpose(1,2).flatten(start_dim=2)

        out_proj, dropout = self.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        return hidden_states


class CrossAttentionDoggettx(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = comfy.ops.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            comfy.ops.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, value=None, mask=None):
        h = self.heads

        q_in = self.to_q(x)
        context = default(context, x)
        k_in = self.to_k(context)
        if value is not None:
            v_in = self.to_v(value)
            del value
        else:
            v_in = self.to_v(context)
        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = model_management.get_free_memory(q.device)

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1


        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #      f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        # print("steps", steps, mem_required, mem_free_total, modifier, q.element_size(), tensor_size)
        first_op_done = False
        cleared_cache = False
        while True:
            try:
                slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
                for i in range(0, q.shape[1], slice_size):
                    end = i + slice_size
                    if _ATTN_PRECISION =="fp32":
                        with torch.autocast(enabled=False, device_type = 'cuda'):
                            s1 = einsum('b i d, b j d -> b i j', q[:, i:end].float(), k.float()) * self.scale
                    else:
                        s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
                    first_op_done = True

                    s2 = s1.softmax(dim=-1).to(v.dtype)
                    del s1

                    r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                    del s2
                break
            except model_management.OOM_EXCEPTION as e:
                if first_op_done == False:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    if cleared_cache == False:
                        cleared_cache = True
                        print("out of memory error, emptying cache and trying again")
                        continue
                    steps *= 2
                    if steps > 64:
                        raise e
                    print("out of memory error, increasing steps and trying again", steps)
                else:
                    raise e

        del q, k, v

        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1

        return self.to_out(r2)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = comfy.ops.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(
            comfy.ops.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, value=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, dtype=None, device=None):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = comfy.ops.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(comfy.ops.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class CrossAttentionPytorch(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., dtype=None, device=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = comfy.ops.Linear(query_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_k = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.to_v = comfy.ops.Linear(context_dim, inner_dim, bias=False, dtype=dtype, device=device)

        self.to_out = nn.Sequential(comfy.ops.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, value=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        if value is not None:
            v = self.to_v(value)
            del value
        else:
            v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.view(b, -1, self.heads, self.dim_head).transpose(1, 2),
            (q, k, v),
        )

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        )

        return self.to_out(out)

if model_management.xformers_enabled():
    print("Using xformers cross attention")
    CrossAttention = MemoryEfficientCrossAttention
elif model_management.pytorch_attention_enabled():
    print("Using pytorch cross attention")
    CrossAttention = CrossAttentionPytorch
else:
    if args.use_split_cross_attention:
        print("Using split optimization for cross attention")
        CrossAttention = CrossAttentionDoggettx
    else:
        print("Using sub quadratic optimization for cross attention, if you have memory or speed issues try using: --use-split-cross-attention")
        CrossAttention = CrossAttentionBirchSan


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, dtype=None, device=None):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, dtype=dtype, device=device)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype, device=device)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, dtype=dtype, device=device)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype, device=device)
        self.checkpoint = checkpoint
        self.n_heads = n_heads
        self.d_head = d_head

    def forward(self, x, context=None, transformer_options={}):
        return checkpoint(self._forward, (x, context, transformer_options), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, transformer_options={}):
        extra_options = {}
        block = None
        block_index = 0
        if "current_index" in transformer_options:
            extra_options["transformer_index"] = transformer_options["current_index"]
        if "block_index" in transformer_options:
            block_index = transformer_options["block_index"]
            extra_options["block_index"] = block_index
        if "original_shape" in transformer_options:
            extra_options["original_shape"] = transformer_options["original_shape"]
        if "block" in transformer_options:
            block = transformer_options["block"]
            extra_options["block"] = block
        if "patches" in transformer_options:
            transformer_patches = transformer_options["patches"]
        else:
            transformer_patches = {}

        extra_options["n_heads"] = self.n_heads
        extra_options["dim_head"] = self.d_head

        if "patches_replace" in transformer_options:
            transformer_patches_replace = transformer_options["patches_replace"]
        else:
            transformer_patches_replace = {}

        n = self.norm1(x)
        if self.disable_self_attn:
            context_attn1 = context
        else:
            context_attn1 = None
        value_attn1 = None

        if "attn1_patch" in transformer_patches:
            patch = transformer_patches["attn1_patch"]
            if context_attn1 is None:
                context_attn1 = n
            value_attn1 = context_attn1
            for p in patch:
                n, context_attn1, value_attn1 = p(n, context_attn1, value_attn1, extra_options)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        attn1_replace_patch = transformer_patches_replace.get("attn1", {})
        block_attn1 = transformer_block
        if block_attn1 not in attn1_replace_patch:
            block_attn1 = block

        if block_attn1 in attn1_replace_patch:
            if context_attn1 is None:
                context_attn1 = n
                value_attn1 = n
            n = self.attn1.to_q(n)
            context_attn1 = self.attn1.to_k(context_attn1)
            value_attn1 = self.attn1.to_v(value_attn1)
            n = attn1_replace_patch[block_attn1](n, context_attn1, value_attn1, extra_options)
            n = self.attn1.to_out(n)
        else:
            n = self.attn1(n, context=context_attn1, value=value_attn1)

        if "attn1_output_patch" in transformer_patches:
            patch = transformer_patches["attn1_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        if "middle_patch" in transformer_patches:
            patch = transformer_patches["middle_patch"]
            for p in patch:
                x = p(x, extra_options)

        n = self.norm2(x)

        context_attn2 = context
        value_attn2 = None
        if "attn2_patch" in transformer_patches:
            patch = transformer_patches["attn2_patch"]
            value_attn2 = context_attn2
            for p in patch:
                n, context_attn2, value_attn2 = p(n, context_attn2, value_attn2, extra_options)

        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block

        if block_attn2 in attn2_replace_patch:
            if value_attn2 is None:
                value_attn2 = context_attn2
            n = self.attn2.to_q(n)
            context_attn2 = self.attn2.to_k(context_attn2)
            value_attn2 = self.attn2.to_v(value_attn2)
            n = attn2_replace_patch[block_attn2](n, context_attn2, value_attn2, extra_options)
            n = self.attn2.to_out(n)
        else:
            n = self.attn2(n, context=context_attn2, value=value_attn2)

        if "attn2_output_patch" in transformer_patches:
            patch = transformer_patches["attn2_output_patch"]
            for p in patch:
                n = p(n, extra_options)

        x += n
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, dtype=None, device=None):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim] * depth
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype, device=device)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0, dtype=dtype, device=device)
        else:
            self.proj_in = comfy.ops.Linear(in_channels, inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, dtype=dtype, device=device)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = nn.Conv2d(inner_dim,in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0, dtype=dtype, device=device)
        else:
            self.proj_out = comfy.ops.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        self.use_linear = use_linear

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


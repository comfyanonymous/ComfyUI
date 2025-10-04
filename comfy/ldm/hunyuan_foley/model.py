from typing import List, Tuple, Optional, Union
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from comfy.ldm.modules.attention import optimized_attention as attention
from comfy.ldm.aura.mmdit import TimestepEmbedder as TimestepEmbedderParent
from comfy.ldm.hydit.posemb_layers import get_1d_rotary_pos_embed

from typing import Union, Tuple

# to get exact matching results
# only difference is the upscale to float32
class RMSNorm(nn.Module):
    def __init__(self, dim: int, elementwise_affine=True, eps: float = 1e-6,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding

class TimestepEmbedder(TimestepEmbedderParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb 

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, device, dtype, operations):
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    ndim = x.ndim
    if head_first:
        shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    cos, sin = cos.to(xq.device), sin.to(xq.device)
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    return xq_out, xk_out

class ConditionProjection(nn.Module):
    def __init__(self, in_channels, hidden_size, dtype=None, device=None, operations = None):
        factory_kwargs = {'dtype': dtype, 'device': device}
        super().__init__()
        self.linear_1 = operations.Linear(in_features=in_channels, out_features=hidden_size, bias=True, **factory_kwargs)
        self.act_1 = nn.SiLU()
        self.linear_2 = operations.Linear(in_features=hidden_size, out_features=hidden_size, bias=True, **factory_kwargs)

    def forward(self, caption):
        return self.linear_2(self.act_1(self.linear_1(caption)))

class PatchEmbed1D(nn.Module):
    def __init__(
        self,
        patch_size=1,
        in_chans=768,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
        operations = None
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.flatten = flatten

        self.proj = operations.Conv1d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias, **factory_kwargs
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.transpose(1, 2)
        x = self.norm(x)
        return x

# avoid classifying as wrapper to work with operations.conv1d
class ChannelLastConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, kernel_size = 3, padding = 0, device=None, dtype=None, operations=None):
        super().__init__()

        operations = operations or nn
        underlying = operations.Conv1d(
            in_channels, out_channels, kernel_size = kernel_size, padding = padding,
            bias=bias, device=device, dtype=dtype
        )

        self.register_parameter("weight", underlying.weight)
        if getattr(underlying, "bias", None) is not None:
            self.register_parameter("bias", underlying.bias)
        else:
            self.register_parameter("bias", None)
        
        object.__setattr__(self, "_underlying", underlying)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._underlying(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class ConvMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
        device=None,
        dtype=None,
        operations = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding, operations = operations, **factory_kwargs)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding, operations = operations, **factory_kwargs)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding, operations = operations, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def modulate(x, shift=None, scale=None):
    if x.ndim == 3:
        shift = shift.unsqueeze(1) if shift is not None and shift.ndim == 2 else None
        scale = scale.unsqueeze(1) if scale is not None and scale.ndim == 2 else None
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale)
    elif scale is None:
        return x + shift
    else:
        return x * (1 + scale) + shift

class ModulateDiT(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None, operations = None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.act = nn.SiLU()
        self.linear = operations.Linear(hidden_size, factor * hidden_size, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(x))
    
class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, device=None, dtype=None, operations = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.norm_final = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.linear = operations.Linear(hidden_size, patch_size * out_channels, bias=True, **factory_kwargs)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), operations.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x
    
class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
        operations = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = (bias, bias)
        drop_probs = (drop, drop)
        linear_layer = partial(operations.Conv2d, kernel_size=1) if use_conv else operations.Linear

        self.fc1 = linear_layer(in_channels, hidden_channels, bias=bias[0], **factory_kwargs)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_channels, out_features, bias=bias[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x)))))) 


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")

def get_meshgrid_nd(start, *args, dim=2):
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)

    return grid

def get_nd_rotary_pos_embed(
    rope_dim_list, start, *args, theta=10000.0, use_real=False, theta_rescale_factor=1.0, freq_scaling=1.0
):
    
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            freq_scaling=freq_scaling,
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        emb = torch.cat(embs, dim=1)
        return emb

def apply_gate(x, gate = None):
    if gate is None:
        return x
    if gate.ndim == 2 and x.ndim == 3:
        gate = gate.unsqueeze(1)
    return x * gate

def interleave_two_sequences(x1: torch.Tensor, x2: torch.Tensor):
    B, N1, H, C = x1.shape
    B, N2, H, C = x2.shape
    assert x1.ndim == x2.ndim == 4

    if N1 != N2:
        x2 = x2.view(B, N2, -1).transpose(1, 2)
        x2 = F.interpolate(x2, size=(N1), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, N1, H, C)
    x = torch.stack((x1, x2), dim=2)
    x = x.reshape(B, N1 * 2, H, C)
    return x

def decouple_interleaved_two_sequences(x: torch.Tensor, len1: int, len2: int):
    B, N, H, C = x.shape
    assert N % 2 == 0 and N // 2 == len1

    x = x.reshape(B, -1, 2, H, C)
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    if x2.shape[1] != len2:
        x2 = x2.view(B, len1, H * C).transpose(1, 2)
        x2 = F.interpolate(x2, size=(len2), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, len2, H, C)
    return x1, x2

def apply_modulated_block(x, norm_layer, shift, scale, mlp_layer, gate):
    x_mod = modulate(norm_layer(x), shift=shift, scale=scale)
    return x + apply_gate(mlp_layer(x_mod), gate=gate)

def prepare_self_attn_qkv(x, norm_layer, qkv_layer, q_norm, k_norm, shift, scale, num_heads):
    x_mod = modulate(norm_layer(x), shift=shift, scale=scale)
    qkv = qkv_layer(x_mod)

    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads)

    q = q_norm(q).to(v)
    k = k_norm(k).to(v)
    return q, k, v

class TwoStreamCABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        interleaved_audio_visual_rope: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations = None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.interleaved_audio_visual_rope = interleaved_audio_visual_rope

        self.audio_mod = ModulateDiT(hidden_size, factor=9, operations = operations, **factory_kwargs)
        self.audio_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.audio_self_attn_qkv = operations.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)

        def make_qk_norm(name: str):
            layer = RMSNorm(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
            setattr(self, name, layer)

        for name in ["v_cond_attn_q_norm", "v_cond_attn_k_norm", "audio_cross_q_norm",
                     "v_cond_cross_q_norm", "text_cross_k_norm", "audio_self_q_norm", "audio_self_k_norm"]:
            make_qk_norm(name)

        self.audio_self_proj = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.v_cond_mod = ModulateDiT(hidden_size, factor = 9, operations = operations, **factory_kwargs)
        self.v_cond_norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_attn_qkv = operations.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)

        self.v_cond_self_proj = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.max_text_len = 100
        self.rope_dim_list = None
        
        self.audio_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.audio_cross_q = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.v_cond_cross_q = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.text_cross_kv = operations.Linear(hidden_size, hidden_size * 2, bias=qkv_bias, **factory_kwargs)
        
        self.audio_cross_proj = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.v_cond_cross_proj = operations.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.audio_norm3 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.audio_mlp = MLP(
            hidden_size, mlp_hidden_dim, bias=True, operations = operations, **factory_kwargs
        )

        self.v_cond_norm3 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_mlp = MLP(
            hidden_size, mlp_hidden_dim, bias=True, operations = operations, **factory_kwargs
        )

    def build_rope_for_text(self, text_len, head_dim, rope_dim_list=None):
        target_ndim = 1  # n-d RoPE
        rope_sizes = [text_len]
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        
        text_freqs_cos, text_freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )
        return text_freqs_cos, text_freqs_sin

    def forward(
        self,
        audio: torch.Tensor,
        cond: torch.Tensor,
        v_cond: torch.Tensor,
        attn_mask: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        v_freqs_cis: tuple = None,
        sync_vec: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        (audio_mod1_shift, audio_mod1_scale, audio_mod1_gate, 
             audio_mod2_shift, audio_mod2_scale, audio_mod2_gate,
             audio_mod3_shift, audio_mod3_scale, audio_mod3_gate,
        ) = self.audio_mod(sync_vec if sync_vec is not None else vec).chunk(9, dim=-1)

        (
            v_cond_mod1_shift,
            v_cond_mod1_scale,
            v_cond_mod1_gate,
            v_cond_mod2_shift,
            v_cond_mod2_scale,
            v_cond_mod2_gate,
            v_cond_mod3_shift,
            v_cond_mod3_scale,
            v_cond_mod3_gate,
        ) = self.v_cond_mod(vec).chunk(9, dim=-1)
        
        audio_q, audio_k, audio_v = prepare_self_attn_qkv(
            audio, self.audio_norm1, self.audio_self_attn_qkv, 
            self.audio_self_q_norm, self.audio_self_k_norm,
            audio_mod1_shift, audio_mod1_scale, self.num_heads
        )

        v_cond_q, v_cond_k, v_cond_v = prepare_self_attn_qkv(
            v_cond, self.v_cond_norm1, self.v_cond_attn_qkv, 
            self.v_cond_attn_q_norm, self.v_cond_attn_k_norm,
            v_cond_mod1_shift, v_cond_mod1_scale, self.num_heads
        )
        
        # Apply RoPE if needed for audio and visual
        if freqs_cis is not None:
            if not self.interleaved_audio_visual_rope:
                audio_qq, audio_kk = apply_rotary_emb(audio_q, audio_k, freqs_cis, head_first=False)
                audio_q, audio_k = audio_qq, audio_kk
            else:
                ori_audio_len = audio_q.shape[1]
                ori_v_con_len = v_cond_q.shape[1]
                interleaved_audio_visual_q = interleave_two_sequences(audio_q, v_cond_q)
                interleaved_audio_visual_k = interleave_two_sequences(audio_k, v_cond_k)
                interleaved_audio_visual_qq, interleaved_audio_visual_kk = apply_rotary_emb(
                    interleaved_audio_visual_q, interleaved_audio_visual_k, freqs_cis, head_first=False
                )
                audio_qq, v_cond_qq = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_qq, ori_audio_len, ori_v_con_len
                )
                audio_kk, v_cond_kk = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_kk, ori_audio_len, ori_v_con_len
                )
                audio_q, audio_k = audio_qq, audio_kk
                v_cond_q, v_cond_k = v_cond_qq, v_cond_kk

        if v_freqs_cis is not None and not self.interleaved_audio_visual_rope:
            v_cond_qq, v_cond_kk = apply_rotary_emb(v_cond_q, v_cond_k, v_freqs_cis, head_first=False)
            v_cond_q, v_cond_k = v_cond_qq, v_cond_kk
        
        q = torch.cat((v_cond_q, audio_q), dim=1)
        k = torch.cat((v_cond_k, audio_k), dim=1)
        v = torch.cat((v_cond_v, audio_v), dim=1)
        
        # TODO: look further into here
        if attention.__name__ == "attention_pytorch":
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        attn = attention(q, k, v, heads = self.num_heads, mask=attn_mask, skip_reshape=True)
        v_cond_attn, audio_attn = torch.split(attn, [v_cond.shape[1], audio.shape[1]], dim=1)
        
        audio = audio + apply_gate(self.audio_self_proj(audio_attn), gate=audio_mod1_gate)
        v_cond = v_cond + apply_gate(self.v_cond_self_proj(v_cond_attn), gate=v_cond_mod1_gate)
        head_dim = self.hidden_size // self.num_heads

        audio_q = self.prepare_modulated_query(audio, self.audio_norm2, self.audio_cross_q,
                                        self.audio_cross_q_norm, audio_mod2_shift, audio_mod2_scale,
                                        self.num_heads, self.rope_dim_list)

        v_cond_q = self.prepare_modulated_query(v_cond, self.v_cond_norm2, self.v_cond_cross_q,
                                        self.v_cond_cross_q_norm, v_cond_mod2_shift, v_cond_mod2_scale,
                                        self.num_heads, self.rope_dim_list)

        text_kv = self.text_cross_kv(cond)
        text_k, text_v = rearrange(text_kv, "B L (K H D) -> K B L H D", K=2, H=self.num_heads)
        text_k = self.text_cross_k_norm(text_k).to(text_v)

        text_len = text_k.shape[1]
        
        text_freqs_cos, text_freqs_sin = self.build_rope_for_text(text_len, head_dim, 
                                                                 rope_dim_list=self.rope_dim_list)
        text_freqs_cis = (text_freqs_cos.to(text_k.device), text_freqs_sin.to(text_k.device))
        text_k = apply_rotary_emb(text_k, text_k, text_freqs_cis, head_first=False)[1]
        
        v_cond_audio_q = torch.cat([v_cond_q, audio_q], dim=1)

        if attention.__name__ == "attention_pytorch":
            v_cond_audio_q, text_k, text_v = [t.transpose(1, 2) for t in (v_cond_audio_q, text_k, text_v)]

        cross_attn = attention(v_cond_audio_q, text_k, text_v, self.num_heads, skip_reshape = True)
        v_cond_cross_attn, audio_cross_attn = torch.split(cross_attn, [v_cond.shape[1], audio.shape[1]], dim=1)
        
        audio = audio + apply_gate(self.audio_cross_proj(audio_cross_attn), gate=audio_mod2_gate)
        v_cond = v_cond + apply_gate(self.v_cond_cross_proj(v_cond_cross_attn), gate=v_cond_mod2_gate)

        audio = apply_modulated_block(audio, self.audio_norm3, audio_mod3_shift, audio_mod3_scale, self.audio_mlp, audio_mod3_gate)
        v_cond = apply_modulated_block(v_cond, self.v_cond_norm3, v_cond_mod3_shift, v_cond_mod3_scale, self.v_cond_mlp, v_cond_mod3_gate)

        return audio, cond, v_cond
    
    def prepare_modulated_query(self, x, norm_layer, q_layer, q_norm_layer, shift, scale, num_heads, rope_dim_list):

        x_mod = modulate(norm_layer(x), shift=shift, scale=scale)
        q = q_layer(x_mod)

        q = rearrange(q, "B L (H D) -> B L H D", H=num_heads)
        q = q_norm_layer(q)

        head_dim = q.shape[-1]
        freqs_cos, freqs_sin = self.build_rope_for_text(q.shape[1], head_dim, rope_dim_list)
        freqs_cis = (freqs_cos.to(q.device), freqs_sin.to(q.device))
        
        q = apply_rotary_emb(q, q, freqs_cis, head_first=False)[0]
        
        return q

class SingleStreamBlock(nn.Module):

    def __init__(self, hidden_size: int,
                    num_heads: int,
                    mlp_ratio: float,
                    dtype: Optional[torch.dtype] = None,
                    device: Optional[torch.device] = None,
                    operations = None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.modulation = ModulateDiT(
            hidden_size=hidden_size,
            factor=6,
            operations = operations,
            **factory_kwargs,
        )
        self.linear_qkv = operations.Linear(hidden_size, hidden_size * 3, bias=True)
        self.linear1 = ChannelLastConv1d(hidden_size, hidden_size, kernel_size=3, padding=1, operations = operations, **factory_kwargs)
        self.linear2 = ConvMLP(hidden_size, hidden_size * mlp_ratio, kernel_size=3, padding=1, operations = operations, **factory_kwargs)
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, **factory_kwargs)
        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, **factory_kwargs)
        self.q_norm = operations.RMSNorm(hidden_size // num_heads, **factory_kwargs)
        self.k_norm = operations.RMSNorm(hidden_size // num_heads, **factory_kwargs)
        self.rearrange = Rearrange("B L (H D K) -> B H L D K", K=3, H=num_heads)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None):

        modulation = self.modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x_norm1 = self.norm1(x) * (1 + scale_msa) + shift_msa

        qkv = self.linear_qkv(x_norm1)
        q, k, v = self.rearrange(qkv).chunk(3, dim=-1)

        q, k, v = [t.squeeze(-1) for t in (q, k, v)]

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis, head_first=True)

        q, k, v = [t.contiguous() for t in (q, k, v)]

        out = attention(q, k, v, self.num_heads, skip_output_reshape = True, skip_reshape = True)
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        x = x + apply_gate(self.linear1(out),gate=gate_msa)
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + apply_gate(self.linear2(x_norm), gate=gate_mlp)

        return x

def _ceil_div(a, b):
    return (a + b - 1) // b

def find_period_by_first_row(mat):

    L, _ = mat.shape

    first = mat[0:1]
    matches = (mat[1:] == first).all(dim=1)
    candidate_positions = (torch.nonzero(matches).squeeze(-1) + 1).tolist()
    if isinstance(candidate_positions, int):
        candidate_positions = [candidate_positions]
    if not candidate_positions:
        return L

    for p in sorted(candidate_positions):
        base = mat[:p]
        reps = _ceil_div(L, p)
        tiled = base.repeat(reps, 1)[:L]
        if torch.equal(tiled, mat):
            return p

    for p in range(1, L + 1):
        base = mat[:p]
        reps = _ceil_div(L, p)
        tiled = base.repeat(reps, 1)[:L]
        if torch.equal(tiled, mat):
            return p

    return L

def trim_repeats(expanded):
    seq = expanded[0]
    p_len = find_period_by_first_row(seq)

    seq_T = seq.transpose(0, 1)
    p_dim = find_period_by_first_row(seq_T)

    return expanded[:, :p_len, :p_dim]

class HunyuanVideoFoley(nn.Module):
    def __init__(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        operations = None,
        **kwargs
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dtype = dtype

        self.depth_triple_blocks = 18
        self.depth_single_blocks = 36
        model_args = {}

        self.interleaved_audio_visual_rope = model_args.get("interleaved_audio_visual_rope", True)

        self.condition_dim = model_args.get("condition_dim", 768)

        self.patch_size = model_args.get("patch_size", 1)
        self.visual_in_channels = model_args.get("clip_dim", 768)
        self.audio_vae_latent_dim = model_args.get("audio_vae_latent_dim", 128)
        self.out_channels = self.audio_vae_latent_dim 
        self.unpatchify_channels = self.out_channels

        self.num_heads = model_args.get("num_heads", 12)
        self.hidden_size = model_args.get("hidden_size", 1536)
        self.rope_dim_list = model_args.get("rope_dim_list", None)
        self.mlp_ratio = model_args.get("mlp_ratio", 4.0)

        self.qkv_bias = model_args.get("qkv_bias", True)
        self.qk_norm = model_args.get("qk_norm", True)

        # sync condition things
        self.sync_modulation = model_args.get("sync_modulation", False)
        self.add_sync_feat_to_audio = model_args.get("add_sync_feat_to_audio", True)
        self.sync_feat_dim = model_args.get("sync_feat_dim", 768)
        self.sync_in_ksz = model_args.get("sync_in_ksz", 1)

        self.clip_len = model_args.get("clip_length", 64)
        self.sync_len = model_args.get("sync_length", 192)

        self.patch_size = 1
        self.audio_embedder = PatchEmbed1D(self.patch_size, self.audio_vae_latent_dim, self.hidden_size, operations=operations, **factory_kwargs)
        self.visual_proj = SwiGLU(dim = self.visual_in_channels, hidden_dim = self.hidden_size, device=device, dtype=dtype, operations=operations)

        self.cond_in = ConditionProjection(
            self.condition_dim, self.hidden_size, operations=operations, **factory_kwargs
        )

        self.time_in = TimestepEmbedder(self.hidden_size, operations = operations, **factory_kwargs)

        # visual sync embedder if needed
        if self.sync_in_ksz == 1:
            sync_in_padding = 0
        elif self.sync_in_ksz == 3:
            sync_in_padding = 1
        else:
            raise ValueError
        if self.sync_modulation or self.add_sync_feat_to_audio:
            self.sync_in = nn.Sequential(
                operations.Linear(self.sync_feat_dim, self.hidden_size, **factory_kwargs),
                nn.SiLU(),
                ConvMLP(self.hidden_size, self.hidden_size * 4, kernel_size=self.sync_in_ksz, padding=sync_in_padding, operations=operations, **factory_kwargs),
            )
            self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, self.sync_feat_dim), **factory_kwargs))

        self.triple_blocks = nn.ModuleList(
            [
                TwoStreamCABlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm=self.qk_norm,
                    qkv_bias=self.qkv_bias,
                    interleaved_audio_visual_rope=self.interleaved_audio_visual_rope,
                    operations=operations,
                    **factory_kwargs,
                )
                for _ in range(self.depth_triple_blocks)
            ]
        )


        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    operations=operations,
                    **factory_kwargs,
                )
                for _ in range(self.depth_single_blocks)
            ]
        )

        self.final_layer = FinalLayer1D(
            self.hidden_size, self.patch_size, self.out_channels, operations = operations,**factory_kwargs
        )
        self.unpatchify_channels = self.out_channels

        self.empty_clip_feat = nn.Parameter(torch.zeros(1, self.visual_in_channels, **factory_kwargs), requires_grad = False)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, self.sync_feat_dim, **factory_kwargs), requires_grad = False)

    def get_empty_clip_sequence(self, bs=None, len=None) -> torch.Tensor:
        len = len if len is not None else self.clip_len
        if bs is None:
            return self.empty_clip_feat.expand(len, -1)  # 15s
        else:
            return self.empty_clip_feat.unsqueeze(0).expand(bs, len, -1)  # 15s

    def get_empty_sync_sequence(self, bs=None, len=None) -> torch.Tensor:
        len = len if len is not None else self.sync_len
        if bs is None:
            return self.empty_sync_feat.expand(len, -1)
        else:
            return self.empty_sync_feat.unsqueeze(0).expand(bs, len, -1)

    def build_rope_for_audio_visual(self, audio_emb_len, visual_cond_len):
        target_ndim = 1  # n-d RoPE
        rope_sizes = [audio_emb_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )

        target_ndim = 1
        rope_sizes = [visual_cond_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        v_freqs_cos, v_freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
            freq_scaling=1.0 * audio_emb_len / visual_cond_len,
        )
        return freqs_cos, freqs_sin, v_freqs_cos, v_freqs_sin

    def build_rope_for_interleaved_audio_visual(self, total_len):
        target_ndim = 1  # n-d RoPE
        rope_sizes = [total_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )
        return freqs_cos, freqs_sin

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        control = None,
        transformer_options = {},
        drop_visual: Optional[List[bool]] = None,
    ):
        device = x.device
        audio = x
        bs, _, ol = x.shape
        tl = ol // self.patch_size

        condition, uncondition = torch.chunk(context, 2)

        condition = condition.view(3, context.size(1) // 3, -1)
        uncondition = uncondition.view(3, context.size(1) // 3, -1)

        uncond_1, uncond_2, cond_neg = torch.chunk(uncondition, 3)
        clip_feat, sync_feat, cond_pos = torch.chunk(condition, 3)
        cond_neg, clip_feat, sync_feat, cond_pos = [trim_repeats(t) for t in (cond_neg, clip_feat, sync_feat, cond_pos)]

        uncond_1 = uncond_1[:, :clip_feat.size(1), :clip_feat.size(2)]
        uncond_2 = uncond_2[:, :sync_feat.size(1), :sync_feat.size(2)]
        
        uncond_1, uncond_2, cond_neg, clip_feat, sync_feat, cond_pos = [t.to(device, allow_gpu=True) for t in (uncond_1, uncond_2, cond_neg, clip_feat, sync_feat, cond_pos)]

        clip_feat, sync_feat, cond = torch.cat([uncond_1, clip_feat]), torch.cat([uncond_2, sync_feat]), torch.cat([cond_neg, cond_pos])

        if drop_visual is not None:
            clip_feat[drop_visual] = self.get_empty_clip_sequence().to(dtype=clip_feat.dtype)
            sync_feat[drop_visual] = self.get_empty_sync_sequence().to(dtype=sync_feat.dtype)

        vec = self.time_in(t)
        sync_vec = None
        if self.add_sync_feat_to_audio:
            sync_feat = sync_feat.view(bs, sync_feat.shape[1] // 8, 8, self.sync_feat_dim) + self.sync_pos_emb.to(sync_feat.device)
            sync_feat = sync_feat.view(bs, -1, self.sync_feat_dim)
            sync_feat = self.sync_in.to(sync_feat.device)(sync_feat)
            add_sync_feat_to_audio = (
                F.interpolate(sync_feat.transpose(1, 2), size=(tl), mode="nearest-exact").contiguous().transpose(1, 2)
            )

        cond = self.cond_in(cond)
        cond_seq_len = cond.shape[1]

        audio = self.audio_embedder(x)
        audio_seq_len = audio.shape[1]
        v_cond = self.visual_proj(clip_feat)
        v_cond_seq_len = v_cond.shape[1]
        attn_mask = None


        freqs_cos, freqs_sin = self.build_rope_for_interleaved_audio_visual(audio_seq_len * 2)
        v_freqs_cos = v_freqs_sin = None
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        v_freqs_cis = (v_freqs_cos, v_freqs_sin) if v_freqs_cos is not None else None

        if self.add_sync_feat_to_audio:
            add_sync_layer = 0

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        def block_wrap(**kwargs):
            return block(**kwargs)

        for layer_num, block in enumerate(self.triple_blocks):
            if self.add_sync_feat_to_audio and layer_num == add_sync_layer:
                audio = audio + add_sync_feat_to_audio
            triple_block_args = [audio, cond, v_cond, attn_mask, vec, freqs_cis, v_freqs_cis, sync_vec]
            if ("triple_block", layer_num) in blocks_replace:
                audio, cond, v_cond = blocks_replace[("triple_block", layer_num)]({
                    "audio": triple_block_args[0],
                    "cond": triple_block_args[1],
                    "v_cond": triple_block_args[2],
                    "attn_mask": triple_block_args[3],
                    "vec": triple_block_args[4],
                    "freqs_cis": triple_block_args[5],
                    "v_freqs_cis": triple_block_args[6],
                    "sync_vec": triple_block_args[7]
                }, {"original_block": block_wrap})
            else:
                audio, cond, v_cond = block(*triple_block_args)

        x = audio 
        if sync_vec is not None:
            vec = vec.unsqueeze(1).repeat(1, cond_seq_len + v_cond_seq_len, 1)
            vec = torch.cat((vec, sync_vec), dim=1)

        freqs_cos, freqs_sin, _, _ = self.build_rope_for_audio_visual(audio_seq_len, v_cond_seq_len)
        if self.add_sync_feat_to_audio:
            vec = add_sync_feat_to_audio + vec.unsqueeze(dim=1)
        if len(self.single_blocks) > 0:
            for layer_num, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    (freqs_cos, freqs_sin),
                ]
                if ("single_block", layer_num) in blocks_replace:
                    x = blocks_replace[("single_block", layer_num)]({
                        "x": single_block_args[0],
                        "vec": single_block_args[1],
                        "freqs_cis": single_block_args[2]
                    }, {"original_block": block_wrap})
                else:
                    x = block(*single_block_args)

        audio = x

        if sync_vec is not None:
            vec = sync_vec
        audio = self.final_layer(audio, vec)
        audio = self.unpatchify1d(audio, tl)

        uncond, cond = torch.chunk(2, audio)
        return torch.cat([cond, uncond])

    def unpatchify1d(self, x, l):
        c = self.unpatchify_channels
        p = self.patch_size

        x = x.reshape(shape=(x.shape[0], l, p, c))
        x = torch.einsum("ntpc->nctp", x)
        audio = x.reshape(shape=(x.shape[0], c, l * p))
        return audio

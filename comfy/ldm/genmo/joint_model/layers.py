#original code from https://github.com/genmoai/models under apache 2.0 license
#adapted to ComfyUI

import collections.abc
import math
from itertools import repeat
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import comfy.ldm.common_dit


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        bias: bool = True,
        timestep_scale: Optional[float] = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            operations.Linear(frequency_embedding_size, hidden_size, bias=bias, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(hidden_size, hidden_size, bias=bias, dtype=dtype, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_scale = timestep_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        freqs.mul_(-math.log(max_period) / half).exp_()
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, out_dtype):
        if self.timestep_scale is not None:
            t = t * self.timestep_scale
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype=out_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device: Optional[torch.device] = None,
        dtype=None,
        operations=None,
    ):
        super().__init__()
        # keep parameter count and computation constant compared to standard FFN
        hidden_size = int(2 * hidden_size / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_size
        self.w1 = operations.Linear(in_features, 2 * hidden_size, bias=False, device=device, dtype=dtype)
        self.w2 = operations.Linear(hidden_size, in_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(x) * gate)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = operations.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        assert norm_layer is None
        self.norm = (
            norm_layer(embed_dim, device=device) if norm_layer else nn.Identity()
        )

    def forward(self, x):
        B, _C, T, H, W = x.shape
        if not self.dynamic_img_pad:
            assert H % self.patch_size[0] == 0, f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
            assert W % self.patch_size[1] == 0, f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
        else:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))

        x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size, padding_mode='circular')
        x = self.proj(x)

        # Flatten temporal and spatial dimensions.
        if not self.flatten:
            raise NotImplementedError("Must flatten output.")
        x = rearrange(x, "(B T) C H W -> B (T H W) C", B=B, T=T)

        x = self.norm(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(self, x):
        return comfy.ldm.common_dit.rms_norm(x, self.weight, self.eps)

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OSA.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:07:42 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn

from .layernorm import LayerNorm2d

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


# helper classes


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1, 0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1, 1, 0),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.0):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# MBConv


class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange("b c -> b c 1 1"),
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.0):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0.0 or (not self.training):
            return x

        keep_mask = (
            torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_()
            > self.prob
        )
        return x * keep_mask / (1 - self.prob)


def MBConv(
    dim_in, dim_out, *, downsample, expansion_rate=4, shrinkage_rate=0.25, dropout=0.0
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(
            hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim
        ),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.0,
        window_size=7,
        with_pe=True,
    ):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False), nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, "c i j -> (i j) c")
            rel_pos = rearrange(grid, "i ... -> i 1 ...") - rearrange(
                grid, "j ... -> 1 j ..."
            )
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(
                dim=-1
            )

            self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = (
            *x.shape,
            x.device,
            self.heads,
        )

        # flatten

        x = rearrange(x, "b x y w1 w2 d -> (b x y) (w1 w2) d")

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, "b n (h d ) -> b h n d", h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, "i j h -> h i j")

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(
            out, "b h (w1 w2) d -> b w1 w2 (h d)", w1=window_height, w2=window_width
        )

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, "(b x y) ... -> b x y ...", x=height, y=width)


class Block_Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        bias=False,
        dropout=0.0,
        window_size=7,
        with_pe=True,
    ):
        super().__init__()
        assert (
            dim % dim_head
        ) == 0, "dimension should be divisible by dimension per head"

        self.heads = dim // dim_head
        self.ps = window_size
        self.scale = dim_head**-0.5
        self.with_pe = with_pe

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )

        self.attend = nn.Sequential(nn.Softmax(dim=-1), nn.Dropout(dropout))

        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # project for queries, keys, values
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # split heads

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d",
                h=self.heads,
                w1=self.ps,
                w2=self.ps,
            ),
            (q, k, v),
        )

        # scale

        q = q * self.scale

        # sim

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # attention
        attn = self.attend(sim)

        # aggregate

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads
        out = rearrange(
            out,
            "(b x y) head (w1 w2) d -> b (head d) (x w1) (y w2)",
            x=h // self.ps,
            y=w // self.ps,
            head=self.heads,
            w1=self.ps,
            w2=self.ps,
        )

        out = self.to_out(out)
        return out


class Channel_Attention(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            ),
            qkv,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            "b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):
    def __init__(self, dim, heads, bias=False, dropout=0.0, window_size=7):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)",
                ph=self.ps,
                pw=self.ps,
                head=self.heads,
            ),
            qkv,
        )

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v

        out = rearrange(
            out,
            "b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)",
            h=h // self.ps,
            w=w // self.ps,
            ph=self.ps,
            pw=self.ps,
            head=self.heads,
        )

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(
        self,
        channel_num=64,
        bias=True,
        ffn_bias=True,
        window_size=8,
        with_pe=False,
        dropout=0.0,
    ):
        super(OSA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25,
            ),
            Rearrange(
                "b d (x w1) (y w2) -> b x y w1 w2 d", w1=w, w2=w
            ),  # block-like attention
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange("b x y w1 w2 d -> b d (x w1) (y w2)"),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention(
                    dim=channel_num, heads=4, dropout=dropout, window_size=window_size
                ),
            ),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            Rearrange(
                "b d (w1 x) (w2 y) -> b x y w1 w2 d", w1=w, w2=w
            ),  # grid-like attention
            PreNormResidual(
                channel_num,
                Attention(
                    dim=channel_num,
                    dim_head=channel_num // 4,
                    dropout=dropout,
                    window_size=window_size,
                    with_pe=with_pe,
                ),
            ),
            Rearrange("b x y w1 w2 d -> b d (w1 x) (w2 y)"),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
            # channel-like attention
            Conv_PreNormResidual(
                channel_num,
                Channel_Attention_grid(
                    dim=channel_num, heads=4, dropout=dropout, window_size=window_size
                ),
            ),
            Conv_PreNormResidual(
                channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)
            ),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

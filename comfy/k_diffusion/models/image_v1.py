import math

import torch
from torch import nn
from torch.nn import functional as F

from .. import layers, utils


def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module


class ResConvBlock(layers.ConditionedResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else orthogonal_(nn.Conv2d(c_in, c_out, 1, bias=False))
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.GELU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            skip=skip)


class DBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., downsample=False, self_attn=False, cross_attn=False, c_enc=0):
        modules = [nn.Identity()]
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.CrossAttention2d(my_c_out, c_enc, max(1, my_c_out // head_size), norm, dropout_rate))
        super().__init__(*modules)
        self.set_downsample(downsample)

    def set_downsample(self, downsample):
        self[0] = layers.Downsample2d() if downsample else nn.Identity()
        return self


class UBlock(layers.ConditionedSequential):
    def __init__(self, n_layers, feats_in, c_in, c_mid, c_out, group_size=32, head_size=64, dropout_rate=0., upsample=False, self_attn=False, cross_attn=False, c_enc=0):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, max(1, my_c_out // head_size), norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.CrossAttention2d(my_c_out, c_enc, max(1, my_c_out // head_size), norm, dropout_rate))
        modules.append(nn.Identity())
        super().__init__(*modules)
        self.set_upsample(upsample)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = torch.cat([input, skip], dim=1)
        return super().forward(input, cond)

    def set_upsample(self, upsample):
        self[-1] = layers.Upsample2d() if upsample else nn.Identity()
        return self


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)


class ImageDenoiserModelV1(nn.Module):
    def __init__(self, c_in, feats_in, depths, channels, self_attn_depths, cross_attn_depths=None, mapping_cond_dim=0, unet_cond_dim=0, cross_cond_dim=0, dropout_rate=0., patch_size=1, skip_stages=0, has_variance=False):
        super().__init__()
        self.c_in = c_in
        self.channels = channels
        self.unet_cond_dim = unet_cond_dim
        self.patch_size = patch_size
        self.has_variance = has_variance
        self.timestep_embed = layers.FourierFeatures(1, feats_in)
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv2d((c_in + unet_cond_dim) * self.patch_size ** 2, channels[max(0, skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(channels[max(0, skip_stages - 1)], c_in * self.patch_size ** 2 + (1 if self.has_variance else 0), 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        if cross_cond_dim == 0:
            cross_attn_depths = [False] * len(self_attn_depths)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[max(0, i - 1)]
            d_blocks.append(DBlock(depths[i], feats_in, my_c_in, channels[i], channels[i], downsample=i > skip_stages, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[max(0, i - 1)]
            u_blocks.append(UBlock(depths[i], feats_in, my_c_in, channels[i], my_c_out, upsample=i > skip_stages, self_attn=self_attn_depths[i], cross_attn=cross_attn_depths[i], c_enc=cross_cond_dim, dropout_rate=dropout_rate))
        self.u_net = layers.UNet(d_blocks, reversed(u_blocks), skip_stages=skip_stages)

    def forward(self, input, sigma, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None, return_variance=False):
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(utils.append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}
        if unet_cond is not None:
            input = torch.cat([input, unet_cond], dim=1)
        if cross_cond is not None:
            cond['cross'] = cross_cond
            cond['cross_padding'] = cross_cond_padding
        if self.patch_size > 1:
            input = F.pixel_unshuffle(input, self.patch_size)
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        if self.has_variance:
            input, logvar = input[:, :-1], input[:, -1].flatten(1).mean(1)
        if self.patch_size > 1:
            input = F.pixel_shuffle(input, self.patch_size)
        if self.has_variance and return_variance:
            return input, logvar
        return input

    def set_skip_stages(self, skip_stages):
        self.proj_in = nn.Conv2d(self.proj_in.in_channels, self.channels[max(0, skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, skip_stages - 1)], self.proj_out.out_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.u_net.skip_stages = skip_stages
        for i, block in enumerate(self.u_net.d_blocks):
            block.set_downsample(i > skip_stages)
        for i, block in enumerate(reversed(self.u_net.u_blocks)):
            block.set_upsample(i > skip_stages)
        return self

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.proj_in = nn.Conv2d((self.c_in + self.unet_cond_dim) * self.patch_size ** 2, self.channels[max(0, self.u_net.skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, self.u_net.skip_stages - 1)], self.c_in * self.patch_size ** 2 + (1 if self.has_variance else 0), 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

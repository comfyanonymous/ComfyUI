# Torch-native reimplementation of the Hunyuan3D 2.1 paint (hunyuan3d-paintpbr-v2-1)
# UNet2p5DConditionModel. Ported from Tencent's hy3dpaint reference
# (hunyuanpaintpbr/unet/modules.py), which wraps a Stable-Diffusion-2 style
# diffusers UNet2DConditionModel with multiview / reference / material / DINO
# attention. No diffusers dependency: the SD2 UNet backbone is reimplemented here
# with module + parameter names matching the diffusers layout so the released
# checkpoint state_dict maps onto these modules.
#
# Reference: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT.

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import Attention, SelfAttnProcessor, RefAttnProcessor, calc_multires_voxel_idxs


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0.0, max_period=10000):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # flip_sin_to_cos=True
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        return get_timestep_embedding(timesteps, self.num_channels)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels, time_embed_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_channels, time_embed_dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = F.silu(sample)
        sample = self.linear_2(sample)
        return sample


# ---------------------------------------------------------------------------
# Resnet / sampling blocks
# ---------------------------------------------------------------------------
class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, groups=32, eps=1e-5,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.GroupNorm(groups, in_channels, eps=eps, affine=True, dtype=dtype, device=device)
        self.conv1 = operations.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, dtype=dtype, device=device)
        self.time_emb_proj = operations.Linear(temb_channels, out_channels, dtype=dtype, device=device)
        self.norm2 = operations.GroupNorm(groups, out_channels, eps=eps, affine=True, dtype=dtype, device=device)
        self.conv2 = operations.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, dtype=dtype, device=device)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = operations.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, dtype=dtype, device=device)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        temb = self.time_emb_proj(F.silu(temb))[:, :, None, None]
        h = h + temb
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class Downsample2D(nn.Module):
    def __init__(self, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, stride=2, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(channels, channels, 3, stride=1, padding=1, dtype=dtype, device=device)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList([
            GEGLU(dim, inner_dim, dtype=dtype, device=device, operations=operations),
            nn.Dropout(0.0),
            operations.Linear(inner_dim, dim, dtype=dtype, device=device),
        ])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class BasicTransformerBlock(nn.Module):
    """SD2-style transformer block (self attn, cross attn, GEGLU FFN)."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim,
                 material_processor=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.LayerNorm(dim, elementwise_affine=True, dtype=dtype, device=device)
        self.attn1 = Attention(dim, num_attention_heads, attention_head_dim, cross_attention_dim=None,
                               bias=False, processor=material_processor,
                               dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.LayerNorm(dim, elementwise_affine=True, dtype=dtype, device=device)
        self.attn2 = Attention(dim, num_attention_heads, attention_head_dim, cross_attention_dim=cross_attention_dim,
                               bias=False, dtype=dtype, device=device, operations=operations)
        self.norm3 = operations.LayerNorm(dim, elementwise_affine=True, dtype=dtype, device=device)
        self.ff = FeedForward(dim, dtype=dtype, device=device, operations=operations)


class Basic2p5DTransformerBlock(nn.Module):
    """Wraps a BasicTransformerBlock with multiview / reference / material / DINO
    attention, matching the reference module layout (``.transformer`` holds the
    original block; ``attn_multiview``/``attn_refview``/``attn_dino`` are siblings)."""

    def __init__(self, dim, num_attention_heads, attention_head_dim, cross_attention_dim, layer_name,
                 use_ma=True, use_ra=True, use_mda=True, use_dino=True, pbr_setting=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.layer_name = layer_name
        self.use_ma = use_ma
        self.use_ra = use_ra
        self.use_mda = use_mda
        self.use_dino = use_dino
        self.pbr_setting = pbr_setting
        inner_dim = num_attention_heads * attention_head_dim

        material_processor = None
        if use_mda and pbr_setting is not None:
            material_processor = SelfAttnProcessor(
                dim, inner_dim, dim, pbr_setting, bias=False, out_bias=True,
                dtype=dtype, device=device, operations=operations)

        self.transformer = BasicTransformerBlock(
            dim, num_attention_heads, attention_head_dim, cross_attention_dim,
            material_processor=material_processor, dtype=dtype, device=device, operations=operations)

        if use_ma:
            self.attn_multiview = Attention(dim, num_attention_heads, attention_head_dim, cross_attention_dim=None,
                                            bias=False, dtype=dtype, device=device, operations=operations)
        if use_ra:
            ref_processor = RefAttnProcessor(dim, inner_dim, dim, pbr_setting, bias=False, out_bias=True,
                                             dtype=dtype, device=device, operations=operations)
            self.attn_refview = Attention(dim, num_attention_heads, attention_head_dim, cross_attention_dim=None,
                                          bias=False, processor=ref_processor,
                                          dtype=dtype, device=device, operations=operations)
        if use_dino:
            self.attn_dino = Attention(dim, num_attention_heads, attention_head_dim,
                                       cross_attention_dim=cross_attention_dim, bias=False,
                                       dtype=dtype, device=device, operations=operations)

    def forward(self, hidden_states, encoder_hidden_states, num_in_batch=1, mode=None,
                mva_scale=1.0, ref_scale=1.0, condition_embed_dict=None,
                dino_hidden_states=None, position_voxel_indices=None):
        t = self.transformer
        N_pbr = len(self.pbr_setting) if self.pbr_setting is not None else 1

        norm_hidden_states = t.norm1(hidden_states)

        # 1. material-dimension self attention
        if self.use_mda:
            mda = rearrange(norm_hidden_states, "(b n_pbr n) l c -> b n_pbr n l c", n=num_in_batch, n_pbr=N_pbr)
            attn_output = t.attn1.forward_material_self(mda, self.pbr_setting)
            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")
        else:
            attn_output = t.attn1(norm_hidden_states)
        hidden_states = attn_output + hidden_states

        # reference write
        if mode is not None and "w" in mode:
            condition_embed_dict[self.layer_name] = rearrange(
                norm_hidden_states, "(b n) l c -> b (n l) c", n=num_in_batch)

        # reference read
        if mode is not None and "r" in mode and self.use_ra:
            condition_embed = condition_embed_dict[self.layer_name]
            ref_norm = rearrange(norm_hidden_states, "(b n_pbr n) l c -> b n_pbr (n l) c",
                                 n=num_in_batch, n_pbr=N_pbr)[:, 0, ...]
            attn_output = self.attn_refview.forward_ref(ref_norm, condition_embed, self.pbr_setting)
            attn_output = rearrange(attn_output, "b n_pbr (n l) c -> (b n_pbr n) l c", n=num_in_batch, n_pbr=N_pbr)
            ref_scale_timing = ref_scale
            if isinstance(ref_scale, torch.Tensor):
                ref_scale_timing = ref_scale.unsqueeze(1).repeat(1, num_in_batch * N_pbr).view(-1)
                for _ in range(attn_output.ndim - 1):
                    ref_scale_timing = ref_scale_timing.unsqueeze(-1)
            hidden_states = ref_scale_timing * attn_output + hidden_states

        # multiview attention
        if num_in_batch > 1 and self.use_ma:
            mv = rearrange(norm_hidden_states, "(b n_pbr n) l c -> (b n_pbr) (n l) c", n_pbr=N_pbr, n=num_in_batch)
            position_indices = None
            if position_voxel_indices is not None and mv.shape[1] in position_voxel_indices:
                position_indices = position_voxel_indices[mv.shape[1]]
            attn_output = self.attn_multiview.forward_multiview(mv, position_indices=position_indices, n_pbrs=N_pbr)
            attn_output = rearrange(attn_output, "(b n_pbr) (n l) c -> (b n_pbr n) l c", n_pbr=N_pbr, n=num_in_batch)
            hidden_states = mva_scale * attn_output + hidden_states

        # cross attention (text / learned clip)
        norm_hidden_states = t.norm2(hidden_states)
        attn_output = t.attn2(norm_hidden_states, encoder_hidden_states)
        hidden_states = attn_output + hidden_states

        # dino attention
        if self.use_dino and dino_hidden_states is not None:
            dino = dino_hidden_states.unsqueeze(1).repeat(1, N_pbr * num_in_batch, 1, 1)
            dino = rearrange(dino, "b n l c -> (b n) l c")
            attn_output = self.attn_dino(norm_hidden_states, dino)
            hidden_states = attn_output + hidden_states

        # feed forward
        norm_hidden_states = t.norm3(hidden_states)
        hidden_states = t.ff(norm_hidden_states) + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    """SD2 Transformer2DModel with linear in/out projection."""

    def __init__(self, in_channels, num_attention_heads, attention_head_dim, cross_attention_dim,
                 num_layers=1, groups=32, block_ctor=None, layer_name=None, block_kwargs=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = operations.GroupNorm(groups, in_channels, eps=1e-6, affine=True, dtype=dtype, device=device)  # noqa: E501
        self.proj_in = operations.Linear(in_channels, inner_dim, dtype=dtype, device=device)
        block_kwargs = block_kwargs or {}
        self.transformer_blocks = nn.ModuleList([
            block_ctor(inner_dim, num_attention_heads, attention_head_dim, cross_attention_dim,
                       layer_name=f"{layer_name}_{k}", dtype=dtype, device=device, operations=operations,
                       **block_kwargs)
            for k in range(num_layers)
        ])
        self.proj_out = operations.Linear(inner_dim, in_channels, dtype=dtype, device=device)

    def forward(self, hidden_states, encoder_hidden_states, **block_kwargs):
        b, c, h, w = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(b, h * w, c)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, **block_kwargs)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return hidden_states + residual


# ---------------------------------------------------------------------------
# Down / mid / up blocks
# ---------------------------------------------------------------------------
def _make_transformer(in_channels, num_heads, head_dim, cross_attention_dim, num_layers, layer_name,
                      block_ctor, block_kwargs, groups, dtype, device, operations):
    return Transformer2DModel(in_channels, num_heads, head_dim, cross_attention_dim, num_layers=num_layers,
                              groups=groups, block_ctor=block_ctor, layer_name=layer_name, block_kwargs=block_kwargs,
                              dtype=dtype, device=device, operations=operations)


class CrossAttnDownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers, num_heads, head_dim,
                 cross_attention_dim, transformer_layers, add_downsample, name, block_ctor, block_kwargs,
                 groups=32, dtype=None, device=None, operations=None):
        super().__init__()
        resnets, attentions = [], []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_ch, out_channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations))
            attentions.append(_make_transformer(out_channels, num_heads, head_dim, cross_attention_dim,
                                                 transformer_layers, f"{name}_{i}", block_ctor, block_kwargs,
                                                 groups, dtype, device, operations))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, dtype=dtype, device=device, operations=operations)]) if add_downsample else None

    def forward(self, hidden_states, temb, encoder_hidden_states, **kw):
        res = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states, **kw)
            res += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            res += (hidden_states,)
        return hidden_states, res


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, num_layers, add_downsample,
                 groups=32, dtype=None, device=None, operations=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            resnets.append(ResnetBlock2D(in_ch, out_channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList([Downsample2D(out_channels, dtype=dtype, device=device, operations=operations)]) if add_downsample else None

    def forward(self, hidden_states, temb, encoder_hidden_states=None, **kw):
        res = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            res += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            res += (hidden_states,)
        return hidden_states, res


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(self, channels, temb_channels, num_heads, head_dim, cross_attention_dim, transformer_layers,
                 name, block_ctor, block_kwargs, groups=32, dtype=None, device=None, operations=None):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations),
            ResnetBlock2D(channels, channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations),
        ])
        self.attentions = nn.ModuleList([
            _make_transformer(channels, num_heads, head_dim, cross_attention_dim, transformer_layers,
                              f"{name}_0", block_ctor, block_kwargs, groups, dtype, device, operations)
        ])

    def forward(self, hidden_states, temb, encoder_hidden_states, **kw):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states, **kw)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, num_layers, num_heads,
                 head_dim, cross_attention_dim, transformer_layers, add_upsample, name, block_ctor, block_kwargs,
                 groups=32, dtype=None, device=None, operations=None):
        super().__init__()
        resnets, attentions = [], []
        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            resnet_in = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in + res_skip, out_channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations))
            attentions.append(_make_transformer(out_channels, num_heads, head_dim, cross_attention_dim,
                                                 transformer_layers, f"{name}_{i}", block_ctor, block_kwargs,
                                                 groups, dtype, device, operations))
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, dtype=dtype, device=device, operations=operations)]) if add_upsample else None

    def forward(self, hidden_states, res_samples, temb, encoder_hidden_states, **kw):
        for resnet, attn in zip(self.resnets, self.attentions):
            res = res_samples[-1]
            res_samples = res_samples[:-1]
            hidden_states = torch.cat([hidden_states, res], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states, **kw)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, temb_channels, num_layers, add_upsample,
                 groups=32, dtype=None, device=None, operations=None):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip = in_channels if (i == num_layers - 1) else out_channels
            resnet_in = prev_output_channel if i == 0 else out_channels
            resnets.append(ResnetBlock2D(resnet_in + res_skip, out_channels, temb_channels, groups=groups, dtype=dtype, device=device, operations=operations))
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = nn.ModuleList([Upsample2D(out_channels, dtype=dtype, device=device, operations=operations)]) if add_upsample else None

    def forward(self, hidden_states, res_samples, temb, encoder_hidden_states=None, **kw):
        for resnet in self.resnets:
            res = res_samples[-1]
            res_samples = res_samples[:-1]
            hidden_states = torch.cat([hidden_states, res], dim=1)
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# UNet2DConditionModel (SD2 backbone)
# ---------------------------------------------------------------------------
class UNet2DConditionModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, block_out_channels=(320, 640, 1280, 1280),
                 layers_per_block=2, cross_attention_dim=1024, num_attention_heads=(5, 10, 20, 20),
                 transformer_layers_per_block=1, norm_num_groups=32, name_prefix="",
                 block_ctor=None, block_kwargs=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channels = in_channels
        self.block_out_channels = list(block_out_channels)
        self.cross_attention_dim = cross_attention_dim
        num_attention_heads = list(num_attention_heads)
        head_dims = [block_out_channels[i] // num_attention_heads[i] for i in range(len(block_out_channels))]
        block_ctor = block_ctor if block_ctor is not None else _plain_block_ctor
        block_kwargs = block_kwargs or {}

        time_embed_dim = block_out_channels[0] * 4
        self.conv_in = operations.Conv2d(in_channels, block_out_channels[0], 3, padding=1, dtype=dtype, device=device)
        self.time_proj = Timesteps(block_out_channels[0])
        self.time_embedding = TimestepEmbedding(block_out_channels[0], time_embed_dim, dtype=dtype, device=device, operations=operations)

        # down blocks: [CrossAttn]*(n-1) + Down
        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        n_blocks = len(block_out_channels)
        for i in range(n_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == n_blocks - 1
            if not is_final:
                self.down_blocks.append(CrossAttnDownBlock2D(
                    input_channel, output_channel, time_embed_dim, layers_per_block, num_attention_heads[i],
                    head_dims[i], cross_attention_dim, transformer_layers_per_block, add_downsample=True,
                    name=f"down_{i}", block_ctor=block_ctor, block_kwargs=block_kwargs, groups=norm_num_groups,
                    dtype=dtype, device=device, operations=operations))
            else:
                self.down_blocks.append(DownBlock2D(
                    input_channel, output_channel, time_embed_dim, layers_per_block, add_downsample=False,
                    groups=norm_num_groups, dtype=dtype, device=device, operations=operations))

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out_channels[-1], time_embed_dim, num_attention_heads[-1], head_dims[-1], cross_attention_dim,
            transformer_layers_per_block, name="mid", block_ctor=block_ctor, block_kwargs=block_kwargs,
            groups=norm_num_groups, dtype=dtype, device=device, operations=operations)

        # up blocks: Up + [CrossAttn]*(n-1)
        self.up_blocks = nn.ModuleList()
        reversed_out = list(reversed(block_out_channels))
        reversed_heads = list(reversed(num_attention_heads))
        reversed_hdim = list(reversed(head_dims))
        output_channel = reversed_out[0]
        for i in range(n_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_out[i]
            input_channel = reversed_out[min(i + 1, n_blocks - 1)]
            is_final = i == n_blocks - 1
            if i == 0:
                self.up_blocks.append(UpBlock2D(
                    input_channel, output_channel, prev_output_channel, time_embed_dim, layers_per_block + 1,
                    add_upsample=True, groups=norm_num_groups, dtype=dtype, device=device, operations=operations))
            else:
                self.up_blocks.append(CrossAttnUpBlock2D(
                    input_channel, output_channel, prev_output_channel, time_embed_dim, layers_per_block + 1,
                    reversed_heads[i], reversed_hdim[i], cross_attention_dim, transformer_layers_per_block,
                    add_upsample=not is_final, name=f"up_{i}", block_ctor=block_ctor, block_kwargs=block_kwargs,
                    groups=norm_num_groups, dtype=dtype, device=device, operations=operations))

        self.conv_norm_out = operations.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-5, affine=True, dtype=dtype, device=device)
        self.conv_out = operations.Conv2d(block_out_channels[0], out_channels, 3, padding=1, dtype=dtype, device=device)

    def forward(self, sample, timestep, encoder_hidden_states, **kw):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])
        t_emb = self.time_proj(timestep).to(sample.dtype)
        emb = self.time_embedding(t_emb)

        hidden_states = self.conv_in(sample)
        down_res = (hidden_states,)
        for block in self.down_blocks:
            hidden_states, res = block(hidden_states, emb, encoder_hidden_states, **kw)
            down_res += res
        hidden_states = self.mid_block(hidden_states, emb, encoder_hidden_states, **kw)
        for i, block in enumerate(self.up_blocks):
            n_res = len(block.resnets)
            res_samples = down_res[-n_res:]
            down_res = down_res[:-n_res]
            hidden_states = block(hidden_states, res_samples, emb, encoder_hidden_states, **kw)
        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


def _plain_block_ctor(inner_dim, num_heads, head_dim, cross_attention_dim, layer_name=None,
                      dtype=None, device=None, operations=None):
    """Dual-stream transformer block: wrapped BasicTransformerBlock with all
    specialized attention disabled (matches init_attention() defaults)."""
    return Basic2p5DTransformerBlock(inner_dim, num_heads, head_dim, cross_attention_dim, layer_name,
                                     use_ma=False, use_ra=False, use_mda=False, use_dino=False, pbr_setting=None,
                                     dtype=dtype, device=device, operations=operations)


def _make_2p5d_block_ctor(pbr_setting):
    def ctor(inner_dim, num_heads, head_dim, cross_attention_dim, layer_name=None,
             dtype=None, device=None, operations=None):
        return Basic2p5DTransformerBlock(inner_dim, num_heads, head_dim, cross_attention_dim, layer_name,
                                         use_ma=True, use_ra=True, use_mda=True, use_dino=True, pbr_setting=pbr_setting,
                                         dtype=dtype, device=device, operations=operations)
    return ctor


# ---------------------------------------------------------------------------
# DINO projection
# ---------------------------------------------------------------------------
class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1536, clip_extra_context_tokens=4,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = operations.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(cross_attention_dim, dtype=dtype, device=device)

    def forward(self, image_embeds):
        embeds = image_embeds
        num_token = 1
        if embeds.dim() == 3:
            num_token = embeds.shape[1]
            embeds = rearrange(embeds, "b n c -> (b n) c")
        tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        tokens = self.norm(tokens)
        tokens = rearrange(tokens, "(b nt) n c -> b (nt n) c", nt=num_token)
        return tokens


# ---------------------------------------------------------------------------
# UNet2p5DConditionModel
# ---------------------------------------------------------------------------
class UNet2p5DConditionModel(nn.Module):
    """Multiview PBR UNet: dual-stream reference + material/multiview/DINO attention."""

    def __init__(self, in_channels=12, ref_in_channels=4, out_channels=4,
                 block_out_channels=(320, 640, 1280, 1280), layers_per_block=2, cross_attention_dim=1024,
                 num_attention_heads=(5, 10, 20, 20), transformer_layers_per_block=1, norm_num_groups=32,
                 pbr_setting=("albedo", "mr"), pbr_token_channels=77, dino_embeddings_dim=1536,
                 use_dino=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.pbr_setting = list(pbr_setting)
        self.use_ma = True
        self.use_ra = True
        self.use_mda = True
        self.use_dino = use_dino
        self.use_dual_stream = True
        self.use_learned_text_clip = True
        self.in_channels = in_channels
        self.out_channels = out_channels

        common = dict(block_out_channels=block_out_channels, layers_per_block=layers_per_block,
                      cross_attention_dim=cross_attention_dim, num_attention_heads=num_attention_heads,
                      transformer_layers_per_block=transformer_layers_per_block, norm_num_groups=norm_num_groups,
                      dtype=dtype, device=device, operations=operations)

        # main stream (12-channel input, full 2.5D attention)
        self.unet = UNet2DConditionModel(
            in_channels=in_channels, out_channels=out_channels,
            block_ctor=_make_2p5d_block_ctor(self.pbr_setting), block_kwargs={}, **common)

        # dual stream reference encoder (4-channel input, plain wrapped blocks)
        self.unet_dual = UNet2DConditionModel(
            in_channels=ref_in_channels, out_channels=out_channels,
            block_ctor=_plain_block_ctor, block_kwargs={}, **common)

        # learned material text-clip embeddings (registered on inner unet to match keys)
        for token in self.pbr_setting:
            self.unet.register_parameter(
                f"learned_text_clip_{token}",
                nn.Parameter(torch.zeros(pbr_token_channels, cross_attention_dim, dtype=dtype, device=device)))
        self.unet.learned_text_clip_ref = nn.Parameter(
            torch.zeros(pbr_token_channels, cross_attention_dim, dtype=dtype, device=device))

        if self.use_dino:
            self.unet.image_proj_model_dino = ImageProjModel(
                cross_attention_dim=cross_attention_dim, clip_embeddings_dim=dino_embeddings_dim,
                clip_extra_context_tokens=4, dtype=dtype, device=device, operations=operations)

    def forward(self, sample, timestep, encoder_hidden_states, dino_hidden_states=None,
                ref_latents=None, embeds_normal=None, embeds_position=None, position_maps=None,
                mva_scale=1.0, ref_scale=1.0, cache=None):
        """sample: (B, N_pbr, N_gen, C, H, W). encoder_hidden_states: (B, N_pbr, L, cross_dim)."""
        B, N_pbr, N_gen, _, H, W = sample.shape
        if cache is None:
            cache = {}

        # concat control embeds along channel dim -> 12 channels
        parts = [sample]
        if embeds_normal is not None:
            parts.append(embeds_normal.unsqueeze(1).repeat(1, N_pbr, 1, 1, 1, 1))
        if embeds_position is not None:
            parts.append(embeds_position.unsqueeze(1).repeat(1, N_pbr, 1, 1, 1, 1))
        sample = torch.cat(parts, dim=-3)
        sample = rearrange(sample, "b n_pbr n c h w -> (b n_pbr n) c h w")

        encoder_hidden_states_gen = encoder_hidden_states.unsqueeze(-3).repeat(1, 1, N_gen, 1, 1)
        encoder_hidden_states_gen = rearrange(encoder_hidden_states_gen, "b n_pbr n l c -> (b n_pbr n) l c")

        # position rope voxel indices
        position_voxel_indices = None
        if position_maps is not None:
            if "position_voxel_indices" in cache:
                position_voxel_indices = cache["position_voxel_indices"]
            else:
                position_voxel_indices = calc_multires_voxel_idxs(
                    position_maps, grid_resolutions=[H, H // 2, H // 4, H // 8],
                    voxel_resolutions=[H * 8, H * 4, H * 2, H])
                cache["position_voxel_indices"] = position_voxel_indices

        # dino projection
        dino_proj = None
        if self.use_dino and dino_hidden_states is not None:
            if "dino_proj" in cache:
                dino_proj = cache["dino_proj"]
            else:
                dino_proj = self.unet.image_proj_model_dino(dino_hidden_states)
                cache["dino_proj"] = dino_proj

        # reference dual-stream write pass
        condition_embed_dict = None
        if self.use_ra and ref_latents is not None:
            if "condition_embed_dict" in cache:
                condition_embed_dict = cache["condition_embed_dict"]
            else:
                condition_embed_dict = {}
                N_ref = ref_latents.shape[1]
                ref = rearrange(ref_latents, "b n c h w -> (b n) c h w")
                enc_ref = self.unet.learned_text_clip_ref.to(ref.dtype)[None, None].repeat(B, N_ref, 1, 1)
                enc_ref = rearrange(enc_ref, "b n l c -> (b n) l c")
                self.unet_dual(ref, 0, enc_ref, num_in_batch=N_ref, mode="w",
                               condition_embed_dict=condition_embed_dict)
                cache["condition_embed_dict"] = condition_embed_dict

        out = self.unet(sample, timestep, encoder_hidden_states_gen, num_in_batch=N_gen, mode="r",
                        mva_scale=mva_scale, ref_scale=ref_scale, condition_embed_dict=condition_embed_dict,
                        dino_hidden_states=dino_proj, position_voxel_indices=position_voxel_indices)
        return out

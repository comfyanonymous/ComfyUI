# CogVideoX VAE - ported to ComfyUI native ops
# Architecture reference: diffusers AutoencoderKLCogVideoX
# Style reference: comfy/ldm/wan/vae.py

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
ops = comfy.ops.disable_weight_init


class SafeConv3d(nn.Conv3d):
    """3D convolution that splits large inputs along temporal dim to avoid OOM."""
    def forward(self, x):
        mem = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4] * 2 / 1024**3
        if mem > 2 and x.shape[2] >= self.kernel_size[0]:
            kernel_t = self.kernel_size[0]
            parts = int(mem / 2) + 1
            # Ensure each chunk has at least kernel_t frames
            max_parts = max(1, x.shape[2] // kernel_t)
            parts = min(parts, max_parts)
            if parts <= 1:
                return super().forward(x)
            chunks = torch.chunk(x, parts, dim=2)
            if kernel_t > 1:
                chunks = [chunks[0]] + [
                    torch.cat((chunks[i - 1][:, :, -kernel_t + 1:], chunks[i]), dim=2)
                    for i in range(1, len(chunks))
                ]
            out = []
            for chunk in chunks:
                out.append(super().forward(chunk))
            return torch.cat(out, dim=2)
        return super().forward(x)


class CausalConv3d(nn.Module):
    """Causal 3D convolution with temporal padding."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode="constant"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel, height_kernel, width_kernel = kernel_size
        time_pad = time_kernel - 1
        height_pad = (height_kernel - 1) // 2
        width_pad = (width_kernel - 1) // 2

        self.pad_mode = pad_mode
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.const_padding = (0, width_pad, height_pad)
        self.time_kernel_size = time_kernel

        stride = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = SafeConv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation,
            padding=0 if pad_mode == "replicate" else self.const_padding,
        )

    def forward(self, x, conv_cache=None):
        if self.pad_mode == "replicate":
            x = F.pad(x, self.time_causal_padding, mode="replicate")
            conv_cache = None
        else:
            kernel_t = self.time_kernel_size
            if kernel_t > 1:
                cached = [conv_cache] if conv_cache is not None else [x[:, :, :1]] * (kernel_t - 1)
                x = torch.cat(cached + [x], dim=2)
            conv_cache = x[:, :, -self.time_kernel_size + 1:].clone() if self.time_kernel_size > 1 else None

        out = self.conv(x)
        return out, conv_cache


class SpatialNorm3D(nn.Module):
    """Spatially conditioned normalization."""
    def __init__(self, f_channels, zq_channels, groups=32):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def forward(self, f, zq, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first.shape[-3:])
            z_rest = F.interpolate(z_rest, size=f_rest.shape[-3:])
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        conv_y, new_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        return self.norm_layer(f) * conv_y + conv_b, new_cache


class ResnetBlock3D(nn.Module):
    """3D ResNet block with optional spatial norm."""
    def __init__(self, in_channels, out_channels=None, temb_channels=512, groups=32,
                 eps=1e-6, act_fn="silu", spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_norm_dim = spatial_norm_dim

        if act_fn == "silu":
            self.nonlinearity = nn.SiLU()
        elif act_fn == "swish":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = nn.SiLU()

        if spatial_norm_dim is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            self.norm1 = SpatialNorm3D(in_channels, spatial_norm_dim, groups=groups)
            self.norm2 = SpatialNorm3D(out_channels, spatial_norm_dim, groups=groups)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

        if in_channels != out_channels:
            self.conv_shortcut = SafeConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        residual = x

        if zq is not None:
            x, new_cache["norm1"] = self.norm1(x, zq, conv_cache=conv_cache.get("norm1"))
        else:
            x = self.norm1(x)

        x = self.nonlinearity(x)
        x, new_cache["conv1"] = self.conv1(x, conv_cache=conv_cache.get("conv1"))

        if temb is not None and hasattr(self, "temb_proj"):
            x = x + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            x, new_cache["norm2"] = self.norm2(x, zq, conv_cache=conv_cache.get("norm2"))
        else:
            x = self.norm2(x)

        x = self.nonlinearity(x)
        x, new_cache["conv2"] = self.conv2(x, conv_cache=conv_cache.get("conv2"))

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return x + residual, new_cache


class Downsample3D(nn.Module):
    """3D downsampling with optional temporal compression."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, compress_time=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            b, c, t, h, w = x.shape
            x = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)
            if t % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = x.reshape(b, h, w, c, x.shape[-1]).permute(0, 3, 4, 1, 2)

        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


class Upsample3D(nn.Module):
    """3D upsampling with optional temporal decompression."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, compress_time=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            if x.shape[2] > 1 and x.shape[2] % 2 == 1:
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]
                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            elif x.shape[2] > 1:
                x = F.interpolate(x, scale_factor=2.0)
            else:
                x = x.squeeze(2)
                x = F.interpolate(x, scale_factor=2.0)
                x = x[:, :, None, :, :]
        else:
            b, c, t, h, w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(x, scale_factor=2.0)
            x = x.reshape(b, t, c, *x.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.conv(x)
        x = x.reshape(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        return x


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, add_downsample=True,
                 compress_time=False, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=groups, eps=eps, act_fn=act_fn, pad_mode=pad_mode,
            )
            for i in range(num_layers)
        ])
        self.downsamplers = nn.ModuleList([Downsample3D(out_channels, out_channels, compress_time=compress_time)]) if add_downsample else None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        if self.downsamplers is not None:
            for ds in self.downsamplers:
                x = ds(x)
        return x, new_cache


class MidBlock3D(nn.Module):
    def __init__(self, in_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels, groups=groups, eps=eps,
                act_fn=act_fn, spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        return x, new_cache


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels=0, num_layers=1,
                 eps=1e-6, act_fn="silu", groups=32, spatial_norm_dim=16,
                 add_upsample=True, compress_time=False, pad_mode="first"):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels, groups=groups, eps=eps,
                act_fn=act_fn, spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode,
            )
            for i in range(num_layers)
        ])
        self.upsamplers = nn.ModuleList([Upsample3D(out_channels, out_channels, compress_time=compress_time)]) if add_upsample else None

    def forward(self, x, temb=None, zq=None, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}
        for i, resnet in enumerate(self.resnets):
            x, new_cache[f"resnet_{i}"] = resnet(x, temb, zq, conv_cache=conv_cache.get(f"resnet_{i}"))
        if self.upsamplers is not None:
            for us in self.upsamplers:
                x = us(x)
        return x, new_cache


class Encoder3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=16,
                 block_out_channels=(128, 256, 256, 512),
                 layers_per_block=3, act_fn="silu",
                 eps=1e-6, groups=32, pad_mode="first",
                 temporal_compression_ratio=4):
        super().__init__()
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.down_blocks = nn.ModuleList()
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            self.down_blocks.append(DownBlock3D(
                in_channels=input_channel, out_channels=output_channel,
                temb_channels=0, num_layers=layers_per_block,
                eps=eps, act_fn=act_fn, groups=groups,
                add_downsample=not is_final, compress_time=compress_time,
            ))

        self.mid_block = MidBlock3D(
            in_channels=block_out_channels[-1], temb_channels=0,
            num_layers=2, eps=eps, act_fn=act_fn, groups=groups, pad_mode=pad_mode,
        )

        self.norm_out = nn.GroupNorm(groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, x, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        x, new_cache["conv_in"] = self.conv_in(x, conv_cache=conv_cache.get("conv_in"))

        for i, block in enumerate(self.down_blocks):
            key = f"down_block_{i}"
            x, new_cache[key] = block(x, None, None, conv_cache.get(key))

        x, new_cache["mid_block"] = self.mid_block(x, None, None, conv_cache=conv_cache.get("mid_block"))

        x = self.norm_out(x)
        x = self.conv_act(x)
        x, new_cache["conv_out"] = self.conv_out(x, conv_cache=conv_cache.get("conv_out"))

        return x, new_cache


class Decoder3D(nn.Module):
    def __init__(self, in_channels=16, out_channels=3,
                 block_out_channels=(128, 256, 256, 512),
                 layers_per_block=3, act_fn="silu",
                 eps=1e-6, groups=32, pad_mode="first",
                 temporal_compression_ratio=4):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CausalConv3d(in_channels, reversed_channels[0], kernel_size=3, pad_mode=pad_mode)

        self.mid_block = MidBlock3D(
            in_channels=reversed_channels[0], temb_channels=0,
            num_layers=2, eps=eps, act_fn=act_fn, groups=groups,
            spatial_norm_dim=in_channels, pad_mode=pad_mode,
        )

        self.up_blocks = nn.ModuleList()
        output_channel = reversed_channels[0]
        for i in range(len(block_out_channels)):
            prev_channel = output_channel
            output_channel = reversed_channels[i]
            is_final = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            self.up_blocks.append(UpBlock3D(
                in_channels=prev_channel, out_channels=output_channel,
                temb_channels=0, num_layers=layers_per_block + 1,
                eps=eps, act_fn=act_fn, groups=groups,
                spatial_norm_dim=in_channels,
                add_upsample=not is_final, compress_time=compress_time,
            ))

        self.norm_out = SpatialNorm3D(reversed_channels[-1], in_channels, groups=groups)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(reversed_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode)

    def forward(self, sample, conv_cache=None):
        new_cache = {}
        conv_cache = conv_cache or {}

        x, new_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        x, new_cache["mid_block"] = self.mid_block(x, None, sample, conv_cache=conv_cache.get("mid_block"))

        for i, block in enumerate(self.up_blocks):
            key = f"up_block_{i}"
            x, new_cache[key] = block(x, None, sample, conv_cache=conv_cache.get(key))

        x, new_cache["norm_out"] = self.norm_out(x, sample, conv_cache=conv_cache.get("norm_out"))
        x = self.conv_act(x)
        x, new_cache["conv_out"] = self.conv_out(x, conv_cache=conv_cache.get("conv_out"))

        return x, new_cache



class AutoencoderKLCogVideoX(nn.Module):
    """CogVideoX VAE. Spatial tiling/slicing handled by ComfyUI's VAE wrapper.

    Temporal frame batching with conv_cache is kept here since the causal 3D
    convolutions need state passed between temporal chunks.
    """

    def __init__(self,
                 in_channels=3, out_channels=3,
                 block_out_channels=(128, 256, 256, 512),
                 latent_channels=16, layers_per_block=3,
                 act_fn="silu", eps=1e-6, groups=32,
                 temporal_compression_ratio=4,
                 ):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = Encoder3D(
            in_channels=in_channels, out_channels=latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            act_fn=act_fn, eps=eps, groups=groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = Decoder3D(
            in_channels=latent_channels, out_channels=out_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            act_fn=act_fn, eps=eps, groups=groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )

        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

    def encode(self, x):
        t = x.shape[2]
        frame_batch = self.num_sample_frames_batch_size
        num_batches = max(t // frame_batch, 1)
        conv_cache = None
        enc = []
        for i in range(num_batches):
            remaining = t % frame_batch
            start = frame_batch * i + (0 if i == 0 else remaining)
            end = frame_batch * (i + 1) + remaining
            chunk, conv_cache = self.encoder(x[:, :, start:end], conv_cache=conv_cache)
            enc.append(chunk.to(x.device))
        enc = torch.cat(enc, dim=2)
        mean, _ = enc.chunk(2, dim=1)
        return mean

    def decode(self, z):
        t = z.shape[2]
        frame_batch = self.num_latent_frames_batch_size
        num_batches = max(t // frame_batch, 1)
        conv_cache = None
        dec = []
        for i in range(num_batches):
            remaining = t % frame_batch
            start = frame_batch * i + (0 if i == 0 else remaining)
            end = frame_batch * (i + 1) + remaining
            chunk, conv_cache = self.decoder(z[:, :, start:end], conv_cache=conv_cache)
            dec.append(chunk.cpu())
        return torch.cat(dec, dim=2).to(z.device)

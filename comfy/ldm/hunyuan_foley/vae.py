import math
import torch
import numpy as np
from typing import List
import torch.nn as nn
from einops import rearrange
from torchvision.transforms import v2
from torch.nn.utils.parametrizations import weight_norm

from comfy.ldm.hunyuan_foley.syncformer import Synchformer

import comfy.ops
ops = comfy.ops.disable_weight_init

# until the higgsv2 pr gets accepted
def WNConv1d(*args, device = None, dtype = None, operations = None, **kwargs):
    return weight_norm(operations.Conv1d(*args, **kwargs, device = device, dtype = dtype))


def WNConvTranspose1d(*args, device = None, dtype = None, operations = None, **kwargs):
    return weight_norm(operations.ConvTranspose1d(*args, **kwargs, device = device, dtype = dtype))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels, device = None, dtype = None):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, device = device, dtype = dtype))

    def forward(self, x):
        return snake(x, self.alpha)

class DACResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim, device = device, dtype = dtype),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad, device = device, dtype = dtype, operations = operations),
            Snake1d(dim, device = device, dtype = dtype),
            WNConv1d(dim, dim, kernel_size=1, device = device, dtype = dtype, operations = operations),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class DACEncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        self.block = nn.Sequential(
            DACResidualUnit(dim // 2, dilation=1, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(dim // 2, dilation=3, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(dim // 2, dilation=9, device = device, dtype = dtype, operations = operations),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                device = device, dtype = dtype, operations = operations
            ),
        )

    def forward(self, x):
        return self.block(x)


class DACEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 256,
        device = None, dtype = None, operations = None
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3, device = device, dtype = dtype, operations = operations)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [DACEncoderBlock(d_model, stride=stride, device = device, dtype = dtype, operations = operations)]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DACDecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, device = None, dtype = None, operations = None):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim, device = device, dtype = dtype),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
                device = device, dtype = dtype, operations = operations
            ),
            DACResidualUnit(output_dim, dilation=1, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(output_dim, dilation=3, device = device, dtype = dtype, operations = operations),
            DACResidualUnit(output_dim, dilation=9, device = device, dtype = dtype, operations = operations),
        )

    def forward(self, x):
        return self.block(x)


class DACDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        device = None, dtype = None, operations = None
    ):
        super().__init__()

        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3, device = device, dtype = dtype, operations = operations )]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DACDecoderBlock(input_dim, output_dim, stride, device = device, dtype = dtype, operations = operations)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DAC(torch.nn.Module):
    def __init__(
        self,
        encoder_dim: int = 128,
        encoder_rates: List[int] = [2, 3, 4, 5],
        latent_dim: int = 128,
        decoder_dim: int = 2048,
        decoder_rates: List[int] = [8, 5, 4, 3],
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = DACEncoder(encoder_dim, encoder_rates, latent_dim, operations = ops)

        self.decoder = DACDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            operations = ops
        )
        self.sample_rate = sample_rate


    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self):
        pass

class FoleyVae(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dac = DAC()
        self.syncformer = Synchformer(None, None, operations = ops)
        self.syncformer_preprocess = v2.Compose(
            [
                v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    def decode(self, x, vae_options = {}):
        return self.dac.decode(x)
    def encode(self, x):
        return self.syncformer(x)
    
    def video_encoding(self, video, step: int):
        t, h, w, c = video.shape

        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video)

        video = video.permute(0, 3, 1, 2)

        video = torch.stack([self.syncformer_preprocess(t) for t in video]).unsqueeze(0)
        seg_len = 16
        t = video.size(0)
        nseg = max(0, (t - seg_len) // step + 1)
        stride_t, stride_c, stride_h, stride_w = video.stride()

        # no copies 
        data = video.as_strided(
            size=(nseg, seg_len, c, h, w),
            stride=(stride_t * step, stride_t, stride_c, stride_h, stride_w),
        )
        data = rearrange(data, "b s t c h w -> (b s) 1 t c h w")

        return data, nseg, lambda x: rearrange(x, "(b s) 1 t d -> b (s t) d", b=video.size(0))

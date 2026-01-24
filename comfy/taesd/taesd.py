#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn

import comfy.utils
import comfy.ops

def conv(n_in, n_out, **kwargs):
    return comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out, use_midblock_gn=False):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = lambda n_in, n_out: comfy.ops.disable_weight_init.Conv2d(n_in, n_out, 1, bias=False), n_in*4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), comfy.ops.disable_weight_init.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))
    def forward(self, x):
        if self.pool is not None:
            x = x + self.pool(x)
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw),
        conv(64, latent_channels),
    )


def Decoder(latent_channels=4, use_midblock_gn=False):
    mb_kw = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), Block(64, 64, **mb_kw), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path=None, decoder_path=None, latent_channels=4, use_midblock_gn=False):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()

        self.latent_channels = latent_channels
        self.use_midblock_gn = use_midblock_gn
        self.taesd_encoder = Encoder(latent_channels=latent_channels, use_midblock_gn=use_midblock_gn)
        self.taesd_decoder = Decoder(latent_channels=latent_channels, use_midblock_gn=use_midblock_gn)

        if encoder_path is not None:
            self.taesd_encoder, self.latent_channels = self._load_model(encoder_path, Encoder)
        if decoder_path is not None:
            self.taesd_decoder, self.latent_channels = self._load_model(decoder_path, Decoder)

        self.vae_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.vae_shift = torch.nn.Parameter(torch.tensor(0.0))

    def _load_model(self, path, model_class):
        """Load a TAESD encoder or decoder from a file."""
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        latent_channels = sd["1.weight"].shape[1]
        model = model_class(latent_channels=latent_channels, use_midblock_gn="3.pool.0.weight" in sd)
        model.load_state_dict(sd)
        return model, latent_channels

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

    def decode(self, x):
        if x.shape[1] == self.latent_channels * 4:
            x = x.reshape(x.shape[0], self.latent_channels, 2, 2, x.shape[-2], x.shape[-1]).permute(0, 1, 4, 2, 5, 3).reshape(x.shape[0], self.latent_channels, x.shape[-2] * 2, x.shape[-1] * 2)

        x_sample = self.taesd_decoder((x - self.vae_shift) * self.vae_scale)
        x_sample = x_sample.sub(0.5).mul(2)
        return x_sample

    def encode(self, x):
        x_sample = (self.taesd_encoder(x * 0.5 + 0.5) / self.vae_scale) + self.vae_shift
        if self.latent_channels == 32 and self.use_midblock_gn: # Only taef2 for Flux2 currently, pack latents: [B, C, H, W] -> [B, C*4, H//2, W//2]
            x_sample = x_sample.reshape(x_sample.shape[0], self.latent_channels, x_sample.shape[-2] // 2, 2, x_sample.shape[-1] // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(x_sample.shape[0], self.latent_channels * 4, x_sample.shape[-2] // 2, x_sample.shape[-1] // 2)
        return x_sample

#!/usr/bin/env python3
"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)
"""
import torch
import torch.nn as nn

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )

def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class TAESD(nn.Module):
    latent_magnitude = 3
    latent_shift = 0.5

    def __init__(self, encoder_path="taesd_encoder.pth", decoder_path="taesd_decoder.pth"):
        """Initialize pretrained TAESD on the given device from the given checkpoints."""
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        if encoder_path is not None:
            if encoder_path.lower().endswith(".safetensors"):
                import safetensors.torch
                self.encoder.load_state_dict(safetensors.torch.load_file(encoder_path, device="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoder_path, map_location="cpu", weights_only=True))
        if decoder_path is not None:
            if decoder_path.lower().endswith(".safetensors"):
                import safetensors.torch
                self.decoder.load_state_dict(safetensors.torch.load_file(decoder_path, device="cpu"))
            else:
                self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))

    @staticmethod
    def scale_latents(x):
        """raw latents -> [0, 1]"""
        return x.div(2 * TAESD.latent_magnitude).add(TAESD.latent_shift).clamp(0, 1)

    @staticmethod
    def unscale_latents(x):
        """[0, 1] -> raw latents"""
        return x.sub(TAESD.latent_shift).mul(2 * TAESD.latent_magnitude)

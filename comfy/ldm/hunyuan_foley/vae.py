import torch
import numpy as np
from typing import List
from einops import rearrange
from torchvision.transforms import v2

from comfy.ldm.hunyuan_foley.syncformer import Synchformer
from comfy.ldm.higgsv2.tokenizer import DACEncoder, DACDecoder

import comfy.ops
ops = comfy.ops.disable_weight_init

class DAC(torch.nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
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

        if not isinstance(video, torch.Tensor):
            video = torch.from_numpy(video).permute(0, 3, 1, 2)

        video = self.syncformer_preprocess(video).unsqueeze(0)
        seg_len = 16
        t = video.size(1)
        nseg = max(0, (t - seg_len) // step + 1)
        clips = [video[:, i*step:i*step + seg_len] for i in range(nseg)]
        data = torch.stack(clips, dim=1)
        data = rearrange(data, "b s t c h w -> (b s) 1 t c h w")

        return data, nseg, lambda x: rearrange(x, "(b s) 1 t d -> b (s t) d", b=video.size(0))

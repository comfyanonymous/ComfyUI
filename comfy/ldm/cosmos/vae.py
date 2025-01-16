# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The causal continuous video tokenizer with VAE or AE formulation for 3D data.."""

import logging
import torch
from torch import nn
from enum import Enum
import math

from .cosmos_tokenizer.layers3d import (
    EncoderFactorized,
    DecoderFactorized,
    CausalConv3d,
)


class IdentityDistribution(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parameters):
        return parameters, (torch.tensor([0.0]), torch.tensor([0.0]))


class GaussianDistribution(torch.nn.Module):
    def __init__(self, min_logvar: float = -30.0, max_logvar: float = 20.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def sample(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(mean)

    def forward(self, parameters):
        mean, logvar = torch.chunk(parameters, 2, dim=1)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return self.sample(mean, logvar), (mean, logvar)


class ContinuousFormulation(Enum):
    VAE = GaussianDistribution
    AE = IdentityDistribution


class CausalContinuousVideoTokenizer(nn.Module):
    def __init__(
        self, z_channels: int, z_factor: int, latent_channels: int, **kwargs
    ) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalContinuousVideoTokenizer")
        self.latent_channels = latent_channels
        self.sigma_data = 0.5

        # encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = EncoderFactorized(
            z_channels=z_factor * z_channels, **kwargs
        )
        if kwargs.get("temporal_compression", 4) == 4:
            kwargs["channels_mult"] = [2, 4]
        # decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = DecoderFactorized(
            z_channels=z_channels, **kwargs
        )

        self.quant_conv = CausalConv3d(
            z_factor * z_channels,
            z_factor * latent_channels,
            kernel_size=1,
            padding=0,
        )
        self.post_quant_conv = CausalConv3d(
            latent_channels, z_channels, kernel_size=1, padding=0
        )

        # formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
        self.distribution = IdentityDistribution()  # ContinuousFormulation[formulation_name].value()

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.debug(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.debug(
            f"z_channels={z_channels}, latent_channels={self.latent_channels}."
        )

        latent_temporal_chunk = 16
        self.latent_mean = nn.Parameter(torch.zeros([self.latent_channels * latent_temporal_chunk], dtype=torch.float32))
        self.latent_std = nn.Parameter(torch.ones([self.latent_channels * latent_temporal_chunk], dtype=torch.float32))


    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        z, posteriors = self.distribution(moments)
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        in_dtype = z.dtype
        mean = self.latent_mean.view(latent_ch, -1)
        std = self.latent_std.view(latent_ch, -1)

        mean = mean.repeat(1, math.ceil(latent_t / mean.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        std = std.repeat(1, math.ceil(latent_t / std.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        return ((z - mean) / std) * self.sigma_data

    def decode(self, z):
        in_dtype = z.dtype
        latent_ch = z.shape[1]
        latent_t = z.shape[2]
        mean = self.latent_mean.view(latent_ch, -1)
        std = self.latent_std.view(latent_ch, -1)

        mean = mean.repeat(1, math.ceil(latent_t / mean.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)
        std = std.repeat(1, math.ceil(latent_t / std.shape[-1]))[:, : latent_t].reshape([1, latent_ch, -1, 1, 1]).to(dtype=in_dtype, device=z.device)

        z = z / self.sigma_data
        z = z * std + mean
        z = self.post_quant_conv(z)
        return self.decoder(z)


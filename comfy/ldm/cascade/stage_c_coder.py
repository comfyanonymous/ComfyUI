"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import torch
import torchvision
from torch import nn

import comfy.ops

ops = comfy.ops.disable_weight_init

# EfficientNet
class EfficientNetEncoder(nn.Module):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
        self.mapper = nn.Sequential(
            ops.Conv2d(1280, c_latent, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_latent, affine=False),  # then normalize them to have mean 0 and std 1
        )
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean.view([3,1,1]).to(device=x.device, dtype=x.dtype)) / self.std.view([3,1,1]).to(device=x.device, dtype=x.dtype)
        o = self.mapper(self.backbone(x))
        return o


# Fast Decoder for Stage C latents. E.g. 16 x 24 x 24 -> 3 x 192 x 192
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            ops.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            ops.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            ops.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            ops.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            ops.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            ops.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            ops.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            ops.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            ops.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return (self.blocks(x) - 0.5) * 2.0

class StageC_coder(nn.Module):
    def __init__(self):
        super().__init__()
        self.previewer = Previewer()
        self.encoder = EfficientNetEncoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.previewer(x)

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
from .common import LayerNorm2d_op


class CNetResBlock(nn.Module):
    def __init__(self, c, dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.Sequential(
            LayerNorm2d_op(operations)(c, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c, c, kernel_size=3, padding=1),
            LayerNorm2d_op(operations)(c, dtype=dtype, device=device),
            nn.GELU(),
            operations.Conv2d(c, c, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.blocks(x)


class ControlNet(nn.Module):
    def __init__(self, c_in=3, c_proj=2048, proj_blocks=None, bottleneck_mode=None, dtype=None, device=None, operations=nn):
        super().__init__()
        if bottleneck_mode is None:
            bottleneck_mode = 'effnet'
        self.proj_blocks = proj_blocks
        if bottleneck_mode == 'effnet':
            embd_channels = 1280
            self.backbone = torchvision.models.efficientnet_v2_s().features.eval()
            if c_in != 3:
                in_weights = self.backbone[0][0].weight.data
                self.backbone[0][0] = operations.Conv2d(c_in, 24, kernel_size=3, stride=2, bias=False, dtype=dtype, device=device)
                if c_in > 3:
                    # nn.init.constant_(self.backbone[0][0].weight, 0)
                    self.backbone[0][0].weight.data[:, :3] = in_weights[:, :3].clone()
                else:
                    self.backbone[0][0].weight.data = in_weights[:, :c_in].clone()
        elif bottleneck_mode == 'simple':
            embd_channels = c_in
            self.backbone = nn.Sequential(
                operations.Conv2d(embd_channels, embd_channels * 4, kernel_size=3, padding=1, dtype=dtype, device=device),
                nn.LeakyReLU(0.2, inplace=True),
                operations.Conv2d(embd_channels * 4, embd_channels, kernel_size=3, padding=1, dtype=dtype, device=device),
            )
        elif bottleneck_mode == 'large':
            self.backbone = nn.Sequential(
                operations.Conv2d(c_in, 4096 * 4, kernel_size=1, dtype=dtype, device=device),
                nn.LeakyReLU(0.2, inplace=True),
                operations.Conv2d(4096 * 4, 1024, kernel_size=1, dtype=dtype, device=device),
                *[CNetResBlock(1024, dtype=dtype, device=device, operations=operations) for _ in range(8)],
                operations.Conv2d(1024, 1280, kernel_size=1, dtype=dtype, device=device),
            )
            embd_channels = 1280
        else:
            raise ValueError(f'Unknown bottleneck mode: {bottleneck_mode}')
        self.projections = nn.ModuleList()
        for _ in range(len(proj_blocks)):
            self.projections.append(nn.Sequential(
                operations.Conv2d(embd_channels, embd_channels, kernel_size=1, bias=False, dtype=dtype, device=device),
                nn.LeakyReLU(0.2, inplace=True),
                operations.Conv2d(embd_channels, c_proj, kernel_size=1, bias=False, dtype=dtype, device=device),
            ))
            # nn.init.constant_(self.projections[-1][-1].weight, 0)  # zero output projection
        self.xl = False
        self.input_channels = c_in
        self.unshuffle_amount = 8

    def forward(self, x):
        x = self.backbone(x)
        proj_outputs = [None for _ in range(max(self.proj_blocks) + 1)]
        for i, idx in enumerate(self.proj_blocks):
            proj_outputs[idx] = self.projections[i](x)
        return {"input": proj_outputs[::-1]}

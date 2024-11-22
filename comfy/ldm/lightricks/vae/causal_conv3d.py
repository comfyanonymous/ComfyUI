from typing import Tuple, Union

import torch
import torch.nn as nn


class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = (kernel_size, kernel_size, kernel_size)
        self.time_kernel_size = kernel_size[0]

        dilation = (dilation, 1, 1)

        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode="zeros",
            groups=groups,
        )

    def forward(self, x, causal: bool = True):
        if causal:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            last_frame_pad = x[:, :, -1:, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self):
        return self.conv.weight

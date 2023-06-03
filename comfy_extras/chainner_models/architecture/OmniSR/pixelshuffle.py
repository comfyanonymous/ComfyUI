#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: pixelshuffle.py
# Created Date: Friday July 1st 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 1st July 2022 10:18:39 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################

import torch.nn as nn


def pixelshuffle_block(
    in_channels, out_channels, upscale_factor=2, kernel_size=3, bias=False
):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(
        in_channels,
        out_channels * (upscale_factor**2),
        kernel_size,
        padding=1,
        bias=bias,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])

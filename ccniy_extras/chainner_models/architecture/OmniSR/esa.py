#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: esa.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 20th April 2023 9:28:06 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layernorm import LayerNorm2d


def moment(x, dim=(2, 3), k=2):
    assert len(x.size()) == 4
    mean = torch.mean(x, dim=dim).unsqueeze(-1).unsqueeze(-1)
    mk = (1 / (x.size(2) * x.size(3))) * torch.sum(torch.pow(x - mean, k), dim=dim)
    return mk


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class LK_ESA(nn.Module):
    def __init__(
        self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True
    ):
        super(LK_ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.vec_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=2,
            bias=bias,
        )
        self.vec_conv3x1 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=2,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=2,
            bias=bias,
        )
        self.hor_conv1x3 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=bias,
        )

        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)

        res = self.vec_conv(c1_) + self.vec_conv3x1(c1_)
        res = self.hor_conv(res) + self.hor_conv1x3(res)

        cf = self.conv_f(c1_)
        c4 = self.conv4(res + cf)
        m = self.sigmoid(c4)
        return x * m


class LK_ESA_LN(nn.Module):
    def __init__(
        self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True
    ):
        super(LK_ESA_LN, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.norm = LayerNorm2d(n_feats)

        self.vec_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=2,
            bias=bias,
        )
        self.vec_conv3x1 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=2,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=2,
            bias=bias,
        )
        self.hor_conv1x3 = nn.Conv2d(
            in_channels=f * kernel_expand,
            out_channels=f * kernel_expand,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=2,
            bias=bias,
        )

        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.norm(x)
        c1_ = self.conv1(c1_)

        res = self.vec_conv(c1_) + self.vec_conv3x1(c1_)
        res = self.hor_conv(res) + self.hor_conv1x3(res)

        cf = self.conv_f(c1_)
        c4 = self.conv4(res + cf)
        m = self.sigmoid(c4)
        return x * m


class AdaGuidedFilter(nn.Module):
    def __init__(
        self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True
    ):
        super(AdaGuidedFilter, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=n_feats,
            out_channels=1,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.r = 5

    def box_filter(self, x, r):
        channel = x.shape[1]
        kernel_size = 2 * r + 1
        weight = 1.0 / (kernel_size**2)
        box_kernel = weight * torch.ones(
            (channel, 1, kernel_size, kernel_size), dtype=torch.float32, device=x.device
        )
        output = F.conv2d(x, weight=box_kernel, stride=1, padding=r, groups=channel)
        return output

    def forward(self, x):
        _, _, H, W = x.shape
        N = self.box_filter(
            torch.ones((1, 1, H, W), dtype=x.dtype, device=x.device), self.r
        )

        # epsilon = self.fc(self.gap(x))
        # epsilon = torch.pow(epsilon, 2)
        epsilon = 1e-2

        mean_x = self.box_filter(x, self.r) / N
        var_x = self.box_filter(x * x, self.r) / N - mean_x * mean_x

        A = var_x / (var_x + epsilon)
        b = (1 - A) * mean_x
        m = A * x + b

        # mean_A = self.box_filter(A, self.r) / N
        # mean_b = self.box_filter(b, self.r) / N
        # m = mean_A * x + mean_b
        return x * m


class AdaConvGuidedFilter(nn.Module):
    def __init__(
        self, esa_channels, n_feats, conv=nn.Conv2d, kernel_expand=1, bias=True
    ):
        super(AdaConvGuidedFilter, self).__init__()
        f = esa_channels

        self.conv_f = conv(f, f, kernel_size=1)

        kernel_size = 17
        kernel_expand = kernel_expand
        padding = kernel_size // 2

        self.vec_conv = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=f,
            bias=bias,
        )

        self.hor_conv = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=f,
            bias=bias,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=f,
            out_channels=f,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        y = self.vec_conv(x)
        y = self.hor_conv(y)

        sigma = torch.pow(y, 2)
        epsilon = self.fc(self.gap(y))

        weight = sigma / (sigma + epsilon)

        m = weight * x + (1 - weight)

        return x * m

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .OSAG import OSAG
from .pixelshuffle import pixelshuffle_block


class OmniSR(nn.Module):
    def __init__(
        self,
        state_dict,
        **kwargs,
    ):
        super(OmniSR, self).__init__()
        self.state = state_dict

        bias = True  # Fine to assume this for now
        block_num = 1  # Fine to assume this for now
        ffn_bias = True
        pe = True

        num_feat = state_dict["input.weight"].shape[0] or 64
        num_in_ch = state_dict["input.weight"].shape[1] or 3
        num_out_ch = num_in_ch  # we can just assume this for now. pixelshuffle smh

        pixelshuffle_shape = state_dict["up.0.weight"].shape[0]
        up_scale = math.sqrt(pixelshuffle_shape / num_out_ch)
        if up_scale - int(up_scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be wrong"
            )
        up_scale = int(up_scale)
        res_num = 0
        for key in state_dict.keys():
            if "residual_layer" in key:
                temp_res_num = int(key.split(".")[1])
                if temp_res_num > res_num:
                    res_num = temp_res_num
        res_num = res_num + 1  # zero-indexed

        residual_layer = []
        self.res_num = res_num

        if (
            "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
            in state_dict.keys()
        ):
            rel_pos_bias_weight = state_dict[
                "residual_layer.0.residual_layer.0.layer.2.fn.rel_pos_bias.weight"
            ].shape[0]
            self.window_size = int((math.sqrt(rel_pos_bias_weight) + 1) / 2)
        else:
            self.window_size = 8

        self.up_scale = up_scale

        for _ in range(res_num):
            temp_res = OSAG(
                channel_num=num_feat,
                bias=bias,
                block_num=block_num,
                ffn_bias=ffn_bias,
                window_size=self.window_size,
                pe=pe,
            )
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(
            in_channels=num_in_ch,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.output = nn.Conv2d(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        # chaiNNer specific stuff
        self.model_arch = "OmniSR"
        self.sub_type = "SR"
        self.in_nc = num_in_ch
        self.out_nc = num_out_ch
        self.num_feat = num_feat
        self.scale = up_scale

        self.supports_fp16 = True  # TODO: Test this
        self.supports_bfp16 = True
        self.min_size_restriction = 16

        self.load_state_dict(state_dict, strict=False)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant", 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, : H * self.up_scale, : W * self.up_scale]
        return out

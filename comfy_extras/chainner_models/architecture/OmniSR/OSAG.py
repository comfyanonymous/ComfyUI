#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OSAG.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:08:49 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import torch.nn as nn

from .esa import ESA
from .OSA import OSA_Block


class OSAG(nn.Module):
    def __init__(
        self,
        channel_num=64,
        bias=True,
        block_num=4,
        ffn_bias=False,
        window_size=0,
        pe=False,
    ):
        super(OSAG, self).__init__()

        # print("window_size: %d" % (window_size))
        # print("with_pe", pe)
        # print("ffn_bias: %d" % (ffn_bias))

        # block_script_name = kwargs.get("block_script_name", "OSA")
        # block_class_name = kwargs.get("block_class_name", "OSA_Block")

        # script_name = "." + block_script_name
        # package = __import__(script_name, fromlist=True)
        block_class = OSA_Block  # getattr(package, block_class_name)
        group_list = []
        for _ in range(block_num):
            temp_res = block_class(
                channel_num,
                bias,
                ffn_bias=ffn_bias,
                window_size=window_size,
                with_pe=pe,
            )
            group_list.append(temp_res)
        group_list.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0, bias=bias))
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)

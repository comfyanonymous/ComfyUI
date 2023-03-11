#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch.nn as nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    It is a compact network structure, which performs upsampling in the last layer and no convolution is
    conducted on the HR feature space.
    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_conv (int): Number of convolution layers in the body network. Default: 16.
        upscale (int): Upsampling factor. Default: 4.
        act_type (str): Activation type, options: 'relu', 'prelu', 'leakyrelu'. Default: prelu.
    """

    def __init__(
        self,
        state_dict,
        act_type: str = "prelu",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.model_arch = "SRVGG (RealESRGAN)"
        self.sub_type = "SR"

        self.act_type = act_type

        self.state = state_dict

        if "params" in self.state:
            self.state = self.state["params"]

        self.key_arr = list(self.state.keys())

        self.in_nc = self.get_in_nc()
        self.num_feat = self.get_num_feats()
        self.num_conv = self.get_num_conv()
        self.out_nc = self.in_nc  # :(
        self.pixelshuffle_shape = None  # Defined in get_scale()
        self.scale = self.get_scale()

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(self.in_nc, self.num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=self.num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)  # type: ignore

        # the body structure
        for _ in range(self.num_conv):
            self.body.append(nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=self.num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)  # type: ignore

        # the last conv
        self.body.append(nn.Conv2d(self.num_feat, self.pixelshuffle_shape, 3, 1, 1))  # type: ignore
        # upsample
        self.upsampler = nn.PixelShuffle(self.scale)

        self.load_state_dict(self.state, strict=False)

    def get_num_conv(self) -> int:
        return (int(self.key_arr[-1].split(".")[1]) - 2) // 2

    def get_num_feats(self) -> int:
        return self.state[self.key_arr[0]].shape[0]

    def get_in_nc(self) -> int:
        return self.state[self.key_arr[0]].shape[1]

    def get_scale(self) -> int:
        self.pixelshuffle_shape = self.state[self.key_arr[-1]].shape[0]
        # Assume out_nc is the same as in_nc
        # I cant think of a better way to do that
        self.out_nc = self.in_nc
        scale = math.sqrt(self.pixelshuffle_shape / self.out_nc)
        if scale - int(scale) > 0:
            print(
                "out_nc is probably different than in_nc, scale calculation might be wrong"
            )
        scale = int(scale)
        return scale

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        out += base
        return out

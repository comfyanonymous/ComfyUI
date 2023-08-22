#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import block as B


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)  # type: ignore

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)  # type: ignore

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class SPSRNet(nn.Module):
    def __init__(
        self,
        state_dict,
        norm=None,
        act: str = "leakyrelu",
        upsampler: str = "upconv",
        mode: B.ConvMode = "CNA",
    ):
        super(SPSRNet, self).__init__()
        self.model_arch = "SPSR"
        self.sub_type = "SR"

        self.state = state_dict
        self.norm = norm
        self.act = act
        self.upsampler = upsampler
        self.mode = mode

        self.num_blocks = self.get_num_blocks()

        self.in_nc: int = self.state["model.0.weight"].shape[1]
        self.out_nc: int = self.state["f_HR_conv1.0.bias"].shape[0]

        self.scale = self.get_scale(4)
        print(self.scale)
        self.num_filters: int = self.state["model.0.weight"].shape[0]

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        n_upscale = int(math.log(self.scale, 2))
        if self.scale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(
            self.in_nc, self.num_filters, kernel_size=3, norm_type=None, act_type=None
        )
        rb_blocks = [
            B.RRDB(
                self.num_filters,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=True,
                pad_type="zero",
                norm_type=norm,
                act_type=act,
                mode="CNA",
            )
            for _ in range(self.num_blocks)
        ]
        LR_conv = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=norm,
            act_type=None,
            mode=mode,
        )

        if upsampler == "upconv":
            upsample_block = B.upconv_block
        elif upsampler == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(f"upsample mode [{upsampler}] is not found")
        if self.scale == 3:
            a_upsampler = upsample_block(
                self.num_filters, self.num_filters, 3, act_type=act
            )
        else:
            a_upsampler = [
                upsample_block(self.num_filters, self.num_filters, act_type=act)
                for _ in range(n_upscale)
            ]
        self.HR_conv0_new = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=act,
        )
        self.HR_conv1_new = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlockSPSR(B.sequential(*rb_blocks, LR_conv)),
            *a_upsampler,
            self.HR_conv0_new,
        )

        self.get_g_nopadding = Get_gradient_nopadding()

        self.b_fea_conv = B.conv_block(
            self.in_nc, self.num_filters, kernel_size=3, norm_type=None, act_type=None
        )

        self.b_concat_1 = B.conv_block(
            2 * self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )
        self.b_block_1 = B.RRDB(
            self.num_filters * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm,
            act_type=act,
            mode="CNA",
        )

        self.b_concat_2 = B.conv_block(
            2 * self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )
        self.b_block_2 = B.RRDB(
            self.num_filters * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm,
            act_type=act,
            mode="CNA",
        )

        self.b_concat_3 = B.conv_block(
            2 * self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )
        self.b_block_3 = B.RRDB(
            self.num_filters * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm,
            act_type=act,
            mode="CNA",
        )

        self.b_concat_4 = B.conv_block(
            2 * self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )
        self.b_block_4 = B.RRDB(
            self.num_filters * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm,
            act_type=act,
            mode="CNA",
        )

        self.b_LR_conv = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=norm,
            act_type=None,
            mode=mode,
        )

        if upsampler == "upconv":
            upsample_block = B.upconv_block
        elif upsampler == "pixelshuffle":
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError(f"upsample mode [{upsampler}] is not found")
        if self.scale == 3:
            b_upsampler = upsample_block(
                self.num_filters, self.num_filters, 3, act_type=act
            )
        else:
            b_upsampler = [
                upsample_block(self.num_filters, self.num_filters, act_type=act)
                for _ in range(n_upscale)
            ]

        b_HR_conv0 = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=act,
        )
        b_HR_conv1 = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )

        self.b_module = B.sequential(*b_upsampler, b_HR_conv0, b_HR_conv1)

        self.conv_w = B.conv_block(
            self.num_filters, self.out_nc, kernel_size=1, norm_type=None, act_type=None
        )

        self.f_concat = B.conv_block(
            self.num_filters * 2,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=None,
        )

        self.f_block = B.RRDB(
            self.num_filters * 2,
            kernel_size=3,
            gc=32,
            stride=1,
            bias=True,
            pad_type="zero",
            norm_type=norm,
            act_type=act,
            mode="CNA",
        )

        self.f_HR_conv0 = B.conv_block(
            self.num_filters,
            self.num_filters,
            kernel_size=3,
            norm_type=None,
            act_type=act,
        )
        self.f_HR_conv1 = B.conv_block(
            self.num_filters, self.out_nc, kernel_size=3, norm_type=None, act_type=None
        )

        self.load_state_dict(self.state, strict=False)

    def get_scale(self, min_part: int = 4) -> int:
        n = 0
        for part in list(self.state):
            parts = part.split(".")
            if len(parts) == 3:
                part_num = int(parts[1])
                if part_num > min_part and parts[0] == "model" and parts[2] == "weight":
                    n += 1
        return 2**n

    def get_num_blocks(self) -> int:
        nb = 0
        for part in list(self.state):
            parts = part.split(".")
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == "sub":
                nb = int(parts[3])
        return nb

    def forward(self, x):
        x_grad = self.get_g_nopadding(x)
        x = self.model[0](x)

        x, block_list = self.model[1](x)

        x_ori = x
        for i in range(5):
            x = block_list[i](x)
        x_fea1 = x

        for i in range(5):
            x = block_list[i + 5](x)
        x_fea2 = x

        for i in range(5):
            x = block_list[i + 10](x)
        x_fea3 = x

        for i in range(5):
            x = block_list[i + 15](x)
        x_fea4 = x

        x = block_list[20:](x)
        # short cut
        x = x_ori + x
        x = self.model[2:](x)
        x = self.HR_conv1_new(x)

        x_b_fea = self.b_fea_conv(x_grad)
        x_cat_1 = torch.cat([x_b_fea, x_fea1], dim=1)

        x_cat_1 = self.b_block_1(x_cat_1)
        x_cat_1 = self.b_concat_1(x_cat_1)

        x_cat_2 = torch.cat([x_cat_1, x_fea2], dim=1)

        x_cat_2 = self.b_block_2(x_cat_2)
        x_cat_2 = self.b_concat_2(x_cat_2)

        x_cat_3 = torch.cat([x_cat_2, x_fea3], dim=1)

        x_cat_3 = self.b_block_3(x_cat_3)
        x_cat_3 = self.b_concat_3(x_cat_3)

        x_cat_4 = torch.cat([x_cat_3, x_fea4], dim=1)

        x_cat_4 = self.b_block_4(x_cat_4)
        x_cat_4 = self.b_concat_4(x_cat_4)

        x_cat_4 = self.b_LR_conv(x_cat_4)

        # short cut
        x_cat_4 = x_cat_4 + x_b_fea
        x_branch = self.b_module(x_cat_4)

        # x_out_branch = self.conv_w(x_branch)
        ########
        x_branch_d = x_branch
        x_f_cat = torch.cat([x_branch_d, x], dim=1)
        x_f_cat = self.f_block(x_f_cat)
        x_out = self.f_concat(x_f_cat)
        x_out = self.f_HR_conv0(x_out)
        x_out = self.f_HR_conv1(x_out)

        #########
        # return x_out_branch, x_out, x_grad
        return x_out

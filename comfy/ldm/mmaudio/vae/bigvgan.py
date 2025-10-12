# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
from types import SimpleNamespace
from . import activations
from .alias_free_torch import Activation1d
import comfy.ops
ops = comfy.ops.disable_weight_init

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class AMPBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[0],
                       padding=get_padding(kernel_size, dilation[0])),
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[1],
                       padding=get_padding(kernel_size, dilation[1])),
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[2],
                       padding=get_padding(kernel_size, dilation[2]))
        ])

        self.convs2 = nn.ModuleList([
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1)),
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1)),
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=1,
                       padding=get_padding(kernel_size, 1))
        ])

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        if activation == 'snake':  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta':  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x


class AMPBlock2(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[0],
                       padding=get_padding(kernel_size, dilation[0])),
                ops.Conv1d(channels,
                       channels,
                       kernel_size,
                       1,
                       dilation=dilation[1],
                       padding=get_padding(kernel_size, dilation[1]))
        ])

        self.num_layers = len(self.convs)  # total number of conv layers

        if activation == 'snake':  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta':  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x


class BigVGANVocoder(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    def __init__(self, h):
        super().__init__()
        if isinstance(h, dict):
            h = SimpleNamespace(**h)
        self.h = h

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = ops.Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList([
                        ops.ConvTranspose1d(h.upsample_initial_channel // (2**i),
                                        h.upsample_initial_channel // (2**(i + 1)),
                                        k,
                                        u,
                                        padding=(k - u) // 2)
                ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))

        # post conv
        if h.activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = ops.Conv1d(ch, 1, 7, 1, padding=3)


    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

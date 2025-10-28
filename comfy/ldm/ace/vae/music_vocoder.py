# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_vocoder.py
import torch
from torch import nn

from functools import partial
from math import prod
from typing import Callable, Tuple, List

import numpy as np
import torch.nn.functional as F
from torch.nn.utils.parametrize import remove_parametrizations as remove_weight_norm

from .music_log_mel import LogMelSpectrogram

import comfy.model_management
import comfy.ops
ops = comfy.ops.disable_weight_init


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, comfy.model_management.cast_to(self.weight, dtype=x.dtype, device=x.device), comfy.model_management.cast_to(self.bias, dtype=x.dtype, device=x.device), self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = comfy.model_management.cast_to(self.weight[:, None], dtype=x.dtype, device=x.device) * x + comfy.model_management.cast_to(self.bias[:, None], dtype=x.dtype, device=x.device)
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = ops.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = ops.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = ops.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(torch.empty((dim)), requires_grad=False)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = comfy.model_management.cast_to(self.gamma, dtype=x.dtype, device=x.device) * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x


class ParallelConvNeXtBlock(nn.Module):
    def __init__(self, kernel_sizes: List[int], *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(kernel_size=kernel_size, *args, **kwargs)
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [block(x, apply_residual=False) for block in self.blocks] + [x],
            dim=1,
        ).sum(dim=1)


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        kernel_sizes: Tuple[int] = (7,),
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.channel_layers = nn.ModuleList()
        stem = nn.Sequential(
            ops.Conv1d(
                input_channels,
                dims[0],
                kernel_size=7,
                padding=3,
                padding_mode="replicate",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.channel_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                ops.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.channel_layers.append(mid_layer)

        block_fn = (
            partial(ConvNeXtBlock, kernel_size=kernel_sizes[0])
            if len(kernel_sizes) == 1
            else partial(ParallelConvNeXtBlock, kernel_sizes=kernel_sizes)
        )

        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    block_fn(
                        dim=dims[i],
                        drop_path=drop_path_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for channel_layer, stage in zip(self.channel_layers, self.stages):
            x = channel_layer(x)
            x = stage(x)

        return self.norm(x)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    ops.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: Tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int]] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        use_template: bool = True,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = torch.nn.utils.parametrizations.weight_norm(
            ops.Conv1d(
                num_mels,
                upsample_initial_channel,
                pre_conv_kernel_size,
                1,
                padding=get_padding(pre_conv_kernel_size),
            )
        )

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                torch.nn.utils.parametrizations.weight_norm(
                    ops.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            if not use_template:
                continue

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    ops.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(ops.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.activation_post = post_activation()
        self.conv_post = torch.nn.utils.parametrizations.weight_norm(
            ops.Conv1d(
                ch,
                1,
                post_conv_kernel_size,
                1,
                padding=get_padding(post_conv_kernel_size),
            )
        )

    def forward(self, x, template=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.use_template:
                x = x + self.noise_convs[i](template)

            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ADaMoSHiFiGANV1(nn.Module):
    def __init__(
        self,
        input_channels: int = 128,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [128, 256, 384, 512],
        drop_path_rate: float = 0.0,
        kernel_sizes: Tuple[int] = (7,),
        upsample_rates: Tuple[int] = (4, 4, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (8, 8, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11, 13),
        resblock_dilation_sizes: Tuple[Tuple[int]] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 512,
        upsample_initial_channel: int = 1024,
        use_template: bool = False,
        pre_conv_kernel_size: int = 13,
        post_conv_kernel_size: int = 13,
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: int = 40,
        f_max: int = 16000,
        n_mels: int = 128,
    ):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=input_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            kernel_sizes=kernel_sizes,
        )

        self.head = HiFiGANGenerator(
            hop_length=hop_length,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
            use_template=use_template,
            pre_conv_kernel_size=pre_conv_kernel_size,
            post_conv_kernel_size=post_conv_kernel_size,
        )
        self.sampling_rate = sampling_rate
        self.mel_transform = LogMelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.eval()

    @torch.no_grad()
    def decode(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)

    def forward(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

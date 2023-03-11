# pylint: skip-file
# type: ignore
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .fused_act import FusedLeakyReLU, fused_leaky_relu


class NormStyleCode(nn.Module):
    def forward(self, x):
        """Normalize the style codes.
        Args:
            x (Tensor): Style codes with shape (b, c).
        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.
    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        bias_init_val=0,
        lr_mul=1,
        activation=None,
    ):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ["fused_lrelu", None]:
            raise ValueError(
                f"Wrong activation value in EqualLinear: {activation}"
                "Supported ones are: ['fused_lrelu', None]."
            )
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == "fused_lrelu":
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, bias={self.bias is not None})"
        )


class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.
    There is no bias in ModulatedConv2d.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_style_feat,
        demodulate=True,
        sample_mode=None,
        eps=1e-8,
        interpolation_mode="bilinear",
    ):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps
        self.interpolation_mode = interpolation_mode
        if self.interpolation_mode == "nearest":
            self.align_corners = None
        else:
            self.align_corners = False

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat,
            in_channels,
            bias=True,
            bias_init_val=1,
            lr_mul=1,
            activation=None,
        )

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.
        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(
            b * self.out_channels, c, self.kernel_size, self.kernel_size
        )

        if self.sample_mode == "upsample":
            x = F.interpolate(
                x,
                scale_factor=2,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )
        elif self.sample_mode == "downsample":
            x = F.interpolate(
                x,
                scale_factor=0.5,
                mode=self.interpolation_mode,
                align_corners=self.align_corners,
            )

        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        # weight: (b*c_out, c_in, k, k), groups=b
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"demodulate={self.demodulate}, sample_mode={self.sample_mode})"
        )


class StyleConv(nn.Module):
    """Style conv.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether demodulate in the conv layer. Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        num_style_feat,
        demodulate=True,
        sample_mode=None,
        interpolation_mode="bilinear",
    ):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size,
            num_style_feat,
            demodulate=demodulate,
            sample_mode=sample_mode,
            interpolation_mode=interpolation_mode,
        )
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activate = FusedLeakyReLU(out_channels)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style)
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # activation (with bias)
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    """To RGB from features.
    Args:
        in_channels (int): Channel number of input.
        num_style_feat (int): Channel number of style features.
        upsample (bool): Whether to upsample. Default: True.
    """

    def __init__(
        self, in_channels, num_style_feat, upsample=True, interpolation_mode="bilinear"
    ):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.interpolation_mode = interpolation_mode
        if self.interpolation_mode == "nearest":
            self.align_corners = None
        else:
            self.align_corners = False
        self.modulated_conv = ModulatedConv2d(
            in_channels,
            3,
            kernel_size=1,
            num_style_feat=num_style_feat,
            demodulate=False,
            sample_mode=None,
            interpolation_mode=interpolation_mode,
        )
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        """Forward function.
        Args:
            x (Tensor): Feature tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).
            skip (Tensor): Base/skip tensor. Default: None.
        Returns:
            Tensor: RGB images.
        """
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(
                    skip,
                    scale_factor=2,
                    mode=self.interpolation_mode,
                    align_corners=self.align_corners,
                )
            out = out + skip
        return out


class ConstantInput(nn.Module):
    """Constant input.
    Args:
        num_channel (int): Channel number of constant input.
        size (int): Spatial size of constant input.
    """

    def __init__(self, num_channel, size):
        super(ConstantInput, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch):
        out = self.weight.repeat(batch, 1, 1, 1)
        return out


class StyleGAN2GeneratorBilinear(nn.Module):
    """StyleGAN2 Generator.
    Args:
        out_size (int): The spatial size of outputs.
        num_style_feat (int): Channel number of style features. Default: 512.
        num_mlp (int): Layer number of MLP style layers. Default: 8.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        lr_mlp (float): Learning rate multiplier for mlp layers. Default: 0.01.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(
        self,
        out_size,
        num_style_feat=512,
        num_mlp=8,
        channel_multiplier=2,
        lr_mlp=0.01,
        narrow=1,
        interpolation_mode="bilinear",
    ):
        super(StyleGAN2GeneratorBilinear, self).__init__()
        # Style MLP layers
        self.num_style_feat = num_style_feat
        style_mlp_layers = [NormStyleCode()]
        for i in range(num_mlp):
            style_mlp_layers.append(
                EqualLinear(
                    num_style_feat,
                    num_style_feat,
                    bias=True,
                    bias_init_val=0,
                    lr_mul=lr_mlp,
                    activation="fused_lrelu",
                )
            )
        self.style_mlp = nn.Sequential(*style_mlp_layers)

        channels = {
            "4": int(512 * narrow),
            "8": int(512 * narrow),
            "16": int(512 * narrow),
            "32": int(512 * narrow),
            "64": int(256 * channel_multiplier * narrow),
            "128": int(128 * channel_multiplier * narrow),
            "256": int(64 * channel_multiplier * narrow),
            "512": int(32 * channel_multiplier * narrow),
            "1024": int(16 * channel_multiplier * narrow),
        }
        self.channels = channels

        self.constant_input = ConstantInput(channels["4"], size=4)
        self.style_conv1 = StyleConv(
            channels["4"],
            channels["4"],
            kernel_size=3,
            num_style_feat=num_style_feat,
            demodulate=True,
            sample_mode=None,
            interpolation_mode=interpolation_mode,
        )
        self.to_rgb1 = ToRGB(
            channels["4"],
            num_style_feat,
            upsample=False,
            interpolation_mode=interpolation_mode,
        )

        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.num_latent = self.log_size * 2 - 2

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channels = channels["4"]
        # noise
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            shape = [1, 1, resolution, resolution]
            self.noises.register_buffer(f"noise{layer_idx}", torch.randn(*shape))
        # style convs and to_rgbs
        for i in range(3, self.log_size + 1):
            out_channels = channels[f"{2**i}"]
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode="upsample",
                    interpolation_mode=interpolation_mode,
                )
            )
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None,
                    interpolation_mode=interpolation_mode,
                )
            )
            self.to_rgbs.append(
                ToRGB(
                    out_channels,
                    num_style_feat,
                    upsample=True,
                    interpolation_mode=interpolation_mode,
                )
            )
            in_channels = out_channels

    def make_noise(self):
        """Make noise for noise injection."""
        device = self.constant_input.weight.device
        noises = [torch.randn(1, 1, 4, 4, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2**i, 2**i, device=device))

        return noises

    def get_latent(self, x):
        return self.style_mlp(x)

    def mean_latent(self, num_latent):
        latent_in = torch.randn(
            num_latent, self.num_style_feat, device=self.constant_input.weight.device
        )
        latent = self.style_mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(
        self,
        styles,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        truncation=1,
        truncation_latent=None,
        inject_index=None,
        return_latents=False,
    ):
        """Forward function for StyleGAN2Generator.
        Args:
            styles (list[Tensor]): Sample codes of styles.
            input_is_latent (bool): Whether input is latent style.
                Default: False.
            noise (Tensor | None): Input noise or None. Default: None.
            randomize_noise (bool): Randomize noise, used when 'noise' is
                False. Default: True.
            truncation (float): TODO. Default: 1.
            truncation_latent (Tensor | None): TODO. Default: None.
            inject_index (int | None): The injection index for mixing noise.
                Default: None.
            return_latents (bool): Whether to return style latents.
                Default: False.
        """
        # style codes -> latents with Style MLP layer
        if not input_is_latent:
            styles = [self.style_mlp(s) for s in styles]
        # noises
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers  # for each style conv layer
            else:  # use the stored noise
                noise = [
                    getattr(self.noises, f"noise{i}") for i in range(self.num_layers)
                ]
        # style truncation
        if truncation < 1:
            style_truncation = []
            for style in styles:
                style_truncation.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )
            styles = style_truncation
        # get style latent with injection
        if len(styles) == 1:
            inject_index = self.num_latent

            if styles[0].ndim < 3:
                # repeat latent code for all the layers
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:  # used for encoder with different latent code for each layer
                latent = styles[0]
        elif len(styles) == 2:  # mixing noises
            if inject_index is None:
                inject_index = random.randint(1, self.num_latent - 1)
            latent1 = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.num_latent - inject_index, 1)
            )
            latent = torch.cat([latent1, latent2], 1)

        # main generation
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        else:
            return image, None


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.
    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        bias_init_val=0,
    ):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size},"
            f" stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        downsample=False,
        bias=True,
        activate=True,
        interpolation_mode="bilinear",
    ):
        layers = []
        self.interpolation_mode = interpolation_mode
        # downsample
        if downsample:
            if self.interpolation_mode == "nearest":
                self.align_corners = None
            else:
                self.align_corners = False

            layers.append(
                torch.nn.Upsample(
                    scale_factor=0.5,
                    mode=interpolation_mode,
                    align_corners=self.align_corners,
                )
            )
        stride = 1
        self.padding = kernel_size // 2
        # conv
        layers.append(
            EqualConv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=self.padding,
                bias=bias and not activate,
            )
        )
        # activation
        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
    """

    def __init__(self, in_channels, out_channels, interpolation_mode="bilinear"):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(
            in_channels,
            out_channels,
            3,
            downsample=True,
            interpolation_mode=interpolation_mode,
            bias=True,
            activate=True,
        )
        self.skip = ConvLayer(
            in_channels,
            out_channels,
            1,
            downsample=True,
            interpolation_mode=interpolation_mode,
            bias=False,
            activate=False,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out

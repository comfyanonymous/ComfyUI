# From https://github.com/Koushik0901/Swift-SRGAN/blob/master/swift-srgan/models.py

import torch
from torch import nn


class SeperableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True
    ):
        super(SeperableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_act=True,
        use_bn=True,
        discriminator=False,
        **kwargs,
    ):
        super(ConvBlock, self).__init__()

        self.use_act = use_act
        self.cnn = SeperableConv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()

        self.conv = SeperableConv2d(
            in_channels,
            in_channels * scale_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.ps = nn.PixelShuffle(
            scale_factor
        )  # (in_channels * 4, H, W) -> (in_channels, H*2, W*2)
        self.act = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block1 = ConvBlock(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.block2 = ConvBlock(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, use_act=False
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x


class Generator(nn.Module):
    """Swift-SRGAN Generator
    Args:
        in_channels (int): number of input image channels.
        num_channels (int): number of hidden channels.
        num_blocks (int): number of residual blocks.
        upscale_factor (int): factor to upscale the image [2x, 4x, 8x].
    Returns:
        torch.Tensor: super resolution image
    """

    def __init__(
        self,
        state_dict,
    ):
        super(Generator, self).__init__()
        self.model_arch = "Swift-SRGAN"
        self.sub_type = "SR"
        self.state = state_dict
        if "model" in self.state:
            self.state = self.state["model"]

        self.in_nc: int = self.state["initial.cnn.depthwise.weight"].shape[0]
        self.out_nc: int = self.state["final_conv.pointwise.weight"].shape[0]
        self.num_filters: int = self.state["initial.cnn.pointwise.weight"].shape[0]
        self.num_blocks = len(
            set([x.split(".")[1] for x in self.state.keys() if "residual" in x])
        )
        self.scale: int = 2 ** len(
            set([x.split(".")[1] for x in self.state.keys() if "upsampler" in x])
        )

        in_channels = self.in_nc
        num_channels = self.num_filters
        num_blocks = self.num_blocks
        upscale_factor = self.scale

        self.supports_fp16 = True
        self.supports_bfp16 = True
        self.min_size_restriction = None

        self.initial = ConvBlock(
            in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False
        )
        self.residual = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.convblock = ConvBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )
        self.upsampler = nn.Sequential(
            *[
                UpsampleBlock(num_channels, scale_factor=2)
                for _ in range(upscale_factor // 2)
            ]
        )
        self.final_conv = SeperableConv2d(
            num_channels, in_channels, kernel_size=9, stride=1, padding=4
        )

        self.load_state_dict(self.state, strict=False)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residual(initial)
        x = self.convblock(x) + initial
        x = self.upsampler(x)
        return (torch.tanh(self.final_conv(x)) + 1) / 2

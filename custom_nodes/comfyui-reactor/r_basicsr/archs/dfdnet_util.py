import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.spectral_norm import spectral_norm


class BlurFunctionBackward(Function):

    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        grad_input = F.conv2d(grad_output, kernel_flip, padding=1, groups=grad_output.shape[1])
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, _ = ctx.saved_tensors
        grad_input = F.conv2d(gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1])
        return grad_input, None, None


class BlurFunction(Function):

    @staticmethod
    def forward(ctx, x, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(x, kernel, padding=1, groups=x.shape[1])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)
        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):

    def __init__(self, channel):
        super().__init__()
        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3)
        kernel = kernel / kernel.sum()
        kernel_flip = torch.flip(kernel, [2, 3])

        self.kernel = kernel.repeat(channel, 1, 1, 1)
        self.kernel_flip = kernel_flip.repeat(channel, 1, 1, 1)

    def forward(self, x):
        return blur(x, self.kernel.type_as(x), self.kernel_flip.type_as(x))


def calc_mean_std(feat, eps=1e-5):
    """Calculate mean and std for adaptive_instance_normalization.

    Args:
        feat (Tensor): 4D tensor.
        eps (float): A small value added to the variance to avoid
            divide-by-zero. Default: 1e-5.
    """
    size = feat.size()
    assert len(size) == 4, 'The input feature should be 4D tensor.'
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    """Adaptive instance normalization.

    Adjust the reference features to have the similar color and illuminations
    as those in the degradate features.

    Args:
        content_feat (Tensor): The reference feature.
        style_feat (Tensor): The degradate features.
    """
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def AttentionBlock(in_channel):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)), nn.LeakyReLU(0.2, True),
        spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)))


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """Conv block used in MSDilationBlock."""

    return nn.Sequential(
        spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) // 2) * dilation,
                bias=bias)),
        nn.LeakyReLU(0.2),
        spectral_norm(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((kernel_size - 1) // 2) * dilation,
                bias=bias)),
    )


class MSDilationBlock(nn.Module):
    """Multi-scale dilation block."""

    def __init__(self, in_channels, kernel_size=3, dilation=(1, 1, 1, 1), bias=True):
        super(MSDilationBlock, self).__init__()

        self.conv_blocks = nn.ModuleList()
        for i in range(4):
            self.conv_blocks.append(conv_block(in_channels, in_channels, kernel_size, dilation=dilation[i], bias=bias))
        self.conv_fusion = spectral_norm(
            nn.Conv2d(
                in_channels * 4,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=bias))

    def forward(self, x):
        out = []
        for i in range(4):
            out.append(self.conv_blocks[i](x))
        out = torch.cat(out, 1)
        out = self.conv_fusion(out) + x
        return out


class UpResBlock(nn.Module):

    def __init__(self, in_channel):
        super(UpResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
        )

    def forward(self, x):
        out = x + self.body(x)
        return out

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from r_basicsr.utils.registry import ARCH_REGISTRY


class DenseBlocksTemporalReduce(nn.Module):
    """A concatenation of 3 dense blocks with reduction in temporal dimension.

    Note that the output temporal dimension is 6 fewer the input temporal dimension, since there are 3 blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_feat=64, num_grow_ch=32, adapt_official_weights=False):
        super(DenseBlocksTemporalReduce, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.temporal_reduce1 = nn.Sequential(
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(num_feat, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

        self.temporal_reduce2 = nn.Sequential(
            nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + num_grow_ch,
                num_feat + num_grow_ch, (1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=True), nn.BatchNorm3d(num_feat + num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(num_feat + num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

        self.temporal_reduce3 = nn.Sequential(
            nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + 2 * num_grow_ch,
                num_feat + 2 * num_grow_ch, (1, 1, 1),
                stride=(1, 1, 1),
                padding=(0, 0, 0),
                bias=True), nn.BatchNorm3d(num_feat + 2 * num_grow_ch, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                num_feat + 2 * num_grow_ch, num_grow_ch, (3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_grow_ch * 3, 1, h, w).
        """
        x1 = self.temporal_reduce1(x)
        x1 = torch.cat((x[:, :, 1:-1, :, :], x1), 1)

        x2 = self.temporal_reduce2(x1)
        x2 = torch.cat((x1[:, :, 1:-1, :, :], x2), 1)

        x3 = self.temporal_reduce3(x2)
        x3 = torch.cat((x2[:, :, 1:-1, :, :], x3), 1)

        return x3


class DenseBlocks(nn.Module):
    """ A concatenation of N dense blocks.

    Args:
        num_feat (int): Number of channels in the blocks. Default: 64.
        num_grow_ch (int): Growing factor of the dense blocks. Default: 32.
        num_block (int): Number of dense blocks. The values are:
            DUF-S (16 layers): 3
            DUF-M (18 layers): 9
            DUF-L (52 layers): 21
        adapt_official_weights (bool): Whether to adapt the weights translated from the official implementation.
            Set to false if you want to train from scratch. Default: False.
    """

    def __init__(self, num_block, num_feat=64, num_grow_ch=16, adapt_official_weights=False):
        super(DenseBlocks, self).__init__()
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.dense_blocks = nn.ModuleList()
        for i in range(0, num_block):
            self.dense_blocks.append(
                nn.Sequential(
                    nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum), nn.ReLU(inplace=True),
                    nn.Conv3d(
                        num_feat + i * num_grow_ch,
                        num_feat + i * num_grow_ch, (1, 1, 1),
                        stride=(1, 1, 1),
                        padding=(0, 0, 0),
                        bias=True), nn.BatchNorm3d(num_feat + i * num_grow_ch, eps=eps, momentum=momentum),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                        num_feat + i * num_grow_ch,
                        num_grow_ch, (3, 3, 3),
                        stride=(1, 1, 1),
                        padding=(1, 1, 1),
                        bias=True)))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor with shape (b, num_feat, t, h, w).

        Returns:
            Tensor: Output with shape (b, num_feat + num_block * num_grow_ch, t, h, w).
        """
        for i in range(0, len(self.dense_blocks)):
            y = self.dense_blocks[i](x)
            x = torch.cat((x, y), 1)
        return x


class DynamicUpsamplingFilter(nn.Module):
    """Dynamic upsampling filter used in DUF.

    Ref: https://github.com/yhjo09/VSR-DUF.
    It only supports input with 3 channels. And it applies the same filters to 3 channels.

    Args:
        filter_size (tuple): Filter size of generated filters. The shape is (kh, kw). Default: (5, 5).
    """

    def __init__(self, filter_size=(5, 5)):
        super(DynamicUpsamplingFilter, self).__init__()
        if not isinstance(filter_size, tuple):
            raise TypeError(f'The type of filter_size must be tuple, but got type{filter_size}')
        if len(filter_size) != 2:
            raise ValueError(f'The length of filter size must be 2, but got {len(filter_size)}.')
        # generate a local expansion filter, similar to im2col
        self.filter_size = filter_size
        filter_prod = np.prod(filter_size)
        expansion_filter = torch.eye(int(filter_prod)).view(filter_prod, 1, *filter_size)  # (kh*kw, 1, kh, kw)
        self.expansion_filter = expansion_filter.repeat(3, 1, 1, 1)  # repeat for all the 3 channels

    def forward(self, x, filters):
        """Forward function for DynamicUpsamplingFilter.

        Args:
            x (Tensor): Input image with 3 channels. The shape is (n, 3, h, w).
            filters (Tensor): Generated dynamic filters.
                The shape is (n, filter_prod, upsampling_square, h, w).
                filter_prod: prod of filter kernel size, e.g., 1*5*5=25.
                upsampling_square: similar to pixel shuffle,
                    upsampling_square = upsampling * upsampling
                    e.g., for x 4 upsampling, upsampling_square= 4*4 = 16

        Returns:
            Tensor: Filtered image with shape (n, 3*upsampling_square, h, w)
        """
        n, filter_prod, upsampling_square, h, w = filters.size()
        kh, kw = self.filter_size
        expanded_input = F.conv2d(
            x, self.expansion_filter.to(x), padding=(kh // 2, kw // 2), groups=3)  # (n, 3*filter_prod, h, w)
        expanded_input = expanded_input.view(n, 3, filter_prod, h, w).permute(0, 3, 4, 1,
                                                                              2)  # (n, h, w, 3, filter_prod)
        filters = filters.permute(0, 3, 4, 1, 2)  # (n, h, w, filter_prod, upsampling_square]
        out = torch.matmul(expanded_input, filters)  # (n, h, w, 3, upsampling_square)
        return out.permute(0, 3, 4, 1, 2).view(n, 3 * upsampling_square, h, w)


@ARCH_REGISTRY.register()
class DUF(nn.Module):
    """Network architecture for DUF

    Paper: Jo et.al. Deep Video Super-Resolution Network Using Dynamic
            Upsampling Filters Without Explicit Motion Compensation, CVPR, 2018
    Code reference:
        https://github.com/yhjo09/VSR-DUF
    For all the models below, 'adapt_official_weights' is only necessary when
    loading the weights converted from the official TensorFlow weights.
    Please set it to False if you are training the model from scratch.

    There are three models with different model size: DUF16Layers, DUF28Layers,
    and DUF52Layers. This class is the base class for these models.

    Args:
        scale (int): The upsampling factor. Default: 4.
        num_layer (int): The number of layers. Default: 52.
        adapt_official_weights_weights (bool): Whether to adapt the weights
            translated from the official implementation. Set to false if you
            want to train from scratch. Default: False.
    """

    def __init__(self, scale=4, num_layer=52, adapt_official_weights=False):
        super(DUF, self).__init__()
        self.scale = scale
        if adapt_official_weights:
            eps = 1e-3
            momentum = 1e-3
        else:  # pytorch default values
            eps = 1e-05
            momentum = 0.1

        self.conv3d1 = nn.Conv3d(3, 64, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.dynamic_filter = DynamicUpsamplingFilter((5, 5))

        if num_layer == 16:
            num_block = 3
            num_grow_ch = 32
        elif num_layer == 28:
            num_block = 9
            num_grow_ch = 16
        elif num_layer == 52:
            num_block = 21
            num_grow_ch = 16
        else:
            raise ValueError(f'Only supported (16, 28, 52) layers, but got {num_layer}.')

        self.dense_block1 = DenseBlocks(
            num_block=num_block, num_feat=64, num_grow_ch=num_grow_ch,
            adapt_official_weights=adapt_official_weights)  # T = 7
        self.dense_block2 = DenseBlocksTemporalReduce(
            64 + num_grow_ch * num_block, num_grow_ch, adapt_official_weights=adapt_official_weights)  # T = 1
        channels = 64 + num_grow_ch * num_block + num_grow_ch * 3
        self.bn3d2 = nn.BatchNorm3d(channels, eps=eps, momentum=momentum)
        self.conv3d2 = nn.Conv3d(channels, 256, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)

        self.conv3d_r1 = nn.Conv3d(256, 256, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_r2 = nn.Conv3d(256, 3 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

        self.conv3d_f1 = nn.Conv3d(256, 512, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)
        self.conv3d_f2 = nn.Conv3d(
            512, 1 * 5 * 5 * (scale**2), (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input with shape (b, 7, c, h, w)

        Returns:
            Tensor: Output with shape (b, c, h * scale, w * scale)
        """
        num_batches, num_imgs, _, h, w = x.size()

        x = x.permute(0, 2, 1, 3, 4)  # (b, c, 7, h, w) for Conv3D
        x_center = x[:, :, num_imgs // 2, :, :]

        x = self.conv3d1(x)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = F.relu(self.bn3d2(x), inplace=True)
        x = F.relu(self.conv3d2(x), inplace=True)

        # residual image
        res = self.conv3d_r2(F.relu(self.conv3d_r1(x), inplace=True))

        # filter
        filter_ = self.conv3d_f2(F.relu(self.conv3d_f1(x), inplace=True))
        filter_ = F.softmax(filter_.view(num_batches, 25, self.scale**2, h, w), dim=1)

        # dynamic filter
        out = self.dynamic_filter(x_center, filter_)
        out += res.squeeze_(2)
        out = F.pixel_shuffle(out, self.scale)

        return out

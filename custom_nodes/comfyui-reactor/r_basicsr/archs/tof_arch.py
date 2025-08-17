import torch
from torch import nn as nn
from torch.nn import functional as F

from r_basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import flow_warp


class BasicModule(nn.Module):
    """Basic module of SPyNet.

    Note that unlike the architecture in spynet_arch.py, the basic module
    here contains batch normalization.
    """

    def __init__(self):
        super(BasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].

        Returns:
            Tensor: Estimated flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class SPyNetTOF(nn.Module):
    """SPyNet architecture for TOF.

    Note that this implementation is specifically for TOFlow. Please use
    spynet_arch.py for general use. They differ in the following aspects:
        1. The basic modules here contain BatchNorm.
        2. Normalization and denormalization are not done here, as
            they are done in TOFlow.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network
    Code reference:
        https://github.com/Coldog2333/pytoflow

    Args:
        load_path (str): Path for pretrained SPyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SPyNetTOF, self).__init__()

        self.basic_module = nn.ModuleList([BasicModule() for _ in range(4)])
        if load_path:
            self.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage)['params'])

    def forward(self, ref, supp):
        """
        Args:
            ref (Tensor): Reference image with shape of (b, 3, h, w).
            supp: The supporting image to be warped: (b, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (b, 2, h, w).
        """
        num_batches, _, h, w = ref.size()
        ref = [ref]
        supp = [supp]

        # generate downsampled frames
        for _ in range(3):
            ref.insert(0, F.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
            supp.insert(0, F.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))

        # flow computation
        flow = ref[0].new_zeros(num_batches, 2, h // 16, w // 16)
        for i in range(4):
            flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            flow = flow_up + self.basic_module[i](
                torch.cat([ref[i], flow_warp(supp[i], flow_up.permute(0, 2, 3, 1)), flow_up], 1))
        return flow


@ARCH_REGISTRY.register()
class TOFlow(nn.Module):
    """PyTorch implementation of TOFlow.

    In TOFlow, the LR frames are pre-upsampled and have the same size with
    the GT frames.
    Paper:
        Xue et al., Video Enhancement with Task-Oriented Flow, IJCV 2018
    Code reference:
        1. https://github.com/anchen1011/toflow
        2. https://github.com/Coldog2333/pytoflow

    Args:
        adapt_official_weights (bool): Whether to adapt the weights translated
            from the official implementation. Set to false if you want to
            train from scratch. Default: False
    """

    def __init__(self, adapt_official_weights=False):
        super(TOFlow, self).__init__()
        self.adapt_official_weights = adapt_official_weights
        self.ref_idx = 0 if adapt_official_weights else 3

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # flow estimation module
        self.spynet = SPyNetTOF()

        # reconstruction module
        self.conv_1 = nn.Conv2d(3 * 7, 64, 9, 1, 4)
        self.conv_2 = nn.Conv2d(64, 64, 9, 1, 4)
        self.conv_3 = nn.Conv2d(64, 64, 1)
        self.conv_4 = nn.Conv2d(64, 3, 1)

        # activation function
        self.relu = nn.ReLU(inplace=True)

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, lrs):
        """
        Args:
            lrs: Input lr frames: (b, 7, 3, h, w).

        Returns:
            Tensor: SR frame: (b, 3, h, w).
        """
        # In the official implementation, the 0-th frame is the reference frame
        if self.adapt_official_weights:
            lrs = lrs[:, [3, 0, 1, 2, 4, 5, 6], :, :, :]

        num_batches, num_lrs, _, h, w = lrs.size()

        lrs = self.normalize(lrs.view(-1, 3, h, w))
        lrs = lrs.view(num_batches, num_lrs, 3, h, w)

        lr_ref = lrs[:, self.ref_idx, :, :, :]
        lr_aligned = []
        for i in range(7):  # 7 frames
            if i == self.ref_idx:
                lr_aligned.append(lr_ref)
            else:
                lr_supp = lrs[:, i, :, :, :]
                flow = self.spynet(lr_ref, lr_supp)
                lr_aligned.append(flow_warp(lr_supp, flow.permute(0, 2, 3, 1)))

        # reconstruction
        hr = torch.stack(lr_aligned, dim=1)
        hr = hr.view(num_batches, -1, h, w)
        hr = self.relu(self.conv_1(hr))
        hr = self.relu(self.conv_2(hr))
        hr = self.relu(self.conv_3(hr))
        hr = self.conv_4(hr) + lr_ref

        return self.denormalize(hr)

"""
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_scripts/interpolate.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/models/ssldtm.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/util_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/twodee_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/pytorch_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/distance_transform_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/sketchers_v1.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/interpolator_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/gridnet_v1.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/flow_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_util/softsplat_v0.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/raft_v1/rfr_new.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/raft_v1/extractor.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/raft_v1/update.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/raft_v1/corr.py
https://github.com/ShuhongChen/eisai-anime-interpolator/blob/master/_train/frame_interpolation/helpers/raft_v1/utils.py
"""

import copy
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as F
import gc
from PIL import Image, ImageFile, ImageFont, ImageDraw
import inspect
from scipy import interpolate
import kornia
import math
from argparse import Namespace
import torch.nn as nn
import numpy as np
import os
from functools import partial
import pathlib
import PIL
import re
import requests
from scipy.spatial.transform import Rotation
import scipy
import shutil
import torchvision.transforms as T
import time
import torch
import torchvision as tv
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm as std_tqdm
from tqdm.auto import trange as std_trange
from vfi_models.ops import FunctionSoftsplat, batch_edt
from comfy.model_management import get_torch_device

device = get_torch_device()
autocast = torch.autocast
tqdm = partial(std_tqdm, dynamic_ncols=True)
trange = partial(std_trange, dynamic_ncols=True)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pixel_ij(x, rounding=True):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return tuple(
        pixel_rounder(i, rounding)
        for i in (x if isinstance(x, tuple) or isinstance(x, list) else (x, x))
    )


def rescale_dry(x, factor):
    h, w = x[-2:] if isinstance(x, tuple) or isinstance(x, list) else I(x).size
    return (h * factor, w * factor)


def pixel_rounder(n, mode):
    if mode == True or mode == "round":
        return round(n)
    elif mode == "ceil":
        return math.ceil(n)
    elif mode == "floor":
        return math.floor(n)
    else:
        return n


def diam(x):
    if isinstance(x, tuple) or isinstance(x, list):
        h, w = x[-2:]
    elif isinstance(x, I):
        h, w = x.size
    else:
        h, w = x.shape[-2:]
    return np.sqrt(h**2 + w**2)


def pixel_logit(x, pixel_margin=1):
    x = (x * (255 - 2 * pixel_margin) + pixel_margin) / 255
    return torch.log(x / (1 - x))


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata((x1, y1), dx, (x0, y0), method="cubic", fill_value=0)

    flow_y = interpolate.griddata((x1, y1), dy, (x0, y0), method="cubic", fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print(img.size())
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode="bilinear"):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convr1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )
        self.convq1 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2)
        )

        self.convz2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convr2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )
        self.convq2 = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0)
        )

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82 + 64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = 0.25 * self.mask(net)
        return net, mask, delta_flow


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class BasicEncoder1(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder1, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


##################################################
#  RFR is implemented based on RAFT optical flow #
##################################################


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(
        gridX,
        requires_grad=False,
    ).cuda()
    gridY = torch.tensor(
        gridY,
        requires_grad=False,
    ).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v
    # range -1 to 1
    x = 2 * (x / (W - 1) - 0.5)
    y = 2 * (y / (H - 1) - 0.5)
    # stacking X and Y
    grid = torch.stack((x, y), dim=3)
    # Sample pixels using bilinear interpolation.
    imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    return imgOut


class ErrorAttention(nn.Module):
    """A three-layer network for predicting mask"""

    def __init__(self, input, output):
        super(ErrorAttention, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(38, output, 3, padding=1)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x1):
        x = self.prelu1(self.conv1(x1))
        x = self.prelu2(torch.cat([self.conv2(x), x1], dim=1))
        x = self.conv3(x)
        return x


class RFR(nn.Module):
    def __init__(self, args):
        super(RFR, self).__init__()
        self.attention2 = ErrorAttention(6, 1)
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        args.dropout = 0
        self.args = args

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn="none", dropout=args.dropout)
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(
        self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        H, W = image1.size()[2:4]
        H8 = H // 8 * 8
        W8 = W // 8 * 8

        if flow_init is not None:
            flow_init_resize = F.interpolate(
                flow_init, size=(H8 // 8, W8 // 8), mode="nearest"
            )

            flow_init_resize[:, :1] = (
                flow_init_resize[:, :1].clone() * (W8 // 8 * 1.0) / flow_init.size()[3]
            )
            flow_init_resize[:, 1:] = (
                flow_init_resize[:, 1:].clone() * (H8 // 8 * 1.0) / flow_init.size()[2]
            )

            if not hasattr(self.args, "not_use_rfr_mask") or (
                hasattr(self.args, "not_use_rfr_mask")
                and (not self.args.not_use_rfr_mask)
            ):
                im18 = F.interpolate(image1, size=(H8 // 8, W8 // 8), mode="bilinear")
                im28 = F.interpolate(image2, size=(H8 // 8, W8 // 8), mode="bilinear")

                warp21 = backwarp(im28, flow_init_resize)
                error21 = torch.sum(torch.abs(warp21 - im18), dim=1, keepdim=True)
                # print('errormin', error21.min(), error21.max())
                f12init = (
                    torch.exp(
                        -self.attention2(
                            torch.cat([im18, error21, flow_init_resize], dim=1)
                        )
                        ** 2
                    )
                    * flow_init_resize
                )
        else:
            flow_init_resize = None
            flow_init = torch.zeros(
                image1.size()[0], 2, image1.size()[2] // 8, image1.size()[3] // 8
            ).cuda()
            error21 = torch.zeros(
                image1.size()[0], 1, image1.size()[2] // 8, image1.size()[3] // 8
            ).cuda()

            f12_init = flow_init
            # print('None inital flow!')

        image1 = F.interpolate(image1, size=(H8, W8), mode="bilinear")
        image2 = F.interpolate(image2, size=(H8, W8), mode="bilinear")

        f12s, f12, f12_init = self.forward_pred(
            image1, image2, iters, flow_init_resize, upsample, test_mode
        )

        if hasattr(self.args, "requires_sq_flow") and self.args.requires_sq_flow:
            for ii in range(len(f12s)):
                f12s[ii] = F.interpolate(f12s[ii], size=(H, W), mode="bilinear")
                f12s[ii][:, :1] = f12s[ii][:, :1].clone() / (1.0 * W8) * W
                f12s[ii][:, 1:] = f12s[ii][:, 1:].clone() / (1.0 * H8) * H
            if self.training:
                return f12s
            else:
                return [f12s[-1]], f12_init
        else:
            f12[:, :1] = f12[:, :1].clone() / (1.0 * W8) * W
            f12[:, 1:] = f12[:, 1:].clone() / (1.0 * H8) * H

            f12 = F.interpolate(f12, size=(H, W), mode="bilinear")
            # print('wo!!')
            return (
                f12,
                f12_init,
                error21,
            )

    def forward_pred(
        self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False
    ):
        """Estimate optical flow between pair of frames"""

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(device.type, enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(device.type, enabled=self.args.mixed_precision):
            cnet = self.fnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            if itr == 0:
                if flow_init is not None:
                    coords1 = coords1 + flow_init
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(device.type, enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        return flow_predictions, flow_up, flow_init

####################### WARPING #######################


# expects batched tensors, considered low-level operation
# img: bs, ch, h, w
# flow: bs, xy (pix displace), h, w
def flow_backwarp(
    img, flow, resample="bilinear", padding_mode="border", align_corners=False
):
    if len(img.shape) != 4:
        img = img[None,]
    if len(flow.shape) != 4:
        flow = flow[None,]
    q = (
        2
        * flow
        / torch.tensor(
            [
                flow.shape[-2],
                flow.shape[-1],
            ],
            device=flow.device,
            dtype=torch.float,
        )[None, :, None, None]
    )
    q = q + torch.stack(
        torch.meshgrid(
            torch.linspace(-1, 1, flow.shape[-2]),
            torch.linspace(-1, 1, flow.shape[-1]),
        )
    )[
        None,
    ].to(
        flow.device
    )
    if img.dtype != q.dtype:
        img = img.type(q.dtype)

    return nn.functional.grid_sample(
        img,
        q.flip(dims=(1,)).permute(0, 2, 3, 1),
        mode=resample,  # nearest, bicubic, bilinear
        padding_mode=padding_mode,  # border, zeros, reflection
        align_corners=align_corners,
    )


backwarp = flow_warp = flow_backwarp


# mode: sum, avg, lin, softmax
# lin/softmax w/out metric defaults to avg
# must use gpu, move back to cpu if retain_device
# typical metric: -20 * | img0 - backwarp(img1,flow) |
# From Fannovel16: Changed mode params for common ops.
def flow_forewarp(
    img, flow, mode="average", metric=None, mask=False, retain_device=True
):
    # setup
    #if mode == "sum":
    #    mode = "summation"
    #elif mode == "avg":
    #    mode = "average"
    if mode in ["lin", "linear"]:
        #mode = "linear" if metric is not None else "average"
        mode = "linear" if metric is not None else "avg"
    elif mode in ["sm", "softmax"]:
        #mode = "softmax" if metric is not None else "average"
        mode = "soft" if metric is not None else "avg"
    if len(img.shape) != 4:
        img = img[None,]
    if len(flow.shape) != 4:
        flow = flow[None,]
    if metric is not None and len(metric.shape) != 4:
        metric = metric[None,]
    flow = flow.flip(dims=(1,))
    if img.dtype != torch.float32:
        img = img.type(torch.float32)
    if flow.dtype != torch.float32:
        flow = flow.type(torch.float32)
    if metric is not None and metric.dtype != torch.float32:
        metric = metric.type(torch.float32)

    # move to gpu if necessary
    assert img.device == flow.device
    if metric is not None:
        assert img.device == metric.device
    was_cpu = img.device.type == "cpu"
    if was_cpu:
        img = img.to("cuda")
        flow = flow.to("cuda")
        if metric is not None:
            metric = metric.to("cuda")

    # add mask
    if mask:
        bs, ch, h, w = img.shape
        img = torch.cat(
            [img, torch.ones(bs, 1, h, w, dtype=img.dtype, device=img.device)], dim=1
        )

    # forward, move back to cpu if desired
    ans = FunctionSoftsplat(img, flow, metric, mode)
    if was_cpu and retain_device:
        ans = ans.cpu()
    return ans


forewarp = flow_forewarp


# resizing utility
def flow_resize(flow, size, mode="nearest", align_corners=False):
    # flow: bs,xy,h,w
    size = pixel_ij(size, rounding=True)
    if flow.dtype != torch.float:
        flow = flow.float()
    if len(flow.shape) == 3:
        flow = flow[None,]
    if flow.shape[-2:] == size:
        return flow
    return (
        nn.functional.interpolate(
            flow,
            size=size,
            mode=mode,
            align_corners=align_corners if mode != "nearest" else None,
        )
        * torch.tensor(
            [b / a for a, b in zip(flow.shape[-2:], size)],
            device=flow.device,
        )[None, :, None, None]
    )


####################### TRADITIONAL #######################

# dense
_lucaskanade = lambda a, b: np.moveaxis(
    cv2.optflow.calcOpticalFlowSparseToDense(
        a,
        b,  # grid_step=5, sigma=0.5,
    ),
    2,
    0,
)[
    None,
]
_farneback = lambda a, b: np.moveaxis(
    cv2.calcOpticalFlowFarneback(
        a,
        b,
        None,
        0.6,
        3,
        25,
        7,
        5,
        1.2,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ),
    2,
    0,
)[
    None,
]
_dtvl1_ = cv2.optflow.createOptFlow_DualTVL1()
_dtvl1 = lambda a, b: np.moveaxis(
    _dtvl1_.calc(
        a,
        b,
        None,
    ),
    2,
    0,
)[
    None,
]
_simple = lambda a, b: np.moveaxis(
    cv2.optflow.calcOpticalFlowSF(
        a,
        b,
        3,
        5,
        5,
    ),
    2,
    0,
)[
    None,
]
_pca_ = cv2.optflow.createOptFlow_PCAFlow()
_pca = lambda a, b: np.moveaxis(
    _pca_.calc(
        a,
        b,
        None,
    ),
    2,
    0,
)[
    None,
]
_drlof = lambda a, b: np.moveaxis(
    cv2.optflow.calcOpticalFlowDenseRLOF(
        a,
        b,
        None,
    ),
    2,
    0,
)[
    None,
]
_deepflow_ = cv2.optflow.createOptFlow_DeepFlow()
_deepflow = lambda a, b: np.moveaxis(
    _deepflow_.calc(
        a,
        b,
        None,
    ),
    2,
    0,
)[
    None,
]


def cv2flow(a, b, method="lucaskanade", back=False):
    if method == "lucaskanade":
        f = _lucaskanade
        a = a.convert("L").cv2()
        b = b.convert("L").cv2()
    elif method == "farneback":
        f = _farneback
        a = a.convert("L").cv2()
        b = b.convert("L").cv2()
    elif method == "dtvl1":
        f = _dtvl1
        a = a.convert("L").cv2()
        b = b.convert("L").cv2()
    elif method == "simple":
        f = _simple
        a = a.convert("RGB").cv2()
        b = b.convert("RGB").cv2()
    elif method == "pca":
        f = _pca
        a = a.convert("L").cv2()
        b = b.convert("L").cv2()
    elif method == "drlof":
        f = _drlof
        a = a.convert("RGB").cv2()
        b = b.convert("RGB").cv2()
    elif method == "deepflow":
        f = _deepflow
        a = a.convert("L").cv2()
        b = b.convert("L").cv2()
    else:
        assert 0
    ans = f(b, a)
    if back:
        ans = np.concatenate(
            [
                ans,
                f(a, b),
            ]
        )
    return torch.tensor(ans).flip(dims=(1,))


####################### FLOWNET2 #######################


def flownet2(img_a, img_b, mode="shm", back=False):
    # package
    url = f"http://localhost:8109/get-flow"
    if mode == "shm":
        t = time.time()
        fn_a = img_a.save(mkfile(f"/dev/shm/_flownet2/{t}/img_a.png"))
        fn_b = img_b.save(mkfile(f"/dev/shm/_flownet2/{t}/img_b.png"))
    elif mode == "net":
        assert False, "not impl"
        q = u2d.img2uri(img.pil("RGB"))
        q.decode()
    resp = requests.get(
        url,
        params={
            "img_a": fn_a,
            "img_b": fn_b,
            "mode": mode,
            "back": back,
            # 'vis': vis,
        },
    )

    # return
    ans = {"response": resp}
    if resp.status_code == 200:
        j = resp.json()
        ans["time"] = j["time"]
        ans["output"] = {
            "flow": torch.tensor(load(j["fn_flow"])),
        }
        # if vis:
        #     ans['output']['vis'] = I(j['fn_vis'])
    if mode == "shm":
        shutil.rmtree(f"/dev/shm/_flownet2/{t}")
    return ans


####################### VISUALIZATION #######################


class Gridnet(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2, total_dropout_p, depth):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.total_dropout_p = p = total_dropout_p
        self.depth = depth
        self.encoders = nn.ModuleList(
            [GridnetEncoder(ch0, ch1, ch2) for i in range(self.depth)]
        )
        self.decoders = nn.ModuleList(
            [GridnetDecoder(ch0, ch1, ch2) for i in range(self.depth)]
        )
        self.total_dropout = GridnetTotalDropout(p)
        return

    def forward(self, x):
        for e, enc in enumerate(self.encoders):
            t = [self.total_dropout(i) for i in t] if e != 0 else x
            t = enc(t)
        for d, dec in enumerate(self.decoders):
            t = [self.total_dropout(i) for i in t]
            t = dec(t)
        return t


class GridnetEncoder(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.resnet_0 = GridnetResnet(ch0)
        self.resnet_1 = GridnetResnet(ch1)
        self.resnet_2 = GridnetResnet(ch2)
        self.downsample_01 = GridnetDownsample(ch0, ch1)
        self.downsample_12 = GridnetDownsample(ch1, ch2)
        return

    def forward(self, x):
        out = [
            None,
        ] * 3
        out[0] = self.resnet_0(x[0])
        out[1] = self.resnet_1(x[1]) + self.downsample_01(out[0])
        out[2] = self.resnet_2(x[2]) + self.downsample_12(out[1])
        return out


class GridnetDecoder(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.resnet_0 = GridnetResnet(ch0)
        self.resnet_1 = GridnetResnet(ch1)
        self.resnet_2 = GridnetResnet(ch2)
        self.upsample_10 = GridnetUpsample(ch1, ch0)
        self.upsample_21 = GridnetUpsample(ch2, ch1)
        return

    def forward(self, x):
        out = [
            None,
        ] * 3
        out[2] = self.resnet_2(x[2])
        out[1] = self.resnet_1(x[1]) + self.upsample_21(out[2])
        out[0] = self.resnet_0(x[0]) + self.upsample_10(out[1])
        return out


class GridnetConverter(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = cin = channels_in
        self.channels_out = cout = channels_out
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.PReLU(a),
                    nn.Conv2d(a, b, kernel_size=1, padding=0),
                    nn.BatchNorm2d(b),
                )
                for a, b in zip(cin, cout)
            ]
        )
        return

    def forward(self, x):
        return [m(q) for m, q in zip(self.nets, x)]


class GridnetResnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = ch = channels
        self.net = nn.Sequential(
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
        )
        return

    def forward(self, x):
        return x + self.net(x)


class GridnetDownsample(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = chin = channels_in
        self.channels_out = chout = channels_out
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, chin, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(chin),
            nn.PReLU(chin),
            nn.Conv2d(chin, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
        )
        return

    def forward(self, x):
        return self.net(x)


class GridnetUpsample(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = chin = channels_in
        self.channels_out = chout = channels_out
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.PReLU(chin),
            nn.Conv2d(chin, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
            nn.PReLU(chout),
            nn.Conv2d(chout, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
        )
        return

    def forward(self, x):
        return self.net(x)


class GridnetTotalDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.weight = 1 / (1 - p)
        return

    def get_drop(self, x):
        d = torch.rand(len(x))[:, None, None, None] < self.p
        d = (1 - d.float()).to(x.device) * self.weight
        return d

    def forward(self, x, force_drop=None):
        if force_drop is True:
            ans = x * self.get_drop(x)
        elif force_drop is False:
            ans = x
        else:
            if self.training:
                ans = x * self.get_drop(x)
            else:
                ans = x
        return ans


class Interpolator(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.size = size
        self.mode = mode
        return

    def forward(self, x, is_flow=False):
        if x.shape[-2] == self.size:
            return x
        if len(x.shape) == 4:
            # bs,ch,h,w
            bs, ch, h, w = x.shape
            ans = nn.functional.interpolate(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=(False, None)[self.mode == "nearest"],
            )
            if is_flow:
                ans = (
                    ans
                    * torch.tensor(
                        [b / a for a, b in zip((h, w), self.size)],
                        device=ans.device,
                    )[None, :, None, None]
                )
            return ans
        elif len(x.shape) == 5:
            # bs,k,ch,h,w (merge bs and k)
            bs, k, ch, h, w = x.shape
            return self.forward(
                x.view(bs * k, ch, h, w),
                is_flow=is_flow,
            ).view(bs, k, ch, *self.size)
        else:
            assert 0


###################### CANNY ######################


def canny(img, a=100, b=200):
    img = I(img).convert("L")
    return I(cv2.Canny(img.cv2(), a, b))


# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def canny_pis(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    img = I(img).convert("L").uint8(ch_last=False)
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(img[0], lower, upper)
    # return the edged image
    return I(edged)


# https://en.wikipedia.org/wiki/Otsu%27s_method
def canny_otsu(img):
    img = I(img).convert("L").uint8(ch_last=False)
    high, _ = cv2.threshold(img[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = 0.5 * high
    return I(cv2.Canny(img[0], low, high))


def xdog(img, t=1.0, epsilon=0.04, phi=100, sigma=3, k=1.6):
    img = I(img).convert("L").uint8(ch_last=False)
    grey = np.asarray(img, dtype=np.float32)
    g0 = scipy.ndimage.gaussian_filter(grey, sigma)
    g1 = scipy.ndimage.gaussian_filter(grey, sigma * k)

    # ans = ((1+p) * g0 - p * g1) / 255
    ans = (g0 - t * g1) / 255
    ans = 1 + np.tanh(phi * (ans - epsilon)) * (ans < epsilon)
    return ans


def dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    img = I(img).convert("L").tensor()[None]
    kern0 = max(2 * int(sigma * kernel_factor) + 1, 3)
    kern1 = max(2 * int(sigma * k * kernel_factor) + 1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img,
        (kern0, kern0),
        (sigma, sigma),
        border_type="replicate",
    )
    g1 = kornia.filters.gaussian_blur2d(
        img,
        (kern1, kern1),
        (sigma * k, sigma * k),
        border_type="replicate",
    )
    ans = 0.5 + t * (g1 - g0) - epsilon
    ans = ans.clip(0, 1) if clip else ans
    return ans[0].numpy()


# input: (bs,rgb(a),h,w) or (bs,1,h,w)
# returns: (bs,1,h,w)
def batch_dog(img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
    # to grayscale if needed
    bs, ch, h, w = img.shape
    if ch in [3, 4]:
        img = kornia.color.rgb_to_grayscale(img[:, :3])
    else:
        assert ch == 1

    # calculate dog
    kern0 = max(2 * int(sigma * kernel_factor) + 1, 3)
    kern1 = max(2 * int(sigma * k * kernel_factor) + 1, 3)
    g0 = kornia.filters.gaussian_blur2d(
        img,
        (kern0, kern0),
        (sigma, sigma),
        border_type="replicate",
    )
    g1 = kornia.filters.gaussian_blur2d(
        img,
        (kern1, kern1),
        (sigma * k, sigma * k),
        border_type="replicate",
    )
    ans = 0.5 + t * (g1 - g0) - epsilon
    ans = ans.clip(0, 1) if clip else ans
    return ans


############### DERIVED DISTANCES ###############

# input: (bs,h,w) or (bs,1,h,w)
# returns: (bs,)
# normalized s.t. metric is same across proportional image scales


# average of two asymmetric distances
# normalized by diameter and area
def batch_chamfer_distance(gt, pred, block=1024, return_more=False):
    t = batch_chamfer_distance_t(gt, pred, block=block)
    p = batch_chamfer_distance_p(gt, pred, block=block)
    cd = (t + p) / 2
    return cd


def batch_chamfer_distance_t(gt, pred, block=1024, return_more=False):
    assert gt.device == pred.device and gt.shape == pred.shape
    bs, h, w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dpred = batch_edt(pred, block=block)
    cd = (gt * dpred).float().mean((-2, -1)) / np.sqrt(h**2 + w**2)
    if len(cd.shape) == 2:
        assert cd.shape[1] == 1
        cd = cd.squeeze(1)
    return cd


def batch_chamfer_distance_p(gt, pred, block=1024, return_more=False):
    assert gt.device == pred.device and gt.shape == pred.shape
    bs, h, w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    cd = (pred * dgt).float().mean((-2, -1)) / np.sqrt(h**2 + w**2)
    if len(cd.shape) == 2:
        assert cd.shape[1] == 1
        cd = cd.squeeze(1)
    return cd


# normalized by diameter
# always between [0,1]
def batch_hausdorff_distance(gt, pred, block=1024, return_more=False):
    assert gt.device == pred.device and gt.shape == pred.shape
    bs, h, w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    dpred = batch_edt(pred, block=block)
    hd = torch.stack(
        [
            (dgt * pred).amax(dim=(-2, -1)),
            (dpred * gt).amax(dim=(-2, -1)),
        ]
    ).amax(dim=0).float() / np.sqrt(h**2 + w**2)
    if len(hd.shape) == 2:
        assert hd.shape[1] == 1
        hd = hd.squeeze(1)
    return hd


#################### UTILITIES ####################


def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    return model


def channel_squeeze(x, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim + 2 :]
    return x.reshape(*a, -1, *b)


def channel_unsqueeze(x, shape, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim + 1 :]
    return x.reshape(*a, *shape, *b)


def default_collate(items, device=None):
    return to(dict(torch.utils.data.dataloader.default_collate(items)), device)


def to(x, device):
    if device is None:
        return x
    if issubclass(x.__class__, dict):
        return dict(
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in x.items()
            }
        )
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    assert 0, "data not understood"


################ PARSING ################

from argparse import Namespace

# args:  all args
# bargs: base args
# pargs: data processing args
# largs: data loading args
# margs: model args
# targs: training args


# typically used to read dataset filters
def read_filter(fn, cast=None, sort=True, sort_key=None):
    if cast is None:
        cast = lambda x: x
    ans = [cast(line) for line in read(fn).split("\n") if line != ""]
    if sort:
        return sorted(ans, key=sort_key)
    else:
        return ans


################ FILE MANAGEMENT ################


def mkfile(fn, parents=True, exist_ok=True):
    dn = "/".join(fn.split("/")[:-1])
    mkdir(dn, parents=parents, exist_ok=exist_ok)
    return fn


def mkdir(dn, parents=True, exist_ok=True):
    pathlib.Path(dn).mkdir(parents=parents, exist_ok=exist_ok)
    return dn if (not dn[-1] == "/" or dn == "/") else dn[:-1]


def fstrip(fn, return_more=False):
    dspl = fn.split("/")
    dn = "/".join(dspl[:-1]) if len(dspl) > 1 else "."
    fn = dspl[-1]
    fspl = fn.split(".")
    if len(fspl) == 1:
        bn = fspl[0]
        ext = ""
    else:
        bn = ".".join(fspl[:-1])
        ext = fspl[-1]
    if return_more:
        return Namespace(
            dn=dn,
            fn=fn,
            path=f"{dn}/{fn}",
            bn_path=f"{dn}/{bn}",
            bn=bn,
            ext=ext,
        )
    else:
        return bn


def read(fn, mode="r"):
    with open(fn, mode) as handle:
        return handle.read()


def write(text, fn, mode="w"):
    mkfile(fn, parents=True, exist_ok=True)
    with open(fn, mode) as handle:
        return handle.write(text)


import pickle


def dump(obj, fn, mode="wb"):
    mkfile(fn, parents=True, exist_ok=True)
    with open(fn, mode) as handle:
        return pickle.dump(obj, handle)


def load(fn, mode="rb"):
    with open(fn, mode) as handle:
        return pickle.load(handle)


import json


def jwrite(x, fn, mode="w", indent="\t", sort_keys=False):
    mkfile(fn, parents=True, exist_ok=True)
    with open(fn, mode) as handle:
        return json.dump(x, handle, indent=indent, sort_keys=sort_keys)


def jread(fn, mode="r"):
    with open(fn, mode) as handle:
        return json.load(handle)


try:
    import yaml

    def ywrite(x, fn, mode="w", default_flow_style=False):
        mkfile(fn, parents=True, exist_ok=True)
        with open(fn, mode) as handle:
            return yaml.dump(x, handle, default_flow_style=default_flow_style)

    def yread(fn, mode="r"):
        with open(fn, mode) as handle:
            return yaml.safe_load(handle)

except:
    pass

try:
    import pyunpack
except:
    pass

try:
    import mysql
    import mysql.connector
except:
    pass


################ MISC ################

hakase = "./env/__hakase__.jpg"
if not os.path.isfile(hakase):
    hakase = "./__env__/__hakase__.jpg"


def mem(units="m"):
    return (
        psProcess(os.getpid()).memory_info().rss
        / {
            "b": 1,
            "k": 1e3,
            "m": 1e6,
            "g": 1e9,
            "t": 1e12,
        }[units[0].lower()]
    )


def chunk(array, length, colwise=True):
    if colwise:
        return [array[i : i + length] for i in range(0, len(array), length)]
    else:
        return chunk(array, int(math.ceil(len(array) / length)), colwise=True)


def classtree(x):
    return inspect.getclasstree(inspect.getmro(x))


################ AESTHETIC ################


class Table:
    def __init__(
        self,
        table,
        delimiter=" ",
        orientation="br",
        double_colon=True,
    ):
        self.delimiter = delimiter
        self.orientation = orientation
        self.t = Table.parse(table, delimiter, orientation, double_colon)
        return

    # rendering
    def __str__(self):
        return self.render()

    def __repr__(self):
        return self.render()

    def render(self):
        # set up empty entry
        empty = ("", Table._spec(self.orientation, transpose=False))

        # calculate table size
        t = copy.deepcopy(self.t)
        totalrows = len(t)
        totalcols = [len(r) for r in t]
        assert min(totalcols) == max(totalcols)
        totalcols = totalcols[0]

        # string-ify
        for i in range(totalrows):
            for j in range(totalcols):
                x, s = t[i][j]
                sp = s[11]
                if sp:
                    x = eval(f'f"{{{x}{sp}}}"')
                Table._put((str(x), s), t, (i, j), empty)

        # expand delimiters
        _repl = (
            lambda s: s[:2] + (1, 0, 0, 0, 0) + s[7:10] + (1,) + s[11:]
            if s[2]
            else s[:2] + (0, 0, 0, 0, 0) + s[7:10] + (1,) + s[11:]
        )
        for i, row in enumerate(t):
            for j, (x, s_own) in enumerate(row):
                # expand delim_up(^)
                if s_own[3]:
                    u, v = i, j
                    while 0 <= u:
                        _, s = t[u][v]
                        if (i, j) != (u, v) and (s[2] and not s[10]):
                            break
                        Table._put((x, _repl(s)), t, (u, v), empty)
                        u -= 1

                # expand delim_down(v)
                if s_own[4]:
                    u, v = i, j
                    while u < totalrows:
                        _, s = t[u][v]
                        if (i, j) != (u, v) and (s[2] and not s[10]):
                            break
                        Table._put((x, _repl(s)), t, (u, v), empty)
                        u += 1

                # expand delim_right(>)
                if s_own[5]:
                    u, v = i, j
                    while v < totalcols:
                        _, s = t[u][v]
                        if (i, j) != (u, v) and (s[2] and not s[10]):
                            break
                        Table._put((x, _repl(s)), t, (u, v), empty)
                        v += 1

                # expand delim_left(<)
                if s_own[6]:
                    u, v = i, j
                    while 0 <= v:
                        _, s = t[u][v]
                        if (i, j) != (u, v) and (s[2] and not s[10]):
                            break
                        Table._put((x, _repl(s)), t, (u, v), empty)
                        v -= 1

        # justification calculation
        widths = [
            0,
        ] * totalcols  # j
        heights = [
            0,
        ] * totalrows  # i
        for i, row in enumerate(t):
            for j, (x, s) in enumerate(row):
                # height caclulation
                heights[i] = max(heights[i], x.count("\n"))

                # width calculation; non-delim fillers no contribution
                if s[2] or not s[10]:
                    w = max(len(q) for q in x.split("\n"))
                    widths[j] = max(widths[j], w)
        # no newline ==> height=1
        heights = [h + 1 for h in heights]

        # render table
        rend = []
        roff = 0
        for i, row in enumerate(t):
            for j, (x, s) in enumerate(row):
                w, h = widths[j], heights[i]

                # expand fillers and delimiters
                if s[2] or s[10]:
                    xs = x.split("\n")
                    xw0 = min(len(l) for l in xs)
                    xw1 = max(len(l) for l in xs)
                    xh = len(xs)
                    if (xw0 == xw1 == w) and (xh == h):
                        pass
                    elif xw0 == xw1 == w:
                        x = "\n".join(
                            [
                                xs[0],
                            ]
                            * h
                        )
                    elif xh == h:
                        x = "\n".join([(l[0] if l else "") * w for l in xs])
                    else:
                        x = x[0] if x else " "
                        x = "\n".join(
                            [
                                x * w,
                            ]
                            * h
                        )

                # justify horizontally
                x = [l.rjust(w) if s[0] else l.ljust(w) for l in x.split("\n")]

                # justify vertically
                plus = [
                    " " * w,
                ] * (h - len(x))
                x = plus + x if not s[1] else x + plus

                # input to table
                for r, xline in enumerate(x):
                    Table._put(xline, rend, (roff + r, j), None)
            roff += h

        # return rendered string
        return "\n".join(["".join(r) for r in rend])

    # parsing
    def _spec(s, transpose=False):
        if ":" in s:
            i = s.index(":")
            sp = s[i:]
            s = s[:i]
        else:
            sp = ""
            s = s.lower()
        return (
            int("r" in s),  #  0:: 0:left(l)   1:right(r)
            int("t" in s),  #  1:: 0:bottom(b) 1:top(t)
            int(any([i in s for i in [".", "<", ">", "^", "v"]])),  #  2:: delim_here(.)
            int("^" in s if not transpose else "<" in s),  #  3:: delim_up(^)
            int("v" in s if not transpose else ">" in s),  #  4:: delim_down(v)
            int(">" in s if not transpose else "v" in s),  #  5:: delim_right(>)
            int("<" in s if not transpose else "^" in s),  #  6:: delim_left(<)
            int("+" in s),  #  7:: subtable(+)
            int("-" in s if not transpose else "|" in s),  #  8:: subtable_horiz(-)
            int("|" in s if not transpose else "-" in s),  #  9:: subtable_vert(|)
            int("_" in s),  # 10:: fill(_); if delim, overwrite; else fit
            sp,  # 11:: special(:) f-string for numbers
        )

    def _put(obj, t, ij, empty):
        i, j = ij
        while i >= len(t):
            t.append([])
        while j >= len(t[i]):
            t[i].append(empty)
        t[i][j] = obj
        return

    def parse(
        table,
        delimiter=" ",
        orientation="br",
        double_colon=True,
    ):
        # disabling transpose
        transpose = False

        # set up empty entry
        empty = ("", Table._spec(orientation, transpose))

        # transpose
        t = []
        for i, row in enumerate(table):
            for j, item in enumerate(row):
                ij = (i, j) if not transpose else (j, i)
                if type(item) == tuple and len(item) == 2 and type(item[1]) == str:
                    item = (item[0], Table._spec(item[1], transpose))
                elif double_colon and type(item) == str and "::" in item:
                    x, s = item.split("::")
                    item = (x, Table._spec(s, transpose))
                else:
                    item = (item, Table._spec(orientation, transpose))
                Table._put(item, t, ij, empty)

        # normalization
        maxcol = 0
        maxrow = len(t)
        for i, row in enumerate(t):
            # take element number into account
            maxcol = max(maxcol, len([i for i in row if not i[1][2]]))

            # take subtables into account
            for j, (x, s) in enumerate(row):
                if s[7]:
                    r = len(x)
                    maxrow = max(maxrow, i + r)
                    c = max(len(q) for q in x)
                    maxcol = max(maxcol, j + c)
                elif s[8]:
                    c = len(x)
                    maxcol = max(maxcol, j + c)
                elif s[9]:
                    r = len(x)
                    maxrow = max(maxrow, i + r)
        totalcols = 2 * maxcol + 1
        totalrows = maxrow
        t += [[]] * (totalrows - len(t))
        newt = []
        delim = (delimiter, Table._spec("._" + orientation, transpose))
        for i, row in enumerate(t):
            wasd = False
            tcount = 0
            for j in range(totalcols):
                item = t[i][tcount] if tcount < len(t[i]) else empty
                isd = item[1][2]
                if wasd and isd:
                    Table._put(empty, newt, (i, j), empty)
                    wasd = False
                elif wasd and not isd:
                    Table._put(item, newt, (i, j), empty)
                    tcount += 1
                    wasd = False
                elif not wasd and isd:
                    Table._put(item, newt, (i, j), empty)
                    tcount += 1
                    wasd = True
                elif not wasd and not isd:
                    Table._put(delim, newt, (i, j), empty)
                    wasd = True
        t = newt

        # normalization: add dummy last column for delimiter
        for row in t:
            row.append(empty)

        # expand subtables
        delim_cols = [i for i in range(totalcols) if i % 2 == 0]
        while True:
            # find a table
            ij = None
            for i, row in enumerate(t):
                for j, item in enumerate(row):
                    st, s = item
                    if s[7]:
                        ij = i, j, 7, st, s
                        break
                    elif s[8]:
                        ij = i, j, 8, st, s
                        break
                    elif s[9]:
                        ij = i, j, 9, st, s
                        break
                if ij is not None:
                    break
            if ij is None:
                break

            # replace its specs
            i, j, k, st, s = ij
            s = list(s)
            s[7] = s[8] = s[9] = 0
            s = tuple(s)

            # expand it
            if k == 7:  # 2d table
                for x, row in enumerate(st):
                    for y, obj in enumerate(row):
                        a = i + x if not transpose else i + y
                        b = j + 2 * y if not transpose else j + 2 * x
                        Table._put((obj, s), t, (a, b), None)
            if k == 8:  # subtable_horiz
                for y, obj in enumerate(st):
                    Table._put((obj, s), t, (i, j + 2 * y), None)
            if k == 9:  # subtable_vert
                for x, obj in enumerate(st):
                    Table._put((obj, s), t, (i + x, j), None)

        # return, finally
        return t


class Resnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = ch = channels
        self.net = nn.Sequential(
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
        )
        return

    def forward(self, x):
        return x + self.net(x)


class Synthesizer(nn.Module):
    def __init__(
        self, size, channels_image, channels_flow, channels_mask, channels_feature
    ):
        super().__init__()
        self.size = size
        self.diam = diam(self.size)
        self.channels_image = cimg = channels_image
        self.channels_flow = cflow = channels_flow
        self.channels_mask = cmask = channels_mask
        self.channels_feature = cfeat = channels_feature
        self.channels = ch = cimg + cflow // 2 + cmask + cfeat
        self.interpolator = Interpolator(self.size, mode="bilinear")
        self.net = nn.Sequential(
            nn.Conv2d(ch + 3, 64, kernel_size=1, padding=0),
            Resnet(64),
            nn.Sequential(
                nn.PReLU(64),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
            ),
            Resnet(32),
            nn.Sequential(
                nn.PReLU(32),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
            ),
            Resnet(16),
            nn.Sequential(
                nn.PReLU(16),
                nn.Conv2d(16, 3, kernel_size=3, padding=1),
            ),
        )
        return

    def forward(self, images, flows, masks, features, return_more=False):
        itp = self.interpolator
        images = [
            (images[0] + images[1]) / 2,
        ] + images
        logimgs = [itp(pixel_logit(i[:, :3])) for i in images]
        cat = torch.cat(
            [
                *logimgs,
                *[itp(f).norm(dim=1, keepdim=True) / self.diam for f in flows],
                *[itp(m) for m in masks],
                *[itp(f) for f in features],
            ],
            dim=1,
        )
        residual = self.net(cat)
        return torch.sigmoid(logimgs[0] + 0.5 * residual), (
            locals() if return_more else None
        )


class FlowZMetric(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, img0, img1, flow0, flow1, return_more=False):
        # B(i0,f0) = i1
        # B(i1,f1) = i0
        # F(x,f0,z0)
        # F(x,f1,z1)
        img0 = kornia.color.rgb_to_lab(img0[:, :3])
        img1 = kornia.color.rgb_to_lab(img1[:, :3])
        return [
            -0.1 * (img1 - flow_backwarp(img0, flow0)).norm(dim=1, keepdim=True),  # z0
            -0.1 * (img0 - flow_backwarp(img1, flow1)).norm(dim=1, keepdim=True),  # z1
        ], (locals() if return_more else None)


class NEDT(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(
        self,
        img,
        t=2.0,
        sigma_factor=1 / 540,
        k=1.6,
        epsilon=0.01,
        kernel_factor=4,
        exp_factor=540 / 15,
        return_more=False,
    ):
        with torch.no_grad():
            dog = batch_dog(
                img,
                t=t,
                sigma=img.shape[-2] * sigma_factor,
                k=k,
                epsilon=epsilon,
                kernel_factor=kernel_factor,
                clip=False,
            )
            edt = batch_edt((dog > 0.5).float())
            ans = 1 - (-edt * exp_factor / max(edt.shape[-2:])).exp()
        return ans, (locals() if return_more else None)


class HalfWarper(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_image = 4 * 3
        self.channels_flow = 2 * 2
        self.channels_mask = 2 * 1
        self.channels = self.channels_image + self.channels_flow + self.channels_mask

    def morph_open(self, x, k):
        if k == 0:
            return x
        else:
            with torch.no_grad():
                return kornia.morphology.opening(x, torch.ones(k, k, device=x.device))

    def forward(self, img0, img1, flow0, flow1, z0, z1, k, t=0.5, return_more=False):
        # forewarps
        flow0_ = (1 - t) * flow0
        flow1_ = t * flow1
        f01 = forewarp(img0, flow1_, mode="sm", metric=z1, mask=True)
        f10 = forewarp(img1, flow0_, mode="sm", metric=z0, mask=True)
        f01i, f01m = f01[:, :-1], self.morph_open(f01[:, -1:], k=k)
        f10i, f10m = f10[:, :-1], self.morph_open(f10[:, -1:], k=k)

        # base guess
        base0 = f01m * f01i + (1 - f01m) * f10i
        base1 = f10m * f10i + (1 - f10m) * f01i
        ans = [
            [  # images
                base0,
                base1,
                f01i,
                f10i,
            ],
            [  # flows
                flow0_,
                flow1_,
            ],
            [  # masks
                f01m,
                f10m,
            ],
        ]
        return ans, (locals() if return_more else None)


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, inferserve_query, size_in=None):
        super().__init__()
        self.inferserve_query = iq = inferserve_query
        self.size_in = si = size_in
        if iq[0] == "torchvision":
            # use pytorch pretrained resnet50
            self.base_hparams = None
            resnet = tv.models.resnet50(pretrained=True)

            self.resize = T.Resize(256)
            self.resnet_preprocess = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu  #   64ch, 128p (assuming 256p input)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  #  256ch,  64p
            self.layer2 = resnet.layer2  #  512ch,  32p
        else:
            base = userving.infer_model_load(*iq).eval()
            self.base_hparams = base.hparams

            self.resize = T.Resize(base.hparams.largs.size)
            self.resnet_preprocess = base.resnet_preprocess
            self.conv1 = base.resnet.conv1
            self.bn1 = base.resnet.bn1
            self.relu = base.resnet.relu  #   64ch, 128p (assuming 256p input)
            self.maxpool = base.resnet.maxpool
            self.layer1 = base.resnet.layer1  #  256ch,  64p
            self.layer2 = base.resnet.layer2  #  512ch,  32p
        if self.size_in is None:
            self.sizes_out = None
        else:
            s = self.resize.size
            self.sizes_out = [
                pixel_ij(
                    rescale_dry(si, (s // 2) / si[0]), rounding="ceil"
                ),  # conv1, 128p
                pixel_ij(
                    rescale_dry(si, (s // 4) / si[0]), rounding="ceil"
                ),  # layer1, 64p
                pixel_ij(
                    rescale_dry(si, (s // 8) / si[0]), rounding="ceil"
                ),  # layer2, 32p
            ]
        self.channels = [
            64,
            256,
            512,
        ]
        return

    def forward(self, x, force_sizes_out=False, return_more=False):
        ans = []
        x = x[:, :3]
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans.append(x)  # conv1
        x = self.maxpool(x)
        x = self.layer1(x)
        ans.append(x)  # layer1
        x = self.layer2(x)
        ans.append(x)  # layer2
        if force_sizes_out or (self.sizes_out is None):
            self.sizes_out = [tuple(q.shape[-2:]) for q in ans]
        return ans, (locals() if return_more else None)


class NetNedt(nn.Module):
    def __init__(self):
        super().__init__()
        chin = 3 + 1 + 4 + 4 + 1 + 1
        ch = 16
        chout = 1
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, chout, kernel_size=3, padding=1),
        )
        return

    def forward(self, out_base, out_base_nedt, hw_imgs, hw_masks, return_more=False):
        cat = torch.cat(
            [
                out_base,  # 3
                out_base_nedt,  # 1
                hw_imgs[0],  # 4
                hw_imgs[1],  # 4
                hw_masks[0],  # 1
                hw_masks[1],  # 1
            ],
            dim=1,
        )
        log = pixel_logit(cat.clip(0, 1))
        ans = torch.sigmoid(self.net(log))
        return ans, (locals() if return_more else None)


class NetTail(nn.Module):
    def __init__(self):
        super().__init__()
        chin = 3 + 1 + 1
        ch = 16
        chout = 3
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, chout, kernel_size=3, padding=1),
        )
        return

    def forward(self, out_base, out_base_nedt, pred_nedt, return_more=False):
        cat = torch.cat(
            [
                out_base,  # 3
                out_base_nedt,  # 1
                pred_nedt,  # 1
            ],
            dim=1,
        )
        log = pixel_logit(cat.clip(0, 1))
        ans = torch.sigmoid(log[:, :3] + self.net(log))
        return ans, (locals() if return_more else None)


class SoftsplatLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResnetFeatureExtractor(
            ("torchvision", "resnet50"),
            (540, 960),
        )
        self.z_metric = FlowZMetric()
        self.flow_downsamplers = [
            Interpolator(s, mode="bilinear") for s in self.feature_extractor.sizes_out
        ]
        self.gridnet_converter = GridnetConverter(
            self.feature_extractor.channels,
            [32, 64, 128],
        )
        self.gridnet = Gridnet(
            *[32, 64, 128],
            total_dropout_p=0.0,
            depth=1,  # equivalent to u-net
        )
        self.nedt = NEDT()
        self.half_warper = HalfWarper()
        self.synthesizer = Synthesizer(
            (540, 960),
            self.half_warper.channels_image,
            self.half_warper.channels_flow,
            self.half_warper.channels_mask,
            self.gridnet.channels_0,
        )
        return

    def forward(self, x, t=0.5, k=5, return_more=False):
        rm = return_more
        flow0, flow1 = x["flows"].swapaxes(0, 1)
        img0, img1 = x["images"][:, 0], x["images"][:, -1]
        (z0, z1), locs_z = self.z_metric(img0, img1, flow0, flow1, return_more=rm)
        img0 = torch.cat([img0, self.nedt(img0)[0]], dim=1)
        img1 = torch.cat([img1, self.nedt(img1)[0]], dim=1)

        # images and flows
        (hw_imgs, hw_flows, hw_masks), locs_hw = self.half_warper(
            img0,
            img1,
            flow0,
            flow1,
            z0,
            z1,
            k,
            t=t,
            return_more=rm,
        )

        # features
        feats0, locs_fe0 = self.feature_extractor(img0, return_more=rm)
        feats1, locs_fe1 = self.feature_extractor(img1, return_more=rm)
        warps = []
        for ft0, ft1, ds in zip(feats0, feats1, self.flow_downsamplers):
            (w, _, _), _ = self.half_warper(
                ft0,
                ft1,
                ds(flow0, 1),
                ds(flow1, 1),
                ds(z0),
                ds(z1),
                k,
                t=t,
            )
            warps.append((w[0] + w[1]) / 2)
        feats = self.gridnet(self.gridnet_converter(warps))

        # synthesis
        pred, locs_synth = self.synthesizer(
            hw_imgs,
            hw_flows,
            hw_masks,
            [
                feats[0],
            ],
            return_more=rm,
        )
        return pred, (locals() if rm else None)


class DTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_nedt = NetNedt()
        self.net_tail = NetTail()
        self.nedt = NEDT()
        return

    def forward(self, x, out_base, locs_base, return_more=False):
        rm = return_more
        with torch.no_grad():
            out_base_nedt, locs_base_nedt = self.nedt(out_base, return_more=rm)
        hw_imgs, hw_masks = locs_base["hw_imgs"], locs_base["hw_masks"]
        pred_nedt, locs_nedt = self.net_nedt(
            out_base, out_base_nedt, hw_imgs, hw_masks, return_more=rm
        )
        pred, locs_tail = self.net_tail(
            out_base, out_base_nedt, pred_nedt.clone().detach(), return_more=rm
        )
        return torch.cat([pred, pred_nedt], dim=1), (locals() if rm else None)


class RAFT(nn.Module):
    def __init__(self, path="/workspace/tensorrt/models/anime_interp_full.ckpt"):
        super().__init__()
        self.raft = RFR(
            Namespace(
                small=False,
                mixed_precision=False,
            )
        )
        if path is not None:
            sd = torch.load(path)["model_state_dict"]
            self.raft.load_state_dict(
                {
                    k[len("module.flownet.") :]: v
                    for k, v in sd.items()
                    if k.startswith("module.flownet.")
                },
                strict=False,
            )
        return

    def forward(self, img0, img1, flow0=None, iters=12, return_more=False):
        if flow0 is not None:
            flow0 = flow0.flip(dims=(1,))
        out = self.raft(img1, img0, iters=iters, flow_init=flow0)
        return out[0].flip(dims=(1,)), (locals() if return_more else None)

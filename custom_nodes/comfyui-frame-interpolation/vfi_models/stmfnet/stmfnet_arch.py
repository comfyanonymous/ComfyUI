# https://github.com/danielism97/ST-MFNet/blob/main/models/stmfnet.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/pwcnet.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/correlation/correlation.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/gridnet.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/feature.py
# https://github.com/danielism97/ST-MFNet/blob/main/utility.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/resnet_3D.py
# https://github.com/danielism97/ST-MFNet/blob/main/cupy_module/adacof.py
# https://github.com/danielism97/ST-MFNet/blob/main/cupy_module/softsplat.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/__init__.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/pwcnet.py
# https://github.com/danielism97/ST-MFNet/blob/main/models/misc/correlation/correlation.py
from torch.nn import functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import cv2
import math
import numpy
import numpy as np
import PIL
import PIL.Image
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from vfi_models.ops import FunctionCorrelation, FunctionAdaCoF, ModuleSoftsplat
from vfi_utils import get_ckpt_container_path
import pathlib
MODEL_TYPE = pathlib.Path(__file__).parent.name

#Simple way to reduce oranges on VSCode bar
def identity(x):
    return x


def backwarp(tenInput, tenFlow):
    backwarp_tenGrid = {}
    backwarp_tenPartial = {}
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[3]),
                1.0 - (1.0 / tenFlow.shape[3]),
                tenFlow.shape[3],
            )
            .view(1, 1, 1, -1)
            .expand(-1, -1, tenFlow.shape[2], -1)
        )
        tenVer = (
            torch.linspace(
                -1.0 + (1.0 / tenFlow.shape[2]),
                1.0 - (1.0 / tenFlow.shape[2]),
                tenFlow.shape[2],
            )
            .view(1, 1, -1, 1)
            .expand(-1, -1, -1, tenFlow.shape[3])
        )

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()
    # end

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]]
        )
    # end

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)

    tenOutput = torch.nn.functional.grid_sample(
        input=tenInput,
        grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask


# end

##########################################################


class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=3,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=96,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=196,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

            # end

        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 1]
                intCurrent = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 0]

                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.netUpfeat = torch.nn.ConvTranspose2d(
                        in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][
                        intLevel + 1
                    ]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            # end

            def forward(self, tenFirst, tenSecond, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=FunctionCorrelation(
                            tenFirst=tenFirst, tenSecond=tenSecond
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious["tenFlow"])
                    tenFeat = self.netUpfeat(objPrevious["tenFeat"])

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=FunctionCorrelation(
                            tenFirst=tenFirst,
                            tenSecond=backwarp(
                                tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp
                            ),
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

                # end

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)

                return {"tenFlow": tenFlow, "tenFeat": tenFeat}

            # end

        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=2,
                        dilation=2,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=4,
                        dilation=4,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=8,
                        dilation=8,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=16,
                        dilation=16,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                )

            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)

            # end

        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict(
            {
                strKey.replace("module", "net"): tenWeight
                for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                    url="http://content.sniklaus.com/github/pytorch-pwc/network-"
                    + "default"
                    + ".pytorch",
                    model_dir=get_ckpt_container_path(MODEL_TYPE)
                ).items()
            }
        )

    # end

    def forward(self, tenFirst, tenSecond, *args):
        # optionally pass pre-extracted feature pyramid in as args
        if len(args) == 0:
            tenFirst = self.netExtractor(tenFirst)
            tenSecond = self.netExtractor(tenSecond)
        else:
            tenFirst, tenSecond = args

        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

        return objEstimate["tenFlow"] + self.netRefiner(objEstimate["tenFeat"])

    # end

    def extract_pyramid(self, tenFirst, tenSecond):
        return self.netExtractor(tenFirst), self.netExtractor(tenSecond)

    def extract_pyramid_single(self, tenFirst):
        return self.netExtractor(tenFirst)


# end

netNetwork = None

##########################################################


def estimate(tenFirst, tenSecond):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert tenFirst.shape[1] == tenSecond.shape[1]
    assert tenFirst.shape[2] == tenSecond.shape[2]

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    assert (
        intWidth == 1024
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
        intHeight == 436
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(
        input=tenPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )
    tenPreprocessedSecond = torch.nn.functional.interpolate(
        input=tenPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow = 20.0 * torch.nn.functional.interpolate(
        input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond),
        size=(intHeight, intWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


# end


class Upsampler_8tap(nn.Module):
    def __init__(self):
        super(Upsampler_8tap, self).__init__()
        filt_8tap = torch.tensor([[-1, 4, -11, 40, 40, -11, 4, -1]]).div(64)
        self.filter = nn.Parameter(filt_8tap.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, im):
        b, c, h, w = im.shape
        im_up = torch.zeros(b, c, h * 2, w * 2).to(im.device)
        im_up[:, :, ::2, ::2] = im

        p = (8 - 1) // 2
        im_up_row = F.conv2d(
            F.pad(im, pad=(p, p + 1, 0, 0), mode="reflect"), self.filter, groups=3
        )
        im_up[:, :, 0::2, 1::2] = im_up_row
        im_up_col = torch.transpose(
            F.conv2d(
                F.pad(torch.transpose(im, 2, 3), pad=(p, p + 1, 0, 0), mode="reflect"),
                self.filter,
                groups=3,
            ),
            2,
            3,
        )
        im_up[:, :, 1::2, 0::2] = im_up_col
        im_up_cross = F.conv2d(
            F.pad(im_up[:, :, 1::2, ::2], pad=(p, p + 1, 0, 0), mode="reflect"),
            self.filter,
            groups=3,
        )
        im_up[:, :, 1::2, 1::2] = im_up_cross
        return im_up

# end


model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride, temporal_stride):
        if temporal_stride:
            return (temporal_stride, stride, stride)
        else:
            return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            batchnorm(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class Conv3DNoTemporal(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return 1, stride, stride


class SEGating(nn.Module):
    def __init__(self, inplanes, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.attn_layer = nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.pool(x)
        y = self.attn_layer(out)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            batchnorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes), batchnorm(planes)
        )
        self.fg = SEGating(planes)  ## Feature Gating, from FLAVR
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.fg(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            batchnorm(planes),
            nn.ReLU(inplace=True),
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            batchnorm(planes),
            nn.ReLU(inplace=True),
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            batchnorm(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem"""

    def __init__(self, outplanes=32):
        super(BasicStem, self).__init__(
            nn.Conv3d(
                3,
                outplanes,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            batchnorm(outplanes),
            nn.ReLU(inplace=True),
        )


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution"""

    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(
                3,
                45,
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            batchnorm(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                45,
                64,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            batchnorm(64),
            nn.ReLU(inplace=True),
        )


class VideoResNet(nn.Module):
    def __init__(
        self,
        block,
        conv_makers,
        layers,
        stem,
        zero_init_residual=False,
        channels=[32, 64, 96, 128],
    ):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = channels[0]  # output channel of first stem

        self.stem = stem()

        self.layer1 = self._make_layer(
            block, conv_makers[0], channels[0], layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, conv_makers[1], channels[1], layers[1], stride=2, temporal_stride=1
        )
        self.layer3 = self._make_layer(
            block, conv_makers[2], channels[2], layers[2], stride=2, temporal_stride=1
        )
        self.layer4 = self._make_layer(
            block, conv_makers[3], channels[3], layers[3], stride=1, temporal_stride=1
        )

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        tensorConv0 = self.stem(x)
        tensorConv1 = self.layer1(tensorConv0)
        tensorConv2 = self.layer2(tensorConv1)
        tensorConv3 = self.layer3(tensorConv2)
        tensorConv4 = self.layer4(tensorConv3)
        return tensorConv0, tensorConv1, tensorConv2, tensorConv3, tensorConv4

    def _make_layer(
        self, block, conv_builder, planes, blocks, stride=1, temporal_stride=None
    ):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride, temporal_stride)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=ds_stride,
                    bias=False,
                ),
                batchnorm(planes * block.expansion),
            )
            stride = ds_stride

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress, model_dir=get_ckpt_container_path(MODEL_TYPE))
        model.load_state_dict(state_dict)
    return model


def r3d_18(bn=False, pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    global batchnorm
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity

    return _video_resnet(
        "r3d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def mc3_18(bn=False, pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    global batchnorm
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity

    return _video_resnet(
        "mc3_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def r2plus1d_18(bn=False, pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """

    global batchnorm
    if bn:
        batchnorm = nn.BatchNorm3d
    else:
        batchnorm = identity

    return _video_resnet(
        "r2plus1d_18",
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs,
    )


class upConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose"):
        super().__init__()
        self.upmode = upmode
        if self.upmode == "transpose":
            self.upconv = nn.ModuleList(
                [
                    nn.ConvTranspose3d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    ),
                    SEGating(out_ch),
                    batchnorm(out_ch),
                ]
            )
        else:
            self.upconv = nn.ModuleList(
                [
                    nn.Upsample(
                        mode="trilinear", scale_factor=(1, 2, 2), align_corners=False
                    ),
                    nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1),
                    SEGating(out_ch),
                    batchnorm(out_ch),
                ]
            )
        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):
        return self.upconv(x)


class Conv_3d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            SEGating(out_ch),
            batchnorm(out_ch),
        )

    def forward(self, x):
        return self.conv(x)


def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == "SGD":
        optimizer_function = optim.SGD
        kwargs = {"momentum": 0.9}
    elif args.optimizer == "ADAM":
        optimizer_function = optim.Adam
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    elif args.optimizer == "ADAMax":
        optimizer_function = optim.Adamax
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    elif args.optimizer == "RMSprop":
        optimizer_function = optim.RMSprop
        kwargs = {"eps": 1e-08}

    kwargs["lr"] = args.lr
    kwargs["weight_decay"] = args.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    if args.decay_type == "step":
        scheduler = lrs.StepLR(my_optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type.find("step") >= 0:
        milestones = args.decay_type.split("_")
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer, milestones=milestones, gamma=args.gamma
        )
    elif args.decay_type == "plateau":
        scheduler = lrs.ReduceLROnPlateau(
            my_optimizer,
            mode="max",
            factor=args.gamma,
            patience=args.patience,
            threshold=0.01,  # metric to be used is psnr
            threshold_mode="abs",
            verbose=True,
        )

    return scheduler


def gaussian_kernel(sz, sigma):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2)
    k = torch.exp(-1.0 / (2 * sigma**2) * k**2)
    k = k.reshape(-1, 1) * k.reshape(1, -1)
    k = k / torch.sum(k)
    return k


def moduleNormalize(frame):
    return torch.cat(
        [
            (frame[:, 0:1, :, :] - 0.4631),
            (frame[:, 1:2, :, :] - 0.4352),
            (frame[:, 2:3, :, :] - 0.3990),
        ],
        1,
    )


class FoldUnfold:
    """
    Class to handle folding tensor frame into batch of patches and back to frame again
    Thanks to Charlie Tan (charlie.tan.2019@bristol.ac.uk) for the earier version.
    """

    def __init__(self, height, width, patch_size, overlap):
        if height % 2 or width % 2 or patch_size % 2 or overlap % 2:
            print(
                "only defined for even values of height, width, patch_size size and overlap, odd values will reconstruct incorrectly"
            )
            return

        self.height = height
        self.width = width

        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def fold_to_patches(self, *frames):
        """
        args: frames -- list of (1,3,H,W) tensors
        returns: list of (B,3,h,w) image patches
        """

        # number of blocks in each direction
        n_blocks_h = (self.height // (self.stride)) + 1
        n_blocks_w = (self.width // (self.stride)) + 1

        # how much to pad each edge by
        self.pad_h = (self.stride * n_blocks_h + self.overlap - self.height) // 2
        self.pad_w = (self.stride * n_blocks_w + self.overlap - self.width) // 2
        self.height_pad = self.height + 2 * self.pad_h
        self.width_pad = self.width + 2 * self.pad_w

        # pad the frames and unfold into patches
        patches_list = []
        for i in range(len(frames)):
            padded = F.pad(
                frames[i],
                (self.pad_w, self.pad_w, self.pad_h, self.pad_h),
                mode="reflect",
            )
            unfolded = F.unfold(padded, self.patch_size, stride=self.stride)
            patches = unfolded.permute(2, 1, 0).reshape(
                -1, 3, self.patch_size, self.patch_size
            )
            patches_list.append(patches)

        return patches_list

    def unfold_to_frame(self, patches):
        """
        args: patches -- tensor of shape (B,3,h,w)
        returns: frame -- tensor of shape (1,3,H,W)
        """

        # reshape and permute back into [frames, chans * patch_size ** 2, num_patches] as expected by fold
        frame_unfold = patches.reshape(-1, 3 * self.patch_size**2, 1).permute(2, 1, 0)

        # fold into tensor of shape pad_shape
        frame_fold = F.fold(
            frame_unfold,
            (self.height_pad, self.width_pad),
            self.patch_size,
            stride=self.stride,
        )

        # unfold sums overlaps instead of averaging so tensor of ones unfolded and
        # folded to track overlaps and take mean of overlapping pixels
        ones = torch.ones_like(frame_fold)
        ones_unfold = F.unfold(ones, self.patch_size, stride=self.stride)

        # divisor is tensor of shape pad_shape where each element is the number of values that have overlapped
        # 1 = no overlaps
        divisor = F.fold(
            ones_unfold,
            (self.height_pad, self.width_pad),
            self.patch_size,
            stride=self.stride,
        )

        # divide reconstructed frame by divisor
        frame_div = frame_fold / divisor

        # crop frame to remove the padded areas
        frame_crop = frame_div[
            :, :, self.pad_h : -self.pad_h, self.pad_w : -self.pad_w
        ].clone()

        return frame_crop


def read_frame_yuv2rgb(stream, width, height, iFrame, bit_depth, pix_fmt="420"):
    if pix_fmt == "420":
        multiplier = 1
        uv_factor = 2
    elif pix_fmt == "444":
        multiplier = 2
        uv_factor = 1
    else:
        print("Pixel format {} is not supported".format(pix_fmt))
        return

    if bit_depth == 8:
        datatype = np.uint8
        stream.seek(iFrame * 1.5 * width * height * multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width * height).reshape(
            (height, width)
        )

        # read chroma samples and upsample since original is 4:2:0 sampling
        U = np.fromfile(
            stream, dtype=datatype, count=(width // uv_factor) * (height // uv_factor)
        ).reshape((height // uv_factor, width // uv_factor))
        V = np.fromfile(
            stream, dtype=datatype, count=(width // uv_factor) * (height // uv_factor)
        ).reshape((height // uv_factor, width // uv_factor))

    else:
        datatype = np.uint16
        stream.seek(iFrame * 3 * width * height * multiplier)
        Y = np.fromfile(stream, dtype=datatype, count=width * height).reshape(
            (height, width)
        )

        U = np.fromfile(
            stream, dtype=datatype, count=(width // uv_factor) * (height // uv_factor)
        ).reshape((height // uv_factor, width // uv_factor))
        V = np.fromfile(
            stream, dtype=datatype, count=(width // uv_factor) * (height // uv_factor)
        ).reshape((height // uv_factor, width // uv_factor))

    if pix_fmt == "420":
        yuv = np.empty((height * 3 // 2, width), dtype=datatype)
        yuv[0:height, :] = Y

        yuv[height : height + height // 4, :] = U.reshape(-1, width)
        yuv[height + height // 4 :, :] = V.reshape(-1, width)

        if bit_depth != 8:
            yuv = (yuv / (2**bit_depth - 1) * 255).astype(np.uint8)

        # convert to rgb
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)

    else:
        yvu = np.stack([Y, V, U], axis=2)
        if bit_depth != 8:
            yvu = (yvu / (2**bit_depth - 1) * 255).astype(np.uint8)
        rgb = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2RGB)

    return rgb


def quantize(imTensor):
    return imTensor.clamp(0.0, 1.0).mul(255).round()


def tensor2rgb(tensor):
    """
    Convert GPU Tensor to RGB image (numpy array)
    """
    out = []
    for b in range(tensor.shape[0]):
        out.append(
            np.moveaxis(quantize(tensor[b]).cpu().detach().numpy(), 0, 2).astype(
                np.uint8
            )
        )
    return np.array(out)  # (B,H,W,C)


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNextBlock(nn.Module):
    def __init__(
        self, down, cin, cout, ks, stride=1, groups=32, base_width=4, norm_layer=None
    ):
        super(ResNextBlock, self).__init__()
        if norm_layer is None or norm_layer == "batch":
            norm_layer = nn.BatchNorm2d
        elif norm_layer == "identity":
            norm_layer = Identity
        width = int(cout * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(cin, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        if down:
            self.conv2 = nn.Conv2d(
                width,
                width,
                kernel_size=ks,
                stride=stride,
                padding=(ks - 1) // 2,
                groups=groups,
                bias=False,
            )
        else:
            self.conv2 = nn.ConvTranspose2d(
                width,
                width,
                kernel_size=ks,
                stride=stride,
                padding=(ks - stride) // 2,
                groups=groups,
                bias=False,
            )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, cout, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or cin != cout:
            if down:
                self.downsample = nn.Sequential(
                    nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                    norm_layer(cout),
                )
            else:
                self.downsample = nn.Sequential(
                    # ks = stride here s.t. resolution can be kept
                    nn.ConvTranspose2d(
                        cin, cout, kernel_size=2, stride=stride, bias=False
                    ),
                    norm_layer(cout),
                )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MultiScaleResNextBlock(nn.Module):
    def __init__(self, down, cin, cout, ks_s, ks_l, stride, norm_layer):
        super(MultiScaleResNextBlock, self).__init__()
        self.resnext_small = ResNextBlock(
            down, cin, cout // 2, ks_s, stride, norm_layer=norm_layer
        )
        self.resnext_large = ResNextBlock(
            down, cin, cout // 2, ks_l, stride, norm_layer=norm_layer
        )
        self.attention = SEBlock(cout)

    def forward(self, tensorCombine):
        out_small = self.resnext_small(tensorCombine)
        out_large = self.resnext_large(tensorCombine)
        out = torch.cat([out_small, out_large], 1)
        out = self.attention(out)
        return out


class UMultiScaleResNext(nn.Module):
    def __init__(
        self, channels=[64, 128, 256, 512], norm_layer="batch", inplanes=6, **kwargs
    ):
        super(UMultiScaleResNext, self).__init__()
        self.conv1 = MultiScaleResNextBlock(
            True, inplanes, channels[0], ks_s=3, ks_l=7, stride=2, norm_layer=norm_layer
        )
        self.conv2 = MultiScaleResNextBlock(
            True,
            channels[0],
            channels[1],
            ks_s=3,
            ks_l=7,
            stride=2,
            norm_layer=norm_layer,
        )
        self.conv3 = MultiScaleResNextBlock(
            True,
            channels[1],
            channels[2],
            ks_s=3,
            ks_l=5,
            stride=2,
            norm_layer=norm_layer,
        )
        self.conv4 = MultiScaleResNextBlock(
            True,
            channels[2],
            channels[3],
            ks_s=3,
            ks_l=5,
            stride=2,
            norm_layer=norm_layer,
        )

        self.deconv4 = MultiScaleResNextBlock(
            True,
            channels[3],
            channels[3],
            ks_s=3,
            ks_l=5,
            stride=1,
            norm_layer=norm_layer,
        )
        self.deconv3 = MultiScaleResNextBlock(
            False,
            channels[3],
            channels[2],
            ks_s=4,
            ks_l=6,
            stride=2,
            norm_layer=norm_layer,
        )
        self.deconv2 = MultiScaleResNextBlock(
            False,
            channels[2],
            channels[1],
            ks_s=4,
            ks_l=8,
            stride=2,
            norm_layer=norm_layer,
        )
        self.deconv1 = MultiScaleResNextBlock(
            False,
            channels[1],
            channels[0],
            ks_s=4,
            ks_l=8,
            stride=2,
            norm_layer=norm_layer,
        )

    def forward(self, im0, im2):
        tensorJoin = torch.cat([im0, im2], 1)  # (B,6,H,W)

        tensorConv1 = self.conv1(tensorJoin)
        tensorConv2 = self.conv2(tensorConv1)
        tensorConv3 = self.conv3(tensorConv2)
        tensorConv4 = self.conv4(tensorConv3)

        tensorDeconv4 = self.deconv4(tensorConv4)
        tensorDeconv3 = self.deconv3(tensorDeconv4 + tensorConv4)
        tensorDeconv2 = self.deconv2(tensorDeconv3 + tensorConv3)
        tensorDeconv1 = self.deconv1(tensorDeconv2 + tensorConv2)

        return tensorDeconv1


class MultiInputGridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6):
        super(MultiInputGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"
        assert (
            len(in_chs) == self.n_row
        ), "should give input channels for each row (scale stream)"

        for r, n_ch in enumerate(self.n_chs):
            setattr(self, f"lateral_{r}_0", LateralBlock(in_chs[r], n_ch))
            for c in range(1, self.n_col):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, *args):
        assert len(args) == self.n_row

        # extensible, memory-efficient
        cur_col = list(args)
        for c in range(int(self.n_col / 2)):
            for r in range(self.n_row):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != 0:
                    cur_col[r] += getattr(self, f"down_{r-1}_{c}")(cur_col[r - 1])

        for c in range(int(self.n_col / 2), self.n_col):
            for r in range(self.n_row - 1, -1, -1):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != self.n_row - 1:
                    cur_col[r] += getattr(self, f"up_{r}_{c-int(self.n_col/2)}")(
                        cur_col[r + 1]
                    )

        return self.lateral_final(cur_col[0])


class MIMOGridNet(nn.Module):
    def __init__(
        self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6, outrow=(0, 1, 2)
    ):
        super(MIMOGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        self.outrow = outrow
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"
        assert (
            len(in_chs) == self.n_row
        ), "should give input channels for each row (scale stream)"
        assert len(out_chs) == len(
            self.outrow
        ), "should give out channels for each output row (scale stream)"

        for r, n_ch in enumerate(self.n_chs):
            setattr(self, f"lateral_{r}_0", LateralBlock(in_chs[r], n_ch))
            for c in range(1, self.n_col):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        for i, r in enumerate(outrow):
            setattr(self, f"lateral_final_{r}", LateralBlock(self.n_chs[r], out_chs[i]))

    def forward(self, *args):
        assert len(args) == self.n_row

        # extensible, memory-efficient
        cur_col = list(args)
        for c in range(int(self.n_col / 2)):
            for r in range(self.n_row):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != 0:
                    cur_col[r] += getattr(self, f"down_{r-1}_{c}")(cur_col[r - 1])

        for c in range(int(self.n_col / 2), self.n_col):
            for r in range(self.n_row - 1, -1, -1):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != self.n_row - 1:
                    cur_col[r] += getattr(self, f"up_{r}_{c-int(self.n_col/2)}")(
                        cur_col[r + 1]
                    )

        out = []
        for r in self.outrow:
            out.append(getattr(self, f"lateral_final_{r}")(cur_col[r]))

        return out


class GeneralGridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96), n_row=3, n_col=6):
        super(GeneralGridNet, self).__init__()

        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = grid_chs
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"

        for r, n_ch in enumerate(self.n_chs):
            if r == 0:
                setattr(self, f"lateral_{r}_0", LateralBlock(in_chs, n_ch))
            for c in range(1, self.n_col):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        cur_col = [x] + [None] * (self.n_row - 1)
        for c in range(int(self.n_col / 2)):
            for r in range(self.n_row):
                if cur_col[r] != None:
                    cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                else:
                    cur_col[r] = 0.0
                if r != 0:
                    cur_col[r] += getattr(self, f"down_{r-1}_{c}")(cur_col[r - 1])

        for c in range(int(self.n_col / 2), self.n_col):
            for r in range(self.n_row - 1, -1, -1):
                cur_col[r] = getattr(self, f"lateral_{r}_{c}")(cur_col[r])
                if r != self.n_row - 1:
                    cur_col[r] += getattr(self, f"up_{r}_{c-int(self.n_col/2)}")(
                        cur_col[r + 1]
                    )

        return self.lateral_final(cur_col[0])


class GridNet(nn.Module):
    def __init__(self, in_chs, out_chs, grid_chs=(32, 64, 96)):
        super(GridNet, self).__init__()

        self.n_row = 3
        self.n_col = 6
        self.n_chs = grid_chs
        assert (
            len(grid_chs) == self.n_row
        ), "should give num channels for each row (scale stream)"

        self.lateral_init = LateralBlock(in_chs, self.n_chs[0])

        for r, n_ch in enumerate(self.n_chs):
            for c in range(self.n_col - 1):
                setattr(self, f"lateral_{r}_{c}", LateralBlock(n_ch, n_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"down_{r}_{c}", DownSamplingBlock(in_ch, out_ch))

        for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
            for c in range(int(self.n_col / 2)):
                setattr(self, f"up_{r}_{c}", UpSamplingBlock(in_ch, out_ch))

        self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

    def forward(self, x):
        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)


class LateralBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(LateralBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
        return fx + x


class DownSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DownSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)


class UpSamplingBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpSamplingBlock, self).__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.f(x)

# end


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=3,
                        out_channels=16,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=16,
                        out_channels=32,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=64,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=96,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=128,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=196,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=196,
                        out_channels=196,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

            # end

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

            # end

        # end

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 1]
                intCurrent = [
                    None,
                    None,
                    81 + 32 + 2 + 2,
                    81 + 64 + 2 + 2,
                    81 + 96 + 2 + 2,
                    81 + 128 + 2 + 2,
                    81,
                    None,
                ][intLevel + 0]

                if intLevel < 6:
                    self.netUpflow = torch.nn.ConvTranspose2d(
                        in_channels=2,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.netUpfeat = torch.nn.ConvTranspose2d(
                        in_channels=intPrevious + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                if intLevel < 6:
                    self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][
                        intLevel + 1
                    ]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=intCurrent + 128 + 128 + 96 + 64 + 32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

            # end

            def forward(self, tenFirst, tenSecond, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=FunctionCorrelation(
                            tenFirst=tenFirst, tenSecond=tenSecond
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious["tenFlow"])
                    tenFeat = self.netUpfeat(objPrevious["tenFeat"])

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=FunctionCorrelation(
                            tenFirst=tenFirst,
                            tenSecond=backwarp(
                                tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp
                            ),
                        ),
                        negative_slope=0.1,
                        inplace=False,
                    )

                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

                # end

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)

                return {"tenFlow": tenFlow, "tenFeat": tenFeat}

            # end

        # end

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=2,
                        dilation=2,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=128,
                        kernel_size=3,
                        stride=1,
                        padding=4,
                        dilation=4,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=128,
                        out_channels=96,
                        kernel_size=3,
                        stride=1,
                        padding=8,
                        dilation=8,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=96,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=16,
                        dilation=16,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=64,
                        out_channels=32,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(
                        in_channels=32,
                        out_channels=2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dilation=1,
                    ),
                )

            # end

            def forward(self, tenInput):
                return self.netMain(tenInput)

            # end

        # end

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        self.load_state_dict(
            {
                strKey.replace("module", "net"): tenWeight
                for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                    url="http://content.sniklaus.com/github/pytorch-pwc/network-"
                    + "default"
                    + ".pytorch",
                    model_dir=get_ckpt_container_path(MODEL_TYPE)
                ).items()
            }
        )

    # end

    def forward(self, tenFirst, tenSecond, *args):
        # optionally pass pre-extracted feature pyramid in as args
        if len(args) == 0:
            tenFirst = self.netExtractor(tenFirst)
            tenSecond = self.netExtractor(tenSecond)
        else:
            tenFirst, tenSecond = args

        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

        return objEstimate["tenFlow"] + self.netRefiner(objEstimate["tenFeat"])

    # end

    def extract_pyramid(self, tenFirst, tenSecond):
        return self.netExtractor(tenFirst), self.netExtractor(tenSecond)

    def extract_pyramid_single(self, tenFirst):
        return self.netExtractor(tenFirst)


# end

netNetwork = None

##########################################################


def estimate(tenFirst, tenSecond):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert tenFirst.shape[1] == tenSecond.shape[1]
    assert tenFirst.shape[2] == tenSecond.shape[2]

    intWidth = tenFirst.shape[2]
    intHeight = tenFirst.shape[1]

    assert (
        intWidth == 1024
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    assert (
        intHeight == 436
    )  # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(
        input=tenPreprocessedFirst,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )
    tenPreprocessedSecond = torch.nn.functional.interpolate(
        input=tenPreprocessedSecond,
        size=(intPreprocessedHeight, intPreprocessedWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow = 20.0 * torch.nn.functional.interpolate(
        input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond),
        size=(intHeight, intWidth),
        mode="bilinear",
        align_corners=False,
    )

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


# end


class UNet3d_18(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], bn=True):
        super(UNet3d_18, self).__init__()
        growth = 2  # since concatenating previous outputs
        upmode = "transpose"  # use transposeConv to upsample

        self.channels = channels

        self.lrelu = nn.LeakyReLU(0.2, True)

        self.encoder = r3d_18(bn=bn, channels=channels)

        self.decoder = nn.Sequential(
            Conv_3d(
                channels[::-1][0],
                channels[::-1][1],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            upConv3D(
                channels[::-1][1] * growth,
                channels[::-1][2],
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                upmode=upmode,
            ),
            upConv3D(
                channels[::-1][2] * growth,
                channels[::-1][3],
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                upmode=upmode,
            ),
            Conv_3d(
                channels[::-1][3] * growth,
                channels[::-1][3],
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            upConv3D(
                channels[::-1][3] * growth,
                channels[::-1][3],
                kernel_size=(3, 4, 4),
                stride=(1, 2, 2),
                padding=(1, 1, 1),
                upmode=upmode,
            ),
        )

        self.feature_fuse = nn.Sequential(
            *(
                [
                    nn.Conv2d(
                        channels[::-1][3] * 5,
                        channels[::-1][3],
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    )
                ]
                + [nn.BatchNorm2d(channels[::-1][3]) if bn else Identity]
            )
        )

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels[::-1][3], 3, kernel_size=7, stride=1, padding=0),
        )

    def forward(self, im1, im3, im5, im7, im4_tilde):
        images = torch.stack((im1, im3, im4_tilde, im5, im7), dim=2)

        x_0, x_1, x_2, x_3, x_4 = self.encoder(images)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        dx_3 = torch.cat([dx_3, x_3], dim=1)

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = torch.cat([dx_2, x_2], dim=1)

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = torch.cat([dx_1, x_1], dim=1)

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = torch.cat([dx_0, x_0], dim=1)

        dx_out = self.lrelu(self.decoder[4](dx_0))
        dx_out = torch.cat(torch.unbind(dx_out, 2), 1)

        out = self.lrelu(self.feature_fuse(dx_out))
        out = self.outconv(out)

        return out


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Subnet_offset(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
            )

        def Subnet_weight(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.Softmax(dim=1),
            )

        def Subnet_offset_ds(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
            )

        def Subnet_weight_ds(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.Softmax(dim=1),
            )

        def Subnet_offset_us(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
            )

        def Subnet_weight_us(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(
                    in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                torch.nn.Conv2d(
                    in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1
                ),
                torch.nn.Softmax(dim=1),
            )

        self.moduleWeight1_ds = Subnet_weight_ds(self.kernel_size**2)
        self.moduleAlpha1_ds = Subnet_offset_ds(self.kernel_size**2)
        self.moduleBeta1_ds = Subnet_offset_ds(self.kernel_size**2)
        self.moduleWeight2_ds = Subnet_weight_ds(self.kernel_size**2)
        self.moduleAlpha2_ds = Subnet_offset_ds(self.kernel_size**2)
        self.moduleBeta2_ds = Subnet_offset_ds(self.kernel_size**2)

        self.moduleWeight1 = Subnet_weight(self.kernel_size**2)
        self.moduleAlpha1 = Subnet_offset(self.kernel_size**2)
        self.moduleBeta1 = Subnet_offset(self.kernel_size**2)
        self.moduleWeight2 = Subnet_weight(self.kernel_size**2)
        self.moduleAlpha2 = Subnet_offset(self.kernel_size**2)
        self.moduleBeta2 = Subnet_offset(self.kernel_size**2)

        self.moduleWeight1_us = Subnet_weight_us(self.kernel_size**2)
        self.moduleAlpha1_us = Subnet_offset_us(self.kernel_size**2)
        self.moduleBeta1_us = Subnet_offset_us(self.kernel_size**2)
        self.moduleWeight2_us = Subnet_weight_us(self.kernel_size**2)
        self.moduleAlpha2_us = Subnet_offset_us(self.kernel_size**2)
        self.moduleBeta2_us = Subnet_offset_us(self.kernel_size**2)

    def forward(self, tensorCombine):
        # Frame 0
        Weight1_ds = self.moduleWeight1_ds(tensorCombine)
        Weight1 = self.moduleWeight1(tensorCombine)
        Weight1_us = self.moduleWeight1_us(tensorCombine)
        Alpha1_ds = self.moduleAlpha1_ds(tensorCombine)
        Alpha1 = self.moduleAlpha1(tensorCombine)
        Alpha1_us = self.moduleAlpha1_us(tensorCombine)
        Beta1_ds = self.moduleBeta1_ds(tensorCombine)
        Beta1 = self.moduleBeta1(tensorCombine)
        Beta1_us = self.moduleBeta1_us(tensorCombine)

        # Frame 2
        Weight2_ds = self.moduleWeight2_ds(tensorCombine)
        Weight2 = self.moduleWeight2(tensorCombine)
        Weight2_us = self.moduleWeight2_us(tensorCombine)
        Alpha2_ds = self.moduleAlpha2_ds(tensorCombine)
        Alpha2 = self.moduleAlpha2(tensorCombine)
        Alpha2_us = self.moduleAlpha2_us(tensorCombine)
        Beta2_ds = self.moduleBeta2_ds(tensorCombine)
        Beta2 = self.moduleBeta2(tensorCombine)
        Beta2_us = self.moduleBeta2_us(tensorCombine)

        return (
            Weight1_ds,
            Alpha1_ds,
            Beta1_ds,
            Weight2_ds,
            Alpha2_ds,
            Beta2_ds,
            Weight1,
            Alpha1,
            Beta1,
            Weight2,
            Alpha2,
            Beta2,
            Weight1_us,
            Alpha1_us,
            Beta1_us,
            Weight2_us,
            Alpha2_us,
            Beta2_us,
        )


class STMFNet_Model(torch.nn.Module):
    def __init__(self):
        super(STMFNet_Model, self).__init__()

        class Metric(torch.nn.Module):
            def __init__(self):
                super(Metric, self).__init__()
                self.paramScale = torch.nn.Parameter(-torch.ones(1, 1, 1, 1))

            def forward(self, tenFirst, tenSecond, tenFlow):
                return self.paramScale * F.l1_loss(
                    input=tenFirst,
                    target=backwarp(tenSecond, tenFlow),
                    reduction="none",
                ).mean(1, True)

        self.kernel_size = 5
        self.dilation = 1
        self.featc = [64, 128, 256, 512]
        self.featnorm = "batch"
        self.finetune_pwc = False

        self.kernel_pad = int(((self.kernel_size - 1) * self.dilation) / 2.0)

        self.feature_extractor = UMultiScaleResNext(
            self.featc, norm_layer=self.featnorm
        )

        self.get_kernel = KernelEstimation(self.kernel_size)

        self.modulePad = torch.nn.ReplicationPad2d(
            [self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad]
        )

        self.moduleAdaCoF = FunctionAdaCoF.apply

        self.gauss_kernel = torch.nn.Parameter(
            gaussian_kernel(5, 0.5).repeat(3, 1, 1, 1), requires_grad=False
        )

        self.upsampler = Upsampler_8tap()

        self.scale_synthesis = MIMOGridNet(
            (6, 6 + 6, 6), (3,), grid_chs=(32, 64, 96), n_row=3, n_col=4, outrow=(1,)
        )

        self.flow_estimator = PWCNet()

        self.softsplat = ModuleSoftsplat(strType="softmax")

        self.metric = Metric()

        self.dyntex_generator = UNet3d_18(bn=self.featnorm)

        # freeze weights of PWCNet if not finetuning it
        if not self.finetune_pwc:
            for param in self.flow_estimator.parameters():
                param.requires_grad = False

    def forward(self, I0, I1, I2, I3):
        h0 = int(list(I1.size())[2])
        w0 = int(list(I1.size())[3])
        h2 = int(list(I2.size())[2])
        w2 = int(list(I2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit("Frame sizes do not match")

        h_padded = False
        w_padded = False
        if h0 % 128 != 0:
            pad_h = 128 - (h0 % 128)
            I0 = F.pad(I0, (0, 0, 0, pad_h), mode="reflect")
            I1 = F.pad(I1, (0, 0, 0, pad_h), mode="reflect")
            I2 = F.pad(I2, (0, 0, 0, pad_h), mode="reflect")
            I3 = F.pad(I3, (0, 0, 0, pad_h), mode="reflect")
            h_padded = True

        if w0 % 128 != 0:
            pad_w = 128 - (w0 % 128)
            I0 = F.pad(I0, (0, pad_w, 0, 0), mode="reflect")
            I1 = F.pad(I1, (0, pad_w, 0, 0), mode="reflect")
            I2 = F.pad(I2, (0, pad_w, 0, 0), mode="reflect")
            I3 = F.pad(I3, (0, pad_w, 0, 0), mode="reflect")
            w_padded = True

        feats = self.feature_extractor(moduleNormalize(I1), moduleNormalize(I2))
        kernelest = self.get_kernel(feats)
        Weight1_ds, Alpha1_ds, Beta1_ds, Weight2_ds, Alpha2_ds, Beta2_ds = kernelest[:6]
        Weight1, Alpha1, Beta1, Weight2, Alpha2, Beta2 = kernelest[6:12]
        Weight1_us, Alpha1_us, Beta1_us, Weight2_us, Alpha2_us, Beta2_us = kernelest[
            12:
        ]

        # Original scale
        tensorAdaCoF1 = (
            self.moduleAdaCoF(self.modulePad(I1), Weight1, Alpha1, Beta1, self.dilation)
            * 1.0
        )
        tensorAdaCoF2 = (
            self.moduleAdaCoF(self.modulePad(I2), Weight2, Alpha2, Beta2, self.dilation)
            * 1.0
        )

        # 1/2 downsampled version
        c, h, w = I1.shape[1:]
        p = (self.gauss_kernel.shape[-1] - 1) // 2
        I1_blur = F.conv2d(
            F.pad(I1, pad=(p, p, p, p), mode="reflect"), self.gauss_kernel, groups=c
        )
        I2_blur = F.conv2d(
            F.pad(I2, pad=(p, p, p, p), mode="reflect"), self.gauss_kernel, groups=c
        )
        I1_ds = F.interpolate(
            I1_blur, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        I2_ds = F.interpolate(
            I2_blur, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )
        tensorAdaCoF1_ds = (
            self.moduleAdaCoF(
                self.modulePad(I1_ds), Weight1_ds, Alpha1_ds, Beta1_ds, self.dilation
            )
            * 1.0
        )
        tensorAdaCoF2_ds = (
            self.moduleAdaCoF(
                self.modulePad(I2_ds), Weight2_ds, Alpha2_ds, Beta2_ds, self.dilation
            )
            * 1.0
        )

        # x2 upsampled version
        I1_us = self.upsampler(I1)
        I2_us = self.upsampler(I2)
        tensorAdaCoF1_us = (
            self.moduleAdaCoF(
                self.modulePad(I1_us), Weight1_us, Alpha1_us, Beta1_us, self.dilation
            )
            * 1.0
        )
        tensorAdaCoF2_us = (
            self.moduleAdaCoF(
                self.modulePad(I2_us), Weight2_us, Alpha2_us, Beta2_us, self.dilation
            )
            * 1.0
        )

        # use softsplat for refinement
        pyramid0, pyramid2 = self.flow_estimator.extract_pyramid(I1, I2)
        flow_0_2 = 20 * self.flow_estimator(I1, I2, pyramid0, pyramid2)
        flow_0_2 = F.interpolate(
            flow_0_2, size=(h, w), mode="bilinear", align_corners=False
        )
        flow_2_0 = 20 * self.flow_estimator(I2, I1, pyramid2, pyramid0)
        flow_2_0 = F.interpolate(
            flow_2_0, size=(h, w), mode="bilinear", align_corners=False
        )
        metric_0_2 = self.metric(I1, I2, flow_0_2)
        metric_2_0 = self.metric(I2, I1, flow_2_0)
        tensorSoftsplat0 = self.softsplat(I1, 0.5 * flow_0_2, metric_0_2)
        tensorSoftsplat2 = self.softsplat(I2, 0.5 * flow_2_0, metric_2_0)

        # synthesize multiple scales
        tensorCombine_us = torch.cat([tensorAdaCoF1_us, tensorAdaCoF2_us], dim=1)
        tensorCombine = torch.cat(
            [tensorAdaCoF1, tensorAdaCoF2, tensorSoftsplat0, tensorSoftsplat2], dim=1
        )
        tensorCombine_ds = torch.cat([tensorAdaCoF1_ds, tensorAdaCoF2_ds], dim=1)
        output_tilde = self.scale_synthesis(
            tensorCombine_us, tensorCombine, tensorCombine_ds
        )[0]

        # generate dynamic texture
        dyntex = self.dyntex_generator(I0, I1, I2, I3, output_tilde)
        output = output_tilde + dyntex

        if h_padded:
            output = output[:, :, 0:h0, :]
        if w_padded:
            output = output[:, :, :, 0:w0]

        if self.training:
            return {"frame1": output}
        else:
            return output
import math
import numpy as np

import torch
import torch.nn as nn

from .common import *
from comfy.model_management import get_torch_device

class Encoder(nn.Module):
    def __init__(self, in_channels=3, depth=3):
        super(Encoder, self).__init__()
        self.device = get_torch_device()
        
        self.shuffler = PixelShuffle(1/2**depth)
        # self.shuffler = nn.Sequential(
        #    PixelShuffle(1/2),
        #    PixelShuffle(1/2),
        #    PixelShuffle(1/2))
        self.interpolate = Interpolation_res(5, 12, in_channels * (4**depth))

    def forward(self, x1, x2):
        feats1 = self.shuffler(x1)
        feats2 = self.shuffler(x2)

        feats = self.interpolate(feats1, feats2)

        return feats


class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()
        self.device = get_torch_device()

        self.shuffler = PixelShuffle(2**depth)
        # self.shuffler = nn.Sequential(
        #    PixelShuffle(2),
        #    PixelShuffle(2),
        #    PixelShuffle(2))

    def forward(self, feats):
        out = self.shuffler(feats)
        return out


class CAIN_NoCA(nn.Module):
    def __init__(self, depth=3):
        super(CAIN_NoCA, self).__init__()
        self.depth = depth

        self.encoder = Encoder(in_channels=3, depth=depth)
        self.decoder = Decoder(depth=depth)

    def forward(self, x1, x2):
        x1, m1 = sub_mean(x1)
        x2, m2 = sub_mean(x2)

        if not self.training:
            paddingInput, paddingOutput = InOutPaddings(x1)
            x1 = paddingInput(x1)
            x2 = paddingInput(x2)

        feats = self.encoder(x1, x2)
        out = self.decoder(feats)

        if not self.training:
            out = paddingOutput(out)

        mi = (m1 + m2) / 2
        out += mi

        return out, feats
"""
26-Dez-21
https://github.com/hzwer/Practical-RIFE
https://github.com/hzwer/Practical-RIFE/blob/main/model/warplayer.py
https://github.com/HolyWu/vs-rife/blob/master/vsrife/__init__.py
"""
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import warnings
from comfy.model_management import get_torch_device

device = get_torch_device()
backwarp_tenGrid = {}


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)

    if tenInput.type() == "torch.cuda.HalfTensor":
        g = g.half()

    padding_mode = "border"
    if device.type == "mps":
        # https://github.com/pytorch/pytorch/issues/125098
        padding_mode = "zeros"
        g = g.clamp(-1, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


def conv(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    arch_ver="4.0",
):
    if arch_ver == "4.0":
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.PReLU(out_planes),
        )
    if arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
        )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
    )


def conv_woact(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1, arch_ver="4.0"):
    if arch_ver == "4.0":
        return nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.PReLU(out_planes),
        )
    if arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
        return nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
        )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, arch_ver="4.0"):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1, arch_ver=arch_ver)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1, arch_ver=arch_ver)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64, arch_ver="4.0"):
        super(IFBlock, self).__init__()
        self.arch_ver = arch_ver
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1, arch_ver=arch_ver),
            conv(c // 2, c, 3, 2, 1, arch_ver=arch_ver),
        )
        self.arch_ver = arch_ver

        if arch_ver in ["4.0", "4.2", "4.3"]:
            self.convblock = nn.Sequential(
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
                conv(c, c, arch_ver=arch_ver),
            )
            self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

        if arch_ver in ["4.5", "4.6", "4.7", "4.10"]:
            self.convblock = nn.Sequential(
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
                ResConv(c),
            )
        if arch_ver == "4.5":
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 5, 4, 2, 1), nn.PixelShuffle(2)
            )
        if arch_ver in ["4.6", "4.7", "4.10"]:
            self.lastconv = nn.Sequential(
                nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2)
            )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        if self.arch_ver == "4.0":
            feat = self.convblock(feat) + feat
        if self.arch_ver in ["4.2", "4.3", "4.5", "4.6", "4.7", "4.10"]:
            feat = self.convblock(feat)

        tmp = self.lastconv(feat)
        if self.arch_ver in ["4.0", "4.2", "4.3"]:
            tmp = F.interpolate(
                tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
            )
            flow = tmp[:, :4] * scale * 2
        if self.arch_ver in ["4.5", "4.6", "4.7", "4.10"]:
            tmp = F.interpolate(
                tmp, scale_factor=scale, mode="bilinear", align_corners=False
            )
            flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class Contextnet(nn.Module):
    def __init__(self, arch_ver="4.0"):
        super(Contextnet, self).__init__()
        c = 16
        self.conv1 = Conv2(3, c, arch_ver=arch_ver)
        self.conv2 = Conv2(c, 2 * c, arch_ver=arch_ver)
        self.conv3 = Conv2(2 * c, 4 * c, arch_ver=arch_ver)
        self.conv4 = Conv2(4 * c, 8 * c, arch_ver=arch_ver)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = (
            F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False)
            * 0.5
        )
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class Unet(nn.Module):
    def __init__(self, arch_ver="4.0"):
        super(Unet, self).__init__()
        c = 16
        self.down0 = Conv2(17, 2 * c, arch_ver=arch_ver)
        self.down1 = Conv2(4 * c, 4 * c, arch_ver=arch_ver)
        self.down2 = Conv2(8 * c, 8 * c, arch_ver=arch_ver)
        self.down3 = Conv2(16 * c, 16 * c, arch_ver=arch_ver)
        self.up0 = deconv(32 * c, 8 * c, arch_ver=arch_ver)
        self.up1 = deconv(16 * c, 4 * c, arch_ver=arch_ver)
        self.up2 = deconv(8 * c, 2 * c, arch_ver=arch_ver)
        self.up3 = deconv(4 * c, c, arch_ver=arch_ver)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(
            torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1)
        )
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)


"""
currently supports 4.0-4.12

4.0: 4.0, 4.1
4.2: 4.2
4.3: 4.3, 4.4
4.5: 4.5
4.6: 4.6
4.7: 4.7, 4.8, 4.9
4.10: 4.10 4.11 4.12
"""


class IFNet(nn.Module):
    def __init__(self, arch_ver="4.0"):
        super(IFNet, self).__init__()
        self.arch_ver = arch_ver
        if arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
            self.block0 = IFBlock(7, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4, c=64, arch_ver=arch_ver)
        if arch_ver in ["4.7"]:
            self.block0 = IFBlock(7 + 8, c=192, arch_ver=arch_ver)
            self.block1 = IFBlock(8 + 4 + 8, c=128, arch_ver=arch_ver)
            self.block2 = IFBlock(8 + 4 + 8, c=96, arch_ver=arch_ver)
            self.block3 = IFBlock(8 + 4 + 8, c=64, arch_ver=arch_ver)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1)
            )
        if arch_ver in ["4.10"]:
            self.block0 = IFBlock(7 + 16, c=192)
            self.block1 = IFBlock(8 + 4 + 16, c=128)
            self.block2 = IFBlock(8 + 4 + 16, c=96)
            self.block3 = IFBlock(8 + 4 + 16, c=64)
            self.encode = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                nn.ConvTranspose2d(32, 8, 4, 2, 1),
            )

        if arch_ver in ["4.0", "4.2", "4.3"]:
            self.contextnet = Contextnet(arch_ver=arch_ver)
            self.unet = Unet(arch_ver=arch_ver)
        self.arch_ver = arch_ver

    def forward(
        self,
        img0,
        img1,
        timestep=0.5,
        scale_list=[8, 4, 2, 1],
        training=True,
        fastmode=True,
        ensemble=False,
        return_flow=False,
    ):
        img0 = torch.clamp(img0, 0, 1)
        img1 = torch.clamp(img1, 0, 1)

        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        x = torch.cat((img0, img1), 1)

        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        flow_list = []
        merged = []
        mask_list = []

        if self.arch_ver in ["4.7", "4.10"]:
            f0 = self.encode(img0[:, :3])
            f1 = self.encode(img1[:, :3])

        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]

        for i in range(4):
            if flow is None:
                # 4.0-4.6
                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    flow, mask = block[i](
                        torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                        None,
                        scale=scale_list[i],
                    )
                    if ensemble:
                        f1, m1 = block[i](
                            torch.cat((img1[:, :3], img0[:, :3], 1 - timestep), 1),
                            None,
                            scale=scale_list[i],
                        )
                        flow = (flow + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                        mask = (mask + (-m1)) / 2

                # 4.7+
                if self.arch_ver in ["4.7", "4.10"]:
                    flow, mask = block[i](
                        torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                        None,
                        scale=scale_list[i],
                    )

                    if ensemble:
                        f_, m_ = block[i](
                            torch.cat(
                                (img1[:, :3], img0[:, :3], f1, f0, 1 - timestep), 1
                            ),
                            None,
                            scale=scale_list[i],
                        )
                        flow = (flow + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                        mask = (mask + (-m_)) / 2

            else:
                # 4.0-4.6
                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    f0, m0 = block[i](
                        torch.cat(
                            (warped_img0[:, :3], warped_img1[:, :3], timestep, mask), 1
                        ),
                        flow,
                        scale=scale_list[i],
                    )

                if self.arch_ver in ["4.0"]:
                    if (
                        i == 1
                        and f0[:, :2].abs().max() > 32
                        and f0[:, 2:4].abs().max() > 32
                        and not training
                    ):
                        for k in range(4):
                            scale_list[k] *= 2
                        flow, mask = block[0](
                            torch.cat((img0[:, :3], img1[:, :3], timestep), 1),
                            None,
                            scale=scale_list[0],
                        )
                        warped_img0 = warp(img0, flow[:, :2])
                        warped_img1 = warp(img1, flow[:, 2:4])
                        f0, m0 = block[i](
                            torch.cat(
                                (
                                    warped_img0[:, :3],
                                    warped_img1[:, :3],
                                    timestep,
                                    mask,
                                ),
                                1,
                            ),
                            flow,
                            scale=scale_list[i],
                        )

                # 4.7+
                if self.arch_ver in ["4.7", "4.10"]:
                    fd, m0 = block[i](
                        torch.cat(
                            (
                                warped_img0[:, :3],
                                warped_img1[:, :3],
                                warp(f0, flow[:, :2]),
                                warp(f1, flow[:, 2:4]),
                                timestep,
                                mask,
                            ),
                            1,
                        ),
                        flow,
                        scale=scale_list[i],
                    )
                    flow = flow + fd

                # 4.0-4.6 ensemble
                if ensemble and self.arch_ver in [
                    "4.0",
                    "4.2",
                    "4.3",
                    "4.5",
                    "4.6",
                ]:
                    f1, m1 = block[i](
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale_list[i],
                    )
                    f0 = (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
                    m0 = (m0 + (-m1)) / 2

                # 4.7+ ensemble
                if ensemble and self.arch_ver in ["4.7", "4.10"]:
                    wf0 = warp(f0, flow[:, :2])
                    wf1 = warp(f1, flow[:, 2:4])

                    f_, m_ = block[i](
                        torch.cat(
                            (
                                warped_img1[:, :3],
                                warped_img0[:, :3],
                                wf1,
                                wf0,
                                1 - timestep,
                                -mask,
                            ),
                            1,
                        ),
                        torch.cat((flow[:, 2:4], flow[:, :2]), 1),
                        scale=scale_list[i],
                    )
                    fd = (fd + torch.cat((f_[:, 2:4], f_[:, :2]), 1)) / 2
                    mask = (m0 + (-m_)) / 2

                if self.arch_ver in ["4.0", "4.2", "4.3", "4.5", "4.6"]:
                    flow = flow + f0
                    mask = mask + m0

                if not ensemble and self.arch_ver in ["4.7", "4.10"]:
                    mask = m0

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        if self.arch_ver in ["4.0", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]:
            mask_list[3] = torch.sigmoid(mask_list[3])
            merged[3] = merged[3][0] * mask_list[3] + merged[3][1] * (1 - mask_list[3])

        if self.arch_ver in ["4.7", "4.10"]:
            mask = torch.sigmoid(mask)
            merged[3] = warped_img0 * mask + warped_img1 * (1 - mask)

        if not fastmode and self.arch_ver in ["4.0", "4.2", "4.3"]:
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            res = tmp[:, :3] * 2 - 1
            merged[3] = torch.clamp(merged[3] + res, 0, 1)
        return merged[3][:, :, :h, :w]
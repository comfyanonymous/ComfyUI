"""
https://github.com/98mxr/IFUNet/blob/main/model/IFUNet.py
https://github.com/98mxr/IFUNet/blob/main/model/cbam.py
https://github.com/98mxr/IFUNet/blob/main/model/warplayer.py
https://github.com/98mxr/IFUNet/blob/5be535c8cff66d6fa1967252685719df4c0620e4/model/RIFE.py
https://github.com/98mxr/IFUNet/blob/main/model/rrdb.py
https://github.com/98mxr/IFUNet/blob/main/model/ResynNet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.model_management import get_torch_device

backwarp_tenGrid = {}
device = get_torch_device()


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
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


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes),
    )


class DegCNN(nn.Module):
    def __init__(self):
        super(DegCNN, self).__init__()
        self.conv0 = conv(3, 32, 3, 2, 1)
        self.conv1 = conv(32, 32, 3, 2, 1)
        self.conv2 = conv(32, 32, 3, 2, 1)
        self.conv3 = conv(32, 32, 3, 2, 1)
        self.deconv = nn.Sequential(
            nn.Dropout2d(0.95),
            nn.ConvTranspose2d(4 * 32, 32, 4, 2, 1),
            nn.PReLU(32),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        f0 = self.conv0(x)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f1 = F.interpolate(f1, scale_factor=2.0, mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, scale_factor=4.0, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, scale_factor=8.0, mode="bilinear", align_corners=False)
        return self.deconv(torch.cat((f0, f1, f2, f3), 1))


class FlowBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(FlowBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv_bn(in_planes, c // 2, 3, 2, 1),
            conv_bn(c // 2, c, 3, 2, 1),
            conv_bn(c, 2 * c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv_bn(2 * c, 2 * c),
            conv_bn(2 * c, 2 * c),
            conv_bn(2 * c, 2 * c),
            conv_bn(2 * c, 2 * c),
            conv_bn(2 * c, 2 * c),
            conv_bn(2 * c, 2 * c),
        )
        self.lastconv = nn.ConvTranspose2d(2 * c, 4, 4, 2, 1)

    def forward(self, x, flow, scale=1):
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
        feat = self.convblock(feat) + feat
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale * 4, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :2] * scale * 4
        mask = tmp[:, 2:3]
        return flow, mask


class ResynNet(nn.Module):
    def __init__(self):
        super(ResynNet, self).__init__()
        self.block0 = FlowBlock(6, c=128)
        self.block1 = FlowBlock(12, c=128)
        self.block2 = FlowBlock(12, c=128)
        self.degrad = DegCNN()
        # Contextual Refinement context + decode
        self.context0 = nn.Sequential(
            conv(3, 16, 3, 2, 1),
            conv(16, 32, 3, 2, 1),
        )
        self.context1 = nn.Sequential(
            conv(3, 16, 3, 2, 1),
            conv(16, 32, 3, 2, 1),
        )
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def calflow(self, img0, lowres, scale):
        flow = None
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](
                    torch.cat((img0, lowres, warped_img0, mask), 1),
                    flow,
                    scale=scale[i],
                )
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](torch.cat((img0, lowres), 1), None, scale=scale[i])
            warped_img0 = warp(img0, flow)
        flow_down = (
            F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        c0 = warp(self.context0(img0), flow_down)
        c1 = self.context1(warped_img0)
        warped_img0 = warped_img0 + self.decode(torch.cat((c0, c1), 1))
        return flow, mask, torch.clamp(warped_img0, 0, 1)

    def forward(
        self, x, deg=None, gt=None, scale=[4, 2, 1], training=False, blend=True
    ):
        if training:
            deg = self.degrad(gt)
            loss_cons = (gt - deg).abs().mean()
        else:
            loss_cons = torch.tensor([0])
        img_list = []
        N = x.shape[1] // 3
        for i in range(N):
            img_list.append(x[:, i * 3 : i * 3 + 3])
        warped_list = []
        merged = []
        mask_list = []
        flow_list = []
        for i in range(N):
            f, m, img = self.calflow(img_list[i], deg.detach(), scale)
            mask_list.append(m)
            warped_list.append(img)
            flow_list.append(f)
        if blend:
            N += 1
            mask_list.append(m * 0)
            warped_list.append(deg)
        mask = F.softmax(torch.clamp(torch.cat(mask_list, 1), -4, 4), dim=1)
        merged = 0
        for i in range(N):
            merged += warped_list[i] * mask[:, i : i + 1]
        return merged, loss_cons


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
        # 只能先取消，default_init_weights来自basicsr.arch_util

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        # 原作者这么说我就这么听着吧
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        # 原作者这么说我就这么听着吧
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(
        self, num_in_ch=16, num_out_ch=1, num_feat=64, num_block=6, num_grow_ch=32
    ):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img0, img1, warped_img0, warped_img1, flow):
        x = torch.cat((img0, img1, warped_img0, warped_img1), 1)
        x = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        flow = (
            F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False)
            * 0.25
        )
        feat = torch.cat((x, flow), 1)

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample，充分利用四倍放大
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest"))
        )
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        out = torch.sigmoid(out)
        return out


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
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
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


class UNetConv(nn.Module):
    def __init__(self, in_planes, out_planes, att=True):
        super(UNetConv, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, 2, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

        if att:
            self.cbam = CBAM(out_planes, 16)  # 这一步导致了通道数最低为128
        else:
            self.cbam = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_planes, out_planes, att=True):
        super(UpConv, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_planes, in_planes // 2, 4, 2, 1),
            nn.PReLU(in_planes // 2),
        )

        # 也许不需要这么卷积，我不确定
        self.conv1 = conv(in_planes, in_planes // 2, 3, 1, 1)
        self.conv2 = conv(in_planes // 2, out_planes, 3, 1, 1)

        if att:
            self.cbam = CBAM(out_planes, 16)
        else:
            self.cbam = None

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        y = self.conv1(torch.cat((x1, x2), 1))
        y = self.conv2(y)
        if self.cbam is not None:
            y = self.cbam(y)
        return y


class FeatureNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FeatureNet, self).__init__()
        # 处理IFBlock0时通道数问题
        self.conv0 = conv(7, in_planes, 1, 1, 0)

        self.conv1 = UNetConv(in_planes, out_planes // 8, att=False)
        self.conv2 = UNetConv(out_planes // 8, out_planes // 4, att=True)
        self.conv3 = UNetConv(out_planes // 4, out_planes // 2, att=True)
        self.conv4 = UNetConv(out_planes // 2, out_planes, att=True)
        self.conv5 = UNetConv(out_planes, 2 * out_planes, att=True)

        self.deconv5 = UpConv(2 * out_planes, out_planes, att=True)
        self.deconv4 = UpConv(out_planes, out_planes // 2, att=False)
        self.deconv3 = UpConv(out_planes // 2, out_planes // 4, att=False)

    def forward(self, x, level=0):
        if x.shape[1] != 17:
            x = self.conv0(x)
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        x16 = self.conv4(x8)
        x32 = self.conv5(x16)
        y = self.deconv5(x32, x16)  # 匹配IFBlock0通道和尺寸

        # “早退机制”以期待用同一个UNet提取特征，不确定是否对训练产生影响
        if level != 0:
            y = self.deconv4(y, x8)  # 匹配IFBlock1通道和尺寸
            if level == 2:
                y = self.deconv3(y, x4)  # 匹配IFBlock2通道和尺寸
        return y


class IFBlock(nn.Module):
    def __init__(self, c=64, level=0):
        super(IFBlock, self).__init__()
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.flowconv = nn.Conv2d(c, 4, 3, 1, 1)
        self.maskconvx16 = nn.Conv2d(c, 16 * 16 * 9, 1, 1, 0)
        self.maskconvx8 = nn.Conv2d(c, 8 * 8 * 9, 1, 1, 0)
        self.maskconvx4 = nn.Conv2d(c, 4 * 4 * 9, 1, 1, 0)

        self.level = level
        assert self.level in [4, 8, 16], "Bitch"

    def mask_conv(self, x):
        if self.level == 4:
            return self.maskconvx4(x)
        if self.level == 8:
            return self.maskconvx8(x)
        if self.level == 16:
            return self.maskconvx16(x)

    def upsample_flow(self, flow, mask):
        # 俺寻思俺懂了
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.level, self.level, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.level * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 4, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 4, self.level * H, self.level * W)

    def forward(self, x, scale):
        x = self.convblock(x) + x  # 类似ResNet的f(x) + x
        tmp = self.flowconv(x)
        up_mask = self.mask_conv(x)
        flow_up = self.upsample_flow(tmp, up_mask)
        flow = (
            F.interpolate(
                flow_up, scale_factor=scale, mode="bilinear", align_corners=False
            )
            * scale
        )
        return flow


class IFUNet(nn.Module):
    def __init__(self):
        super(IFUNet, self).__init__()
        # block0通道数必须为128的整倍数
        self.fmap = FeatureNet(in_planes=17, out_planes=256)
        self.block0 = IFBlock(c=256, level=16)
        self.block1 = IFBlock(c=128, level=8)
        self.block2 = IFBlock(c=64, level=4)

    def forward(self, x, scale=1.0, timestep=0.5, ensemble=True):
        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                x = torch.cat((img0, img1, timestep, warped_img0, warped_img1), 1)
                flowtmp = flow
                if scale != 1:
                    x = F.interpolate(
                        x, scale_factor=scale, mode="bilinear", align_corners=False
                    )
                    flowtmp = (
                        F.interpolate(
                            flow,
                            scale_factor=scale,
                            mode="bilinear",
                            align_corners=False,
                        )
                        * scale
                    )
                x = torch.cat((x, flowtmp), 1)
                # 期待UNet能提取到特征，不再需要ensemble
                Fmap = self.fmap(x, level=i)
                flow_d = block[i](Fmap, scale=1.0 / scale)
                flow = flow + flow_d

                if ensemble:
                    x = torch.cat(
                        (img1, img0, 1 - timestep, warped_img0, warped_img1), 1
                    )
                    flowtmp = flow
                    if scale != 1:
                        x = F.interpolate(
                            x, scale_factor=scale, mode="bilinear", align_corners=False
                        )
                        flowtmp = (
                            F.interpolate(
                                flow,
                                scale_factor=scale,
                                mode="bilinear",
                                align_corners=False,
                            )
                            * scale
                        )
                    x = torch.cat((x, flowtmp), 1)
                    # 期待UNet能提取到特征，不再需要ensemble
                    Fmap = self.fmap(x, level=i)
                    flow_d = block[i](Fmap, scale=1.0 / scale)
                    flow2 = flow + flow_d
                    flow = (flow + flow2) / 2
            else:
                x = torch.cat((img0, img1, timestep), 1)
                if scale != 1:
                    x = F.interpolate(
                        x, scale_factor=scale, mode="bilinear", align_corners=False
                    )
                Fmap = self.fmap(x, level=i)
                flow = block[i](Fmap, scale=1.0 / scale)

                if ensemble:
                    x = torch.cat((img1, img0, 1 - timestep), 1)
                    if scale != 1:
                        x = F.interpolate(
                            x, scale_factor=scale, mode="bilinear", align_corners=False
                        )
                    Fmap = self.fmap(x, level=i)
                    flow2 = block[i](Fmap, scale=1.0 / scale)
                    flow = (flow + flow2) / 2

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        return flow, warped_img0, warped_img1


class IFUNetModel(nn.Module):
    def __init__(self, local_rank=-1):
        super(IFUNetModel, self).__init__()
        self.flownet = IFUNet()
        self.fusionnet = RRDBNet()
        self.refinenet = ResynNet()

    def forward(self, img0, img1, timestep=0.5, scale=1.0, ensemble=False):
        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        imgs = torch.cat((img0, img1), 1)
        flow, warped_img0, warped_img1 = self.flownet(imgs, scale, timestep, ensemble)
        mask = self.fusionnet(img0, img1, warped_img0, warped_img1, flow)
        merged = warped_img0 * mask + warped_img1 * (1 - mask)
        merged, _ = self.refinenet(imgs, deg=merged, scale=[4, 2, 1])
        return merged[:, :, :h, :w]

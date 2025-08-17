import torch
import torch.nn as nn
import torch.nn.functional as F

from r_basicsr.utils.registry import ARCH_REGISTRY


class SeqConv3x3(nn.Module):
    """The re-parameterizable block used in the ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        seq_type (str): Sequence type, option: conv1x1-conv3x3 | conv1x1-sobelx | conv1x1-sobely | conv1x1-laplacian.
        in_channels (int): Channel number of input.
        out_channels (int): Channel number of output.
        depth_multiplier (int): Width multiplier in the expand-and-squeeze conv. Default: 1.
    """

    def __init__(self, seq_type, in_channels, out_channels, depth_multiplier=1):
        super(SeqConv3x3, self).__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.seq_type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_channels * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.in_channels, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_channels, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.seq_type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale and bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels, ))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.seq_type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale and bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.seq_type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale and bias
            scale = torch.randn(size=(self.out_channels, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_channels) * 1e-3
            bias = torch.reshape(bias, (self.out_channels, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_channels, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_channels):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('The type of seqconv is not supported!')

    def forward(self, x):
        if self.seq_type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_channels)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.seq_type == 'conv1x1-conv3x3':
            # re-param conv kernel
            rep_weight = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            rep_bias = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_channels, self.out_channels, 3, 3), device=device)
            for i in range(self.out_channels):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            rep_weight = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            rep_bias = torch.ones(1, self.out_channels, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=k1).view(-1, ) + b1
        return rep_weight, rep_bias


class ECB(nn.Module):
    """The ECB block used in the ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        in_channels (int): Channel number of input.
        out_channels (int): Channel number of output.
        depth_multiplier (int): Width multiplier in the expand-and-squeeze conv. Default: 1.
        act_type (str): Activation type. Option: prelu | relu | rrelu | softplus | linear. Default: prelu.
        with_idt (bool): Whether to use identity connection. Default: False.
    """

    def __init__(self, in_channels, out_channels, depth_multiplier, act_type='prelu', with_idt=False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_type = act_type

        if with_idt and (self.in_channels == self.out_channels):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.in_channels, self.out_channels, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.in_channels, self.out_channels)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.in_channels, self.out_channels)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.in_channels, self.out_channels)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_channels)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x) + self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            rep_weight, rep_bias = self.rep_params()
            y = F.conv2d(input=x, weight=rep_weight, bias=rep_bias, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        weight0, bias0 = self.conv3x3.weight, self.conv3x3.bias
        weight1, bias1 = self.conv1x1_3x3.rep_params()
        weight2, bias2 = self.conv1x1_sbx.rep_params()
        weight3, bias3 = self.conv1x1_sby.rep_params()
        weight4, bias4 = self.conv1x1_lpl.rep_params()
        rep_weight, rep_bias = (weight0 + weight1 + weight2 + weight3 + weight4), (
            bias0 + bias1 + bias2 + bias3 + bias4)

        if self.with_idt:
            device = rep_weight.get_device()
            if device < 0:
                device = None
            weight_idt = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=device)
            for i in range(self.out_channels):
                weight_idt[i, i, 1, 1] = 1.0
            bias_idt = 0.0
            rep_weight, rep_bias = rep_weight + weight_idt, rep_bias + bias_idt
        return rep_weight, rep_bias


@ARCH_REGISTRY.register()
class ECBSR(nn.Module):
    """ECBSR architecture.

    Paper: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices
    Ref git repo: https://github.com/xindongzhang/ECBSR

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_block (int): Block number in the trunk network.
        num_channel (int): Channel number.
        with_idt (bool): Whether use identity in convolution layers.
        act_type (str): Activation type.
        scale (int): Upsampling factor.
    """

    def __init__(self, num_in_ch, num_out_ch, num_block, num_channel, with_idt, act_type, scale):
        super(ECBSR, self).__init__()
        self.num_in_ch = num_in_ch
        self.scale = scale

        backbone = []
        backbone += [ECB(num_in_ch, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        for _ in range(num_block):
            backbone += [ECB(num_channel, num_channel, depth_multiplier=2.0, act_type=act_type, with_idt=with_idt)]
        backbone += [
            ECB(num_channel, num_out_ch * scale * scale, depth_multiplier=2.0, act_type='linear', with_idt=with_idt)
        ]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(scale)

    def forward(self, x):
        if self.num_in_ch > 1:
            shortcut = torch.repeat_interleave(x, self.scale * self.scale, dim=1)
        else:
            shortcut = x  # will repeat the input in the channel dimension (repeat  scale * scale times)
        y = self.backbone(x) + shortcut
        y = self.upsampler(y)
        return y

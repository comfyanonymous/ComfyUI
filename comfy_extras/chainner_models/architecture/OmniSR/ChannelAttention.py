import math

import torch.nn as nn


class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=(1, 1), bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=(1, 1), bias=False),
            # nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x * y.expand_as(x)


class Simple_CA_layer(nn.Module):
    def __init__(self, channel):
        super(Simple_CA_layer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

    def forward(self, x):
        return x * self.fc(self.gap(x))


class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(ECA_layer, self).__init__()

        b = 1
        gamma = 2
        k_size = int(abs(math.log(channel, 2) + b) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        # y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECA_MaxPool_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel):
        super(ECA_MaxPool_layer, self).__init__()

        b = 1
        gamma = 2
        k_size = int(abs(math.log(channel, 2) + b) / gamma)
        k_size = k_size if k_size % 2 else k_size + 1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        # b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.max_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        # y = self.sigmoid(y)

        return x * y.expand_as(x)

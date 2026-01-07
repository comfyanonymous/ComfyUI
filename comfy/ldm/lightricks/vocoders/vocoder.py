import torch
import torch.nn.functional as F
import torch.nn as nn
import comfy.ops
import numpy as np

ops = comfy.ops.disable_weight_init

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1),
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]),
                ),
                ops.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]),
                ),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Vocoder(torch.nn.Module):
    """
    Vocoder model for synthesizing audio from spectrograms, based on: https://github.com/jik876/hifi-gan.

    """

    def __init__(self, config=None):
        super(Vocoder, self).__init__()

        if config is None:
            config = self.get_default_config()

        resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
        upsample_rates = config.get("upsample_rates", [6, 5, 2, 2, 2])
        upsample_kernel_sizes = config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4])
        resblock_dilation_sizes = config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        upsample_initial_channel = config.get("upsample_initial_channel", 1024)
        stereo = config.get("stereo", True)
        resblock = config.get("resblock", "1")

        self.output_sample_rate = config.get("output_sample_rate")
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        in_channels = 128 if stereo else 64
        self.conv_pre = ops.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        resblock_class = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ops.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_class(ch, k, d))

        out_channels = 2 if stereo else 1
        self.conv_post = ops.Conv1d(ch, out_channels, 7, 1, padding=3)

        self.upsample_factor = np.prod([self.ups[i].stride[0] for i in range(len(self.ups))])

    def get_default_config(self):
        """Generate default configuration for the vocoder."""

        config = {
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [6, 5, 2, 2, 2],
            "upsample_kernel_sizes": [16, 15, 8, 4, 4],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_initial_channel": 1024,
            "stereo": True,
            "resblock": "1",
        }

        return config

    def forward(self, x):
        """
        Forward pass of the vocoder.

        Args:
            x: Input spectrogram tensor. Can be:
               - 3D: (batch_size, channels, time_steps) for mono
               - 4D: (batch_size, 2, channels, time_steps) for stereo

        Returns:
            Audio tensor of shape (batch_size, out_channels, audio_length)
        """
        if x.dim() == 4:  # stereo
            assert x.shape[1] == 2, "Input must have 2 channels for stereo"
            x = torch.cat((x[:, 0, :, :], x[:, 1, :, :]), dim=1)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

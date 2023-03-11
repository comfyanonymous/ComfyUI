# pylint: skip-file
"""Original MAT project is copyright of fenglingwb: https://github.com/fenglinglwb/MAT
Code used for this implementation of MAT is modified from lama-cleaner,
copyright of Sanster: https://github.com/fenglinglwb/MAT"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .mat.utils import (
    Conv2dLayer,
    FullyConnectedLayer,
    activation_funcs,
    bias_act,
    conv2d_resample,
    normalize_2nd_moment,
    setup_filter,
    to_2tuple,
    upsample2d,
)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        style_dim,  # dimension of the style code
        demodulate=True,  # perfrom demodulation
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
    ):
        super().__init__()
        self.demodulate = demodulate

        self.weight = torch.nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1).to(x.device)
        weight = self.weight.to(x.device) * self.weight_gain * style

        if self.demodulate:
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch,
        )
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


class StyleConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        style_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
        use_noise=False,  # Enable noise input?
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        demodulate=True,  # perform demodulation
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            up=up,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise:
            self.register_buffer("noise_const", torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.activation = activation
        self.act_gain = activation_funcs[activation].def_gain
        self.conv_clamp = conv_clamp

    def forward(self, x, style, noise_mode="random", gain=1):
        x = self.conv(x, style)

        assert noise_mode in ["random", "const", "none"]

        if self.use_noise:
            if noise_mode == "random":
                xh, xw = x.size()[-2:]
                noise = (
                    torch.randn([x.shape[0], 1, xh, xw], device=x.device)
                    * self.noise_strength
                )
            if noise_mode == "const":
                noise = self.noise_const * self.noise_strength
            x = x + noise

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )

        return out


class ToRGB(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        kernel_size=1,
        resample_filter=[1, 3, 3, 1],
        conv_clamp=None,
        demodulate=False,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=demodulate,
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
        )
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act(x, self.bias, clamp=self.conv_clamp)

        if skip is not None:
            if skip.shape != out.shape:
                skip = upsample2d(skip, self.resample_filter)
            out = out + skip

        return out


def get_style_code(a, b):
    return torch.cat([a, b.to(a.device)], dim=1)


class DecBlockFirst(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.fc = FullyConnectedLayer(
            in_features=in_channels * 2,
            out_features=in_channels * 4**2,
            activation=activation,
        )
        self.conv = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=4,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = x + E_features[2]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


class MappingNet(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        num_ws,  # Number of intermediate latents to output, None = do not broadcast.
        num_layers=8,  # Number of mapping layers.
        embed_features=None,  # Label embedding dimensionality, None = same as w_dim.
        layer_features=None,  # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.995,  # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = (
            [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        )

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(
                in_features,
                out_features,
                activation=activation,
                lr_multiplier=lr_multiplier,
            )
            setattr(self, f"fc{idx}", layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer("w_avg", torch.zeros([w_dim]))

    def forward(
        self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False
    ):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function("input"):
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f"fc{idx}")
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function("update_w_avg"):
                self.w_avg.copy_(
                    x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
                )

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function("broadcast"):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function("truncate"):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(
                        x[:, :truncation_cutoff], truncation_psi
                    )

        return x


class DisFromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )

    def forward(self, x):
        return self.conv(x)


class DisBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=2,
            activation=activation,
        )
        self.skip = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            down=2,
            bias=False,
        )

    def forward(self, x):
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x

        return out


def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2**stage]


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(
            in_features=in_features, out_features=hidden_features, activation="lrelu"
        )
        self.fc2 = FullyConnectedLayer(
            in_features=hidden_features, out_features=out_features
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # B = windows.shape[0] / (H * W / window_size / window_size)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Conv2dLayerPartial(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[
            1,
            3,
            3,
            1,
        ],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.conv = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size,
            bias,
            activation,
            up,
            down,
            resample_filter,
            conv_clamp,
            trainable,
        )

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size**2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                )
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        down_ratio=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)
            ).masked_fill(attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(
                    torch.sum(mask_windows, dim=1, keepdim=True), 0, 1
                ).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        down_ratio=1,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.fuse = FullyConnectedLayer(
            in_features=dim * 2, out_features=dim, activation="lrelu"
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, _, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                shifted_mask = torch.roll(
                    mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
                )
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows, mask_windows = self.attn(
                x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
            )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
            if mask is not None:
                mask = torch.roll(
                    shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
                )
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            down=down,
        )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation="lrelu",
            up=up,
        )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        down_ratio=1,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    down_ratio=down_ratio,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.conv = Conv2dLayerPartial(
            in_channels=dim, out_channels=dim, kernel_size=3, activation="lrelu"
        )

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=kernel_size,
            activation="lrelu",
        )

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask


class EncFromRGB(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            activation=activation,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


class ConvBlockDown(nn.Module):
    def __init__(
        self, in_channels, out_channels, activation
    ):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            activation=activation,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, _, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, _, _ = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


class Encoder(nn.Module):
    def __init__(
        self,
        res_log2,
        img_channels,
        activation,
        patch_size=5,
        channels=16,
        drop_path_rate=0.1,
    ):
        super().__init__()

        self.resolution = []

        for i in range(res_log2, 3, -1):  # from input size to 16x16
            res = 2**i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, "EncConv_Block_%dx%d" % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, "EncConv_Block_%dx%d" % (res, res))(x)
            out[res_log2] = x

        return out


class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
            Conv2dLayer(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                activation=activation,
                down=2,
            ),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(
            in_features=in_channels, out_features=out_channels, activation=activation
        )
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


class DecBlockFirstV2(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            activation=activation,
        )
        self.conv1 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


class DecBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, ws, gs, E_features, noise_mode="random"):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


class Decoder(nn.Module):
    def __init__(
        self, res_log2, activation, style_dim, use_noise, demodulate, img_channels
    ):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(
            4, nf(4), nf(4), activation, style_dim, use_noise, demodulate, img_channels
        )
        for res in range(5, res_log2 + 1):
            setattr(
                self,
                "Dec_%dx%d" % (2**res, 2**res),
                DecBlock(
                    res,
                    nf(res - 1),
                    nf(res),
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                ),
            )
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode="random"):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, "Dec_%dx%d" % (2**res, 2**res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


class DecStyleBlock(nn.Module):
    def __init__(
        self,
        res,
        in_channels,
        out_channels,
        activation,
        style_dim,
        use_noise,
        demodulate,
        img_channels,
    ):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            use_noise=use_noise,
            activation=activation,
            demodulate=demodulate,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
            kernel_size=1,
            demodulate=False,
        )

    def forward(self, x, img, style, skip, noise_mode="random"):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


class FirstStage(nn.Module):
    def __init__(
        self,
        img_channels,
        img_resolution=256,
        dim=180,
        w_dim=512,
        use_noise=False,
        demodulate=True,
        activation="lrelu",
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels + 1,
            out_channels=dim,
            kernel_size=3,
            activation=activation,
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        # 根据图片尺寸构建 swim transformer 的层数
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(
                    dim=dim,
                    input_resolution=[res, res],
                    depth=depth,
                    num_heads=num_heads,
                    window_size=window_sizes[i],
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=merge,
                )
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    down=2,
                    activation=activation,
                )
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(
            in_features=dim, out_features=dim * 2, activation=activation
        )
        self.ws_style = FullyConnectedLayer(
            in_features=w_dim, out_features=dim, activation=activation
        )
        self.to_square = FullyConnectedLayer(
            in_features=dim, out_features=16 * 16, activation=activation
        )

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # from 64 to input size
            res = res * 2
            self.dec_conv.append(
                DecStyleBlock(
                    res,
                    dim,
                    dim,
                    activation,
                    style_dim,
                    use_noise,
                    demodulate,
                    img_channels,
                )
            )

    def forward(self, images_in, masks_in, ws, noise_mode="random"):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else:
                x, x_size, mask = block(x, x_size, None)

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True).to(x.device)
                ws = self.ws_style(ws[:, -1]).to(x.device)
                add_n = self.to_square(ws).unsqueeze(1).to(x.device)
                add_n = (
                    F.interpolate(
                        add_n, size=x.size(1), mode="linear", align_corners=False
                    )
                    .squeeze(1)
                    .unsqueeze(-1)
                ).to(x.device)
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                ).to(x.device)
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv):
            x, img = block(
                x, img, style, skips[len(self.dec_conv) - i - 1], noise_mode=noise_mode
            )

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


class SynthesisNet(nn.Module):
    def __init__(
        self,
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output image resolution.
        img_channels=3,  # Number of color channels.
        channel_base=32768,  # Overall multiplier for the number of channels.
        channel_decay=1.0,
        channel_max=512,  # Maximum number of channels in any layer.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        drop_rate=0.5,
        use_noise=False,
        demodulate=True,
    ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(
            img_channels,
            img_resolution=img_resolution,
            w_dim=w_dim,
            use_noise=False,
            demodulate=demodulate,
        )

        # second stage
        self.enc = Encoder(
            resolution_log2, img_channels, activation, patch_size=5, channels=16
        )
        self.to_square = FullyConnectedLayer(
            in_features=w_dim, out_features=16 * 16, activation=activation
        )
        self.to_style = ToStyle(
            in_channels=nf(4),
            out_channels=nf(2) * 2,
            activation=activation,
            drop_rate=drop_rate,
        )
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(
            resolution_log2, activation, style_dim, use_noise, demodulate, img_channels
        )

    def forward(self, images_in, masks_in, ws, noise_mode="random", return_stg1=False):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4].to(x.device)
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True).to(x.device)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode="bilinear", align_corners=False
        ).to(x.device)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16).to(x.device)

        # decoder
        img = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode).to(x.device)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        if not return_stg1:
            return img
        else:
            return img, out_stg1


class Generator(nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # resolution of generated image
        img_channels,  # Number of input color channels.
        synthesis_kwargs={},  # Arguments for SynthesisNetwork.
        mapping_kwargs={},  # Arguments for MappingNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            **synthesis_kwargs,
        )
        self.mapping = MappingNet(
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            num_ws=self.synthesis.num_layers,
            **mapping_kwargs,
        )

    def forward(
        self,
        images_in,
        masks_in,
        z,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        skip_w_avg_update=False,
        noise_mode="none",
        return_stg1=False,
    ):
        ws = self.mapping(
            z,
            c,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            skip_w_avg_update=skip_w_avg_update,
        )
        img = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        return img


class MAT(nn.Module):
    def __init__(self, state_dict):
        super(MAT, self).__init__()
        self.model_arch = "MAT"
        self.sub_type = "Inpaint"
        self.in_nc = 3
        self.out_nc = 3
        self.scale = 1

        self.supports_fp16 = False
        self.supports_bf16 = True

        self.min_size = 512
        self.pad_mod = 512
        self.pad_to_square = True

        seed = 240  # pick up a random number
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model = Generator(
            z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3
        )
        self.z = torch.from_numpy(np.random.randn(1, self.model.z_dim))  # [1., 512]
        self.label = torch.zeros([1, self.model.c_dim])
        self.state = {
            k.replace("synthesis", "model.synthesis").replace(
                "mapping", "model.mapping"
            ): v
            for k, v in state_dict.items()
        }
        self.load_state_dict(self.state, strict=False)

    def forward(self, image, mask):
        """Input images and output images have same size
        images: [H, W, C] RGB
        masks: [H, W] mask area == 255
        return: BGR IMAGE
        """

        image = image * 2 - 1  # [0, 1] -> [-1, 1]
        mask = 1 - mask

        output = self.model(
            image, mask, self.z, self.label, truncation_psi=1, noise_mode="none"
        )

        return output * 0.5 + 0.5

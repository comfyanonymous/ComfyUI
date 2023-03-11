"""Code used for this implementation of the MAT helper utils is modified from
lama-cleaner, copyright of Sanster: https://github.com/fenglinglwb/MAT"""

import collections
from itertools import repeat
from typing import Any

import numpy as np
import torch
from torch import conv2d, conv_transpose2d


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


activation_funcs = {
    "linear": EasyDict(
        func=lambda x, **_: x,
        def_alpha=0,
        def_gain=1,
        cuda_idx=1,
        ref="",
        has_2nd_grad=False,
    ),
    "relu": EasyDict(
        func=lambda x, **_: torch.nn.functional.relu(x),
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=2,
        ref="y",
        has_2nd_grad=False,
    ),
    "lrelu": EasyDict(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2,
        def_gain=np.sqrt(2),
        cuda_idx=3,
        ref="y",
        has_2nd_grad=False,
    ),
    "tanh": EasyDict(
        func=lambda x, **_: torch.tanh(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=4,
        ref="y",
        has_2nd_grad=True,
    ),
    "sigmoid": EasyDict(
        func=lambda x, **_: torch.sigmoid(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=5,
        ref="y",
        has_2nd_grad=True,
    ),
    "elu": EasyDict(
        func=lambda x, **_: torch.nn.functional.elu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=6,
        ref="y",
        has_2nd_grad=True,
    ),
    "selu": EasyDict(
        func=lambda x, **_: torch.nn.functional.selu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=7,
        ref="y",
        has_2nd_grad=True,
    ),
    "softplus": EasyDict(
        func=lambda x, **_: torch.nn.functional.softplus(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=8,
        ref="y",
        has_2nd_grad=True,
    ),
    "swish": EasyDict(
        func=lambda x, **_: torch.sigmoid(x) * x,
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=9,
        ref="x",
        has_2nd_grad=True,
    ),
}


def _bias_act_ref(x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None):
    """Slow reference implementation of `bias_act()` using standard TensorFlow ops."""
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)]).to(x.device)

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        x = x.clamp(-clamp, clamp)  # pylint: disable=invalid-unary-operand-type
    return x


def bias_act(
    x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None, impl="ref"
):
    r"""Fused bias and activation function.
    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports first and second order gradients,
    but not third order gradients.
    Args:
        x:      Input activation tensor. Can be of any shape.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `dim`.
        dim:    The dimension in `x` corresponding to the elements of `b`.
                The value of `dim` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying 1.
        clamp:  Clamp the output values to `[-clamp, +clamp]`, or `None` to disable
                the clamping (default).
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).
    Returns:
        Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ["ref", "cuda"]
    return _bias_act_ref(
        x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp
    )


def setup_filter(
    f,
    device=torch.device("cpu"),
    normalize=True,
    flip_filter=False,
    gain=1,
    separable=None,
):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.
    Args:
        f:           Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        device:      Result device (default: cpu).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).
    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = f.ndim == 1 and f.numel() >= 8
    if f.ndim == 1 and not separable:
        f = f.ger(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain ** (f.ndim / 2))
    f = f.to(device=device)
    return f


def _get_filter_size(f):
    if f is None:
        return 1, 1

    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]

    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh


def _get_weight_shape(w):
    shape = [int(sz) for sz in w.shape]
    return shape


def _parse_scaling(scaling):
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def _upfirdn2d_ref(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch ops."""
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape
    # upx, upy = _parse_scaling(up)
    # downx, downy = _parse_scaling(down)

    upx, upy = up, up
    downx, downy = down, down

    # padx0, padx1, pady0, pady1 = _parse_padding(padding)
    padx0, padx1, pady0, pady1 = padding[0], padding[1], padding[2], padding[3]

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(
        x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)]
    )
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1, impl="cuda"):
    r"""Pad, upsample, filter, and downsample a batch of 2D images.
    Performs the following sequence of operations for each channel:
    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).
    2. Pad the image with the specified number of zeros on each side (`padding`).
       Negative padding corresponds to cropping the image.
    3. Convolve the image with the specified 2D FIR filter (`f`), shrinking it
       so that the footprint of all output pixels lies within the input image.
    4. Downsample the image by keeping every Nth pixel (`down`).
    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard PyTorch ops. It supports gradients of arbitrary order.
    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        down:        Integer downsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the upsampled image. Can be a single number
                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    # assert isinstance(x, torch.Tensor)
    # assert impl in ['ref', 'cuda']
    return _upfirdn2d_ref(
        x, f, up=up, down=down, padding=padding, flip_filter=flip_filter, gain=gain
    )


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1, impl="cuda"):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.
    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.
    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    # upx, upy = up, up
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    # padx0, padx1, pady0, pady1 = padding, padding, padding, padding
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(
        x,
        f,
        up=up,
        padding=p,
        flip_filter=flip_filter,
        gain=gain * upx * upy,
        impl=impl,
    )


class FullyConnectedLayer(torch.nn.Module):
    def __init__(
        self,
        in_features,  # Number of input features.
        out_features,  # Number of output features.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier=1,  # Learning rate multiplier.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn([out_features, in_features]) / lr_multiplier
        )
        self.bias = (
            torch.nn.Parameter(torch.full([out_features], np.float32(bias_init)))
            if bias
            else None
        )
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            # out = torch.addmm(b.unsqueeze(0), x, w.t())
            x = x.matmul(w.t().to(x.device))
            out = x + b.reshape(
                [-1 if i == x.ndim - 1 else 1 for i in range(x.ndim)]
            ).to(x.device)
        else:
            x = x.matmul(w.t().to(x.device))
            out = bias_act(x, b, act=self.activation, dim=x.ndim - 1).to(x.device)
        return out


def _conv2d_wrapper(
    x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True
):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations."""
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)

    # Flip weight if requested.
    if (
        not flip_weight
    ):  # conv2d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if (
        kw == 1
        and kh == 1
        and stride == 1
        and padding in [0, [0, 0], (0, 0)]
        and not transpose
    ):
        if x.stride()[1] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape(
                    [in_shape[0], in_channels_per_group, -1]
                )
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                x = x.to(memory_format=torch.contiguous_format)
                w = w.to(memory_format=torch.contiguous_format)
                x = conv2d(x, w, groups=groups)
            return x.to(memory_format=torch.channels_last)

    # Otherwise => execute using conv2d_gradfix.
    op = conv_transpose2d if transpose else conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


def conv2d_resample(
    x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False
):
    r"""2D convolution with optional up/downsampling.
    Padding is performed only once at the beginning, not between the operations.
    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_height, in_width]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_height, kernel_width]`.
        f:              Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).
    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    # Validate arguments.
    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype == x.dtype)
    assert f is None or (
        isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32
    )
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    # assert isinstance(groups, int) and (groups >= 1), f"!!!!!! groups: {groups} isinstance(groups, int)  {isinstance(groups, int)} {type(groups)}"
    out_channels, in_channels_per_group, kh, kw = _get_weight_shape(w)
    fw, fh = _get_filter_size(f)
    # px0, px1, py0, py1 = _parse_padding(padding)
    px0, px1, py0, py1 = padding, padding, padding, padding

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d(
            x=x, f=f, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter
        )
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d(
            x=x,
            f=f,
            up=up,
            padding=[px0, px1, py0, py1],
            gain=up**2,
            flip_filter=flip_filter,
        )
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(
            x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight
        )
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(
                groups * in_channels_per_group, out_channels // groups, kh, kw
            )
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=[pyt, pxt],
            groups=groups,
            transpose=True,
            flip_weight=(not flip_weight),
        )
        x = upfirdn2d(
            x=x,
            f=f,
            padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
            gain=up**2,
            flip_filter=flip_filter,
        )
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(
                x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight
            )

    # Fallback: Generic reference implementation.
    x = upfirdn2d(
        x=x,
        f=(f if up > 1 else None),
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter,
    )
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x


class Conv2dLayer(torch.nn.Module):
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
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter(resample_filter))
        self.conv_clamp = conv_clamp
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.act_gain = activation_funcs[activation].def_gain

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(
            memory_format=memory_format
        )
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        out = bias_act(
            x, self.bias, act=self.activation, gain=act_gain, clamp=act_clamp
        )
        return out

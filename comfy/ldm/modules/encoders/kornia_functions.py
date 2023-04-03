

from typing import List, Tuple, Union

import torch
import torch.nn as nn

#from: https://github.com/kornia/kornia/blob/master/kornia/enhance/normalize.py

def enhance_normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    r"""Normalize an image/video tensor with mean and standard deviation.
    .. math::
        \text{input[channel] = (input[channel] - mean[channel]) / std[channel]}
    Where `mean` is :math:`(M_1, ..., M_n)` and `std` :math:`(S_1, ..., S_n)` for `n` channels,
    Args:
        data: Image tensor of size :math:`(B, C, *)`.
        mean: Mean for each channel.
        std: Standard deviations for each channel.
    Return:
        Normalised tensor with same size as input :math:`(B, C, *)`.
    Examples:
        >>> x = torch.rand(1, 4, 3, 3)
        >>> out = normalize(x, torch.tensor([0.0]), torch.tensor([255.]))
        >>> out.shape
        torch.Size([1, 4, 3, 3])
        >>> x = torch.rand(1, 4, 3, 3)
        >>> mean = torch.zeros(4)
        >>> std = 255. * torch.ones(4)
        >>> out = normalize(x, mean, std)
        >>> out.shape
        torch.Size([1, 4, 3, 3])
    """
    shape = data.shape
    if len(mean.shape) == 0 or mean.shape[0] == 1:
        mean = mean.expand(shape[1])
    if len(std.shape) == 0 or std.shape[0] == 1:
        std = std.expand(shape[1])

    # Allow broadcast on channel dimension
    if mean.shape and mean.shape[0] != 1:
        if mean.shape[0] != data.shape[1] and mean.shape[:2] != data.shape[:2]:
            raise ValueError(f"mean length and number of channels do not match. Got {mean.shape} and {data.shape}.")

    # Allow broadcast on channel dimension
    if std.shape and std.shape[0] != 1:
        if std.shape[0] != data.shape[1] and std.shape[:2] != data.shape[:2]:
            raise ValueError(f"std length and number of channels do not match. Got {std.shape} and {data.shape}.")

    mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)
    std = torch.as_tensor(std, device=data.device, dtype=data.dtype)

    if mean.shape:
        mean = mean[..., :, None]
    if std.shape:
        std = std[..., :, None]

    out: torch.Tensor = (data.view(shape[0], shape[1], -1) - mean) / std

    return out.view(shape)

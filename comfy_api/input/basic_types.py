import torch
from typing import TypedDict

ImageInput = torch.Tensor
"""
An image in format [B, H, W, C] where B is the batch size, C is the number of channels,
"""

class AudioInput(TypedDict):
    """
    TypedDict representing audio input.
    """

    waveform: torch.Tensor
    """
    Tensor in the format [B, C, T] where B is the batch size, C is the number of channels,
    """

    sample_rate: int


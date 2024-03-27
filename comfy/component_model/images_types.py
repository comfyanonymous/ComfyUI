from typing import NamedTuple

from torch import Tensor


class RgbMaskTuple(NamedTuple):
    rgb: Tensor
    mask: Tensor

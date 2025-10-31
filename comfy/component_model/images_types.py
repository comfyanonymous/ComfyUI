from typing import NamedTuple

from .tensor_types import ImageBatch, MaskBatch


class ImageMaskTuple(NamedTuple):
    image: ImageBatch
    mask: MaskBatch

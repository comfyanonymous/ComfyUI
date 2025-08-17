"""
@author: chrisgoringe
@title: Image Filter
@nickname: Image Filter
@description: A custom node that pauses the flow while you choose which image or images to pass on to the rest of the workflow. Simplified and improved version of cg-image-picker.
"""

from .image_filter import ImageFilter, MaskImageFilter, TextImageFilterWithExtras
from .list_utility_nodes import PickFromList, BatchFromImageList, ImageListFromBatch, StringListFromStrings
from .string_utility_nodes import SplitByCommas, StringToFloat, StringToInt, AnyListToString
from .mask_utility_nodes import MaskedSection

VERSION = "1.6.1"
WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS= {
    "Image Filter": ImageFilter,
    "Text Image Filter": TextImageFilterWithExtras,
    "Text Image Filter with Extras": TextImageFilterWithExtras,
    "Mask Image Filter": MaskImageFilter,
    "Split String by Commas": SplitByCommas,
    "String to Int": StringToInt,
    "String to Float": StringToFloat,
    "Pick from List": PickFromList,
    "Any List to String": AnyListToString,
    "String List from Strings": StringListFromStrings,
    "Batch from Image List": BatchFromImageList,
    "Image List From Batch": ImageListFromBatch,
    "Masked Section": MaskedSection,
}

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]

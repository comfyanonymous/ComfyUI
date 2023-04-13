import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import comfy.utils

class ImageToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "image": ("IMAGE",),
                    "channel": (["red", "green", "blue"],),
                }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("MASK",)
    FUNCTION = "image_to_mask"

    def image_to_mask(self, image, channel):
        channels = ["red", "green", "blue"]
        mask = image[0, :, :, channels.index(channel)]
        return (mask,)

class MaskToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask):
        result = mask[None, :, :, None].expand(-1, -1, -1, 3)
        return (result,)
        
NODE_CLASS_MAPPINGS = {
    "ImageToMask": ImageToMask,
    "MaskToImage": MaskToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToMask": "Convert Image to Mask",
    "MaskToImage": "Convert Mask to Image",
}

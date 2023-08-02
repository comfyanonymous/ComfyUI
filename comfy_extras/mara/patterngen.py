# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Simple image pattern generators.
'''
import os

import numpy as np
from PIL import Image
import torch

MAX_RESOLUTION = 8192

class ImageSolidColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 64, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 64, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"

    CATEGORY = "image/pattern"

    def render(self, width, height, r, g, b):
        color = torch.tensor([r, g, b]) / 255.0
        result = color.expand(1, height, width, 3)
        return (result, )


NODE_CLASS_MAPPINGS = {
    "ImageSolidColor": ImageSolidColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSolidColor": "Solid Color",
}


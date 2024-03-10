#From https://github.com/kornia/kornia
import math

import torch
import torch.nn.functional as F
import comfy.model_management

from kornia.filters import canny


class Canny:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                                "low_threshold": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 0.99, "step": 0.01}),
                                "high_threshold": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 0.99, "step": 0.01})
                                }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "detect_edge"

    CATEGORY = "image/preprocessors"

    def detect_edge(self, image, low_threshold, high_threshold):
        output = canny(image.to(comfy.model_management.get_torch_device()).movedim(-1, 1), low_threshold, high_threshold)
        img_out = output[1].to(comfy.model_management.intermediate_device()).repeat(1, 3, 1, 1).movedim(1, -1)
        return (img_out,)

NODE_CLASS_MAPPINGS = {
    "Canny": Canny,
}

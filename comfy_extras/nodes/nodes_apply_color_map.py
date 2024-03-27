import cv2
import numpy as np
import torch
from torch import Tensor

from comfy.nodes.package_typing import CustomNode, InputTypes, ValidatedNodeResult

_available_colormaps = ["Grayscale"] + [attr for attr in dir(cv2) if attr.startswith('COLORMAP')]


class ImageApplyColorMap(CustomNode):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypes:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "colormap": (_available_colormaps, {"default": "COLORMAP_INFERNO"}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.001, "round": 0.001}),
                "min_depth": ("FLOAT", {"default": 0.001, "min": 0.001, "round": 0.00001, "step": 0.001}),
                "max_depth": ("FLOAT", {"default": 1e2, "round": 0.00001, "step": 0.1}),
                "one_minus": ("BOOLEAN", {"default": False}),
                "clip_min": ("BOOLEAN", {"default": True}),
                "clip_max": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "image/postprocessing"
    FUNCTION = "execute"

    def execute(self,
                image: Tensor,
                gamma: float = 1.0,
                min_depth: float = 0.001,
                max_depth: float = 1e3,
                colormap: str = "COLORMAP_INFERNO",
                one_minus: bool = False,
                clip_min: bool = True,
                clip_max: bool = False,
                ) -> ValidatedNodeResult:
        """
        Invert and apply a colormap to a batch of absolute distance depth images.

        For Zoe and Midas, set colormap to be `COLORMAP_INFERNO`. Diffusers Depth expects `Grayscale`.

        As per https://huggingface.co/SargeZT/controlnet-v1e-sdxl-depth/discussions/7, some ControlNet checkpoints
        expect one_minus to be true.
        """
        colored_images = []

        for i in range(image.shape[0]):
            depth_image = image[i, :, :, 0].numpy()
            depth_image = np.where(depth_image <= min_depth, np.nan if not clip_min else min_depth, depth_image)
            if clip_max:
                depth_image = np.where(depth_image >= max_depth, max_depth, depth_image)
            depth_image = np.power(depth_image, 1.0 / gamma)
            inv_depth_image = 1.0 / depth_image

            xp = [1.0 / max_depth, 1.0 / min_depth]
            fp = [0, 1]
            normalized_depth = np.interp(inv_depth_image, xp, fp, left=0, right=1)
            normalized_depth = np.nan_to_num(normalized_depth, nan=0)

            normalized_depth_uint8 = (normalized_depth * 255).astype(np.uint8)
            if one_minus:
                normalized_depth_uint8 = 255 - normalized_depth_uint8
            if colormap == "Grayscale":
                colored_image = normalized_depth_uint8
            else:
                cv2_colormap = getattr(cv2, colormap)
                colored_image = cv2.applyColorMap(normalized_depth_uint8, cv2_colormap)
            colored_image_rgb = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
            rgb_tensor = torch.tensor(colored_image_rgb) * 1.0 / 255.0
            colored_images.append(rgb_tensor)

        return torch.stack(colored_images),


NODE_CLASS_MAPPINGS = {
    ImageApplyColorMap.__name__: ImageApplyColorMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    ImageApplyColorMap.__name__: "Apply ColorMap to Image (CV2)",
}

import json
import os
from typing import Literal, Tuple

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy import utils
from comfy.cli_args import args
from comfy.cmd import folder_paths
from comfy.component_model.tensor_types import ImageBatch, RGBImageBatch
from comfy.nodes.base_nodes import ImageScale
from comfy.nodes.common import MAX_RESOLUTION
from comfy.nodes.package_typing import CustomNode
from comfy_extras.constants.resolutions import SDXL_SD3_FLUX_RESOLUTIONS, LTVX_RESOLUTIONS, SD_RESOLUTIONS, \
    IDEOGRAM_RESOLUTIONS, COSMOS_RESOLUTIONS


def levels_adjustment(image: ImageBatch, black_level: float = 0.0, mid_level: float = 0.5, white_level: float = 1.0, clip: bool = True) -> ImageBatch:
    """
    Apply a levels adjustment to an sRGB image.

    Args:
    image (torch.Tensor): Input image tensor of shape (B, H, W, C) with values in range [0, 1]
    black_level (float): Black point (default: 0.0)
    mid_level (float): Midtone point (default: 0.5)
    white_level (float): White point (default: 1.0)
    clip (bool): Whether to clip the output values to [0, 1] range (default: True)

    Returns:
    torch.Tensor: Adjusted image tensor of shape (B, H, W, C)
    """
    # Ensure input is in correct shape and range
    assert image.dim() == 4 and image.shape[-1] == 3, "Input should be of shape (B, H, W, 3)"
    assert 0 <= black_level < mid_level < white_level <= 1, "Levels should be in ascending order in range [0, 1]"

    def srgb_to_linear(x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(x):
        return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)

    linear = srgb_to_linear(image)

    adjusted = (linear - black_level) / (white_level - black_level)

    power_factor = torch.log2(torch.tensor(0.5, device=image.device)) / torch.log2(torch.tensor(mid_level, device=image.device))

    # apply power function to avoid nans
    adjusted = torch.where(adjusted > 0, torch.pow(adjusted.clamp(min=1e-8), power_factor), adjusted)

    result = linear_to_srgb(adjusted)

    if clip:
        result = torch.clamp(result, 0.0, 1.0)

    return result


class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                             "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                             "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                             "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop"

    CATEGORY = "image/transform"

    def crop(self, image, width, height, x, y):
        x = min(x, image.shape[2] - 1)
        y = min(y, image.shape[1] - 1)
        to_x = width + x
        to_y = height + y
        img = image[:, y:to_y, x:to_x, :]
        return (img,)


class RepeatImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"

    CATEGORY = "image/batch"

    def repeat(self, image, amount):
        s = image.repeat((amount, 1, 1, 1))
        return (s,)


class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "batch_index": ("INT", {"default": 0, "min": 0, "max": 4095}),
                             "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "frombatch"

    CATEGORY = "image/batch"

    def frombatch(self, image, batch_index, length):
        s_in = image
        batch_index = min(s_in.shape[0] - 1, batch_index)
        length = min(s_in.shape[0] - batch_index, length)
        s = s_in[batch_index:batch_index + length].clone()
        return (s,)


class SaveAnimatedWEBP:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    methods = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "lossless": ("BOOLEAN", {"default": True}),
                     "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                     "method": (list(s.methods.keys()),),
                     # "num_frames": ("INT", {"default": 0, "min": 0, "max": 8192}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/animation"

    def save_images(self, images, fps, filename_prefix, lossless, quality, method, num_frames=0, prompt=None, extra_pnginfo=None):
        method = self.methods.get(method)
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        pil_images = []
        for image in images:
            i = 255. * image.float().cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        if num_frames == 0:
            num_frames = len(pil_images)

        c = len(pil_images)
        for i in range(0, c, num_frames):
            file = f"{filename}_{counter:05}_.webp"
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0 / fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return {"ui": {"images": results, "animated": (animated,)}}


class SaveAnimatedPNG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                     "compress_level": ("INT", {"default": 4, "min": 0, "max": 9})
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image/animation"

    def save_images(self, images, fps, compress_level, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        pil_images = []
        for image in images:
            i = 255. * image.float().cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if prompt is not None:
                metadata.add(b"comf", "prompt".encode("latin-1", "strict") + b"\0" + json.dumps(prompt).encode("latin-1", "strict"), after_idat=True)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add(b"comf", x.encode("latin-1", "strict") + b"\0" + json.dumps(extra_pnginfo[x]).encode("latin-1", "strict"), after_idat=True)

        file = f"{filename}_{counter:05}_.png"
        pil_images[0].save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level, save_all=True, duration=int(1000.0 / fps), append_images=pil_images[1:])
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })

        return {"ui": {"images": results, "animated": (True,)}}


class ImageShape:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "image_width_height"

    CATEGORY = "image/operations"

    def image_width_height(self, image: ImageBatch):
        shape = image.shape
        return shape[2], shape[1]


class ImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["cover", "contain", "auto"], {"default": "cover"}),
                "resolutions": (["SDXL/SD3/Flux", "SD1.5", "LTXV", "Ideogram", "Cosmos"], {"default": "SDXL/SD3/Flux"}),
                "interpolation": (ImageScale.upscale_methods, {"default": "bilinear"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"

    def resize_image(self, image: RGBImageBatch, resize_mode: Literal["cover", "contain", "auto"], resolutions: Literal["SDXL/SD3/Flux", "SD1.5"], interpolation: str) -> tuple[RGBImageBatch]:
        resolutions = resolutions.lower()
        if resolutions == "sdxl/sd3/flux":
            supported_resolutions = SDXL_SD3_FLUX_RESOLUTIONS
        elif resolutions == "ltxv":
            supported_resolutions = LTVX_RESOLUTIONS
        elif resolutions == "ideogram":
            supported_resolutions = IDEOGRAM_RESOLUTIONS
        elif resolutions == "cosmos":
            supported_resolutions = COSMOS_RESOLUTIONS
        else:
            supported_resolutions = SD_RESOLUTIONS
        return self.resize_image_with_supported_resolutions(image, resize_mode, supported_resolutions, interpolation)

    def resize_image_with_supported_resolutions(self, image: RGBImageBatch, resize_mode: Literal["cover", "contain", "auto"], supported_resolutions: list[tuple[int, int]], interpolation: str) -> tuple[RGBImageBatch]:
        resized_images = []
        for img in image:
            h, w = img.shape[:2]
            current_aspect_ratio = w / h
            target_resolution = min(supported_resolutions,
                                    key=lambda res: abs(res[0] / res[1] - current_aspect_ratio))

            if resize_mode == "cover":
                scale = max(target_resolution[0] / w, target_resolution[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
            elif resize_mode == "contain":
                scale = min(target_resolution[0] / w, target_resolution[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
            else:  # auto
                if current_aspect_ratio > target_resolution[0] / target_resolution[1]:
                    new_w, new_h = target_resolution[0], int(h * target_resolution[0] / w)
                else:
                    new_w, new_h = int(w * target_resolution[1] / h), target_resolution[1]

            # convert to b, c, h, w
            img_tensor = img.permute(2, 0, 1).unsqueeze(0)

            # Use common_upscale for resizing
            resized = utils.common_upscale(img_tensor, new_w, new_h, interpolation, "disabled")

            # handle padding or cropping
            if resize_mode == "contain":
                canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                y1 = (target_resolution[1] - new_h) // 2
                x1 = (target_resolution[0] - new_w) // 2
                canvas[:, :, y1:y1 + new_h, x1:x1 + new_w] = resized
                resized = canvas
            elif resize_mode == "cover":
                y1 = (new_h - target_resolution[1]) // 2
                x1 = (new_w - target_resolution[0]) // 2
                resized = resized[:, :, y1:y1 + target_resolution[1], x1:x1 + target_resolution[0]]
            else:  # auto
                if new_w != target_resolution[0] or new_h != target_resolution[1]:
                    canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                    y1 = (target_resolution[1] - new_h) // 2
                    x1 = (target_resolution[0] - new_w) // 2
                    canvas[:, :, y1:y1 + new_h, x1:x1 + new_w] = resized
                    resized = canvas

            resized_images.append(resized.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0))

        return (torch.stack(resized_images),)


class ImageResize1(ImageResize):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["cover", "contain", "auto"], {"default": "cover"}),
                "width": ("INT", {"min": 1}),
                "height": ("INT", {"min": 1}),
                "interpolation": (ImageScale.upscale_methods, {"default": "bilinear"}),
            }
        }

    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE",)
    def execute(self, image: RGBImageBatch, resize_mode: Literal["cover", "contain", "auto"], width: int, height: int, interpolation: str) -> tuple[RGBImageBatch]:
        return self.resize_image_with_supported_resolutions(image, resize_mode, [(width, height)], interpolation)


class ImageLevels(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "black_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mid_level": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "white_level": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_levels"
    CATEGORY = "image/adjust"

    def apply_levels(self, image: ImageBatch, black_level: float, mid_level: float, white_level: float, clip: bool) -> Tuple[ImageBatch]:
        adjusted_image = levels_adjustment(image, black_level, mid_level, white_level, clip)
        return (adjusted_image,)


class ImageLuminance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute_luminance"
    CATEGORY = "image/color"

    def compute_luminance(self, image: ImageBatch) -> Tuple[ImageBatch]:
        assert image.dim() == 4 and image.shape[-1] == 3, "Input should be of shape (B, H, W, 3)"

        # define srgb luminance coefficients
        coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)

        luminance = torch.sum(image * coeffs, dim=-1, keepdim=True)
        luminance = luminance.expand(-1, -1, -1, 3)

        return (luminance,)


NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResize,
    "ImageResize1": ImageResize1,
    "ImageShape": ImageShape,
    "ImageCrop": ImageCrop,
    "ImageLevels": ImageLevels,
    "ImageLuminance": ImageLuminance,
    "RepeatImageBatch": RepeatImageBatch,
    "ImageFromBatch": ImageFromBatch,
    "SaveAnimatedWEBP": SaveAnimatedWEBP,
    "SaveAnimatedPNG": SaveAnimatedPNG,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "Fit Image to Diffusion Size",
    "ImageResize1": "Fit Image to Width Height"
}

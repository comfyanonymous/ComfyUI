import json
import numpy as np
import os
import re
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from inspect import cleandoc
from io import BytesIO
from typing import Literal, Tuple

from comfy import utils
from comfy.cli_args import args
from comfy.cmd import folder_paths
from comfy.comfy_types import IO
from comfy.component_model.tensor_types import ImageBatch, RGBImageBatch
from comfy.execution_context import current_execution_context
from comfy.nodes.base_nodes import ImageScale
from comfy.nodes.common import MAX_RESOLUTION
from comfy.nodes.package_typing import CustomNode
from comfy.utils import common_upscale
from ..constants.resolutions import RESOLUTION_MAP, RESOLUTION_NAMES, SD_RESOLUTIONS


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


class ImageAddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                             "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_noise"
    CATEGORY = "image"

    def add_noise(self, image, seed, strength):
        generator = torch.manual_seed(seed)
        s = torch.clip((image + strength * torch.randn(image.size(), generator=generator, device="cpu").to(image)), min=0.0, max=1.0)
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
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_shape"
    CATEGORY = "image/operations"

    def get_shape(self, image: ImageBatch):
        shape = image.shape
        return shape[2], shape[1]


class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = (IO.INT, IO.INT, IO.INT)
    RETURN_NAMES = ("width", "height", "batch_size")
    FUNCTION = "get_size"
    CATEGORY = "image"
    DESCRIPTION = """Returns width and height of the image, and passes it through unchanged."""

    def get_size(self, image, unique_id=None):
        height = image.shape[1]
        width = image.shape[2]
        batch_size = image.shape[0]

        if unique_id:
            server = current_execution_context().server
            server.send_progress_text(f"width: {width}, height: {height}\n batch size: {batch_size}", unique_id)

        return width, height, batch_size


class ImageResize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["cover", "contain", "auto"], {"default": "cover"}),
                "resolutions": (RESOLUTION_NAMES, {"default": RESOLUTION_NAMES[0]}),
                "interpolation": (ImageScale.upscale_methods, {"default": "lanczos"}),
            },
            "optional": {
                "aspect_ratio_tolerance": ("FLOAT", {"min": 0, "max": 1.0, "default": 0.05, "step": 0.001})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"

    def resize_image(self, image: ImageBatch, resize_mode: Literal["cover", "contain", "auto"], resolutions: str, interpolation: str, aspect_ratio_tolerance=0.05) -> tuple[RGBImageBatch]:
        supported_resolutions = RESOLUTION_MAP.get(resolutions, SD_RESOLUTIONS)
        return self.resize_image_with_supported_resolutions(image, resize_mode, supported_resolutions, interpolation, aspect_ratio_tolerance=aspect_ratio_tolerance)

    def resize_image_with_supported_resolutions(self, image: ImageBatch, resize_mode: Literal["cover", "contain", "auto"], supported_resolutions: list[tuple[int, int]], interpolation: str, aspect_ratio_tolerance=0.05) -> tuple[RGBImageBatch]:
        resized_images = []
        for img in image:
            h, w = img.shape[:2]
            current_aspect_ratio = w / h
            aspect_ratio_diffs = [(abs(res[0] / res[1] - current_aspect_ratio), res) for res in supported_resolutions]
            min_diff = min(aspect_ratio_diffs, key=lambda x: x[0])[0]
            close_enough_resolutions = [res for diff, res in aspect_ratio_diffs if diff <= min_diff + aspect_ratio_tolerance]
            target_resolution = max(close_enough_resolutions, key=lambda res: res[0] * res[1])

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

            img_tensor = img.permute(2, 0, 1).unsqueeze(0)
            resized = utils.common_upscale(img_tensor, new_w, new_h, interpolation, "disabled")

            if resize_mode == "contain":
                canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                y1, x1 = (target_resolution[1] - new_h) // 2, (target_resolution[0] - new_w) // 2
                canvas[:, :, y1:y1 + new_h, x1:x1 + new_w] = resized
                resized = canvas
            elif resize_mode == "cover":
                y1, x1 = (new_h - target_resolution[1]) // 2, (new_w - target_resolution[0]) // 2
                resized = resized[:, :, y1:y1 + target_resolution[1], x1:x1 + target_resolution[0]]
            else:  # auto
                if new_w != target_resolution[0] or new_h != target_resolution[1]:
                    canvas = torch.zeros((1, 3, target_resolution[1], target_resolution[0]), device=resized.device, dtype=resized.dtype)
                    y1, x1 = (target_resolution[1] - new_h) // 2, (target_resolution[0] - new_w) // 2
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
                "interpolation": (ImageScale.upscale_methods, {"default": "lanczos"}),
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


class ImageLuminance(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compute_luminance"
    CATEGORY = "image/color"

    def compute_luminance(self, image: ImageBatch) -> Tuple[ImageBatch]:
        assert image.dim() == 4 and image.shape[-1] == 3, "Input should be of shape (B, H, W, 3)"
        coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=image.device, dtype=image.dtype)
        luminance = torch.sum(image * coeffs, dim=-1, keepdim=True)
        luminance = luminance.expand(-1, -1, -1, 3)
        return (luminance,)


class SVG:
    """Stores SVG representations via a list of BytesIO objects."""

    def __init__(self, data: list[BytesIO]):
        self.data = data

    def combine(self, other: 'SVG') -> 'SVG':
        return SVG(self.data + other.data)

    @staticmethod
    def combine_all(svgs: list['SVG']) -> 'SVG':
        all_svgs_list: list[BytesIO] = []
        for svg_item in svgs:
            all_svgs_list.extend(svg_item.data)
        return SVG(all_svgs_list)


class ImageStitch:
    """Upstreamed from https://github.com/kijai/ComfyUI-KJNodes"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "direction": (["right", "down", "left", "up"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1024, "step": 2},
                ),
                "spacing_color": (
                    ["white", "black", "red", "green", "blue"],
                    {"default": "white"},
                ),
            },
            "optional": {
                "image2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stitch"
    CATEGORY = "image/transform"
    DESCRIPTION = """
Stitches image2 to image1 in the specified direction.
If image2 is not provided, returns image1 unchanged.
Optional spacing can be added between images.
"""

    def stitch(
            self,
            image1,
            direction,
            match_image_size,
            spacing_width,
            spacing_color,
            image2=None,
    ):
        if image2 is None:
            return (image1,)

        # Handle batch size differences
        if image1.shape[0] != image2.shape[0]:
            max_batch = max(image1.shape[0], image2.shape[0])
            if image1.shape[0] < max_batch:
                image1 = torch.cat(
                    [image1, image1[-1:].repeat(max_batch - image1.shape[0], 1, 1, 1)]
                )
            if image2.shape[0] < max_batch:
                image2 = torch.cat(
                    [image2, image2[-1:].repeat(max_batch - image2.shape[0], 1, 1, 1)]
                )

        # Match image sizes if requested
        if match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            aspect_ratio = w2 / h2

            if direction in ["left", "right"]:
                target_h, target_w = h1, int(h1 * aspect_ratio)
            else:  # up, down
                target_w, target_h = w1, int(w1 / aspect_ratio)

            image2 = common_upscale(
                image2.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
            ).movedim(1, -1)

        color_map = {
            "white": 1.0,
            "black": 0.0,
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
        }

        color_val = color_map[spacing_color]

        # When not matching sizes, pad to align non-concat dimensions
        if not match_image_size:
            h1, w1 = image1.shape[1:3]
            h2, w2 = image2.shape[1:3]
            pad_value = 0.0
            if not isinstance(color_val, tuple):
                pad_value = color_val

            if direction in ["left", "right"]:
                # For horizontal concat, pad heights to match
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 < target_h:
                        pad_h = target_h - h1
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
                    if h2 < target_h:
                        pad_h = target_h - h2
                        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, 0, 0, pad_top, pad_bottom), mode='constant', value=pad_value)
            else:  # up, down
                # For vertical concat, pad widths to match
                if w1 != w2:
                    target_w = max(w1, w2)
                    if w1 < target_w:
                        pad_w = target_w - w1
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image1 = torch.nn.functional.pad(image1, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)
                    if w2 < target_w:
                        pad_w = target_w - w2
                        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
                        image2 = torch.nn.functional.pad(image2, (0, 0, pad_left, pad_right), mode='constant', value=pad_value)

        # Ensure same number of channels
        if image1.shape[-1] != image2.shape[-1]:
            max_channels = max(image1.shape[-1], image2.shape[-1])
            if image1.shape[-1] < max_channels:
                image1 = torch.cat(
                    [
                        image1,
                        torch.ones(
                            *image1.shape[:-1],
                            max_channels - image1.shape[-1],
                            device=image1.device,
                        ),
                    ],
                    dim=-1,
                )
            if image2.shape[-1] < max_channels:
                image2 = torch.cat(
                    [
                        image2,
                        torch.ones(
                            *image2.shape[:-1],
                            max_channels - image2.shape[-1],
                            device=image2.device,
                        ),
                    ],
                    dim=-1,
                )

        # Add spacing if specified
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # Ensure even

            if direction in ["left", "right"]:
                spacing_shape = (
                    image1.shape[0],
                    max(image1.shape[1], image2.shape[1]),
                    spacing_width,
                    image1.shape[-1],
                )
            else:
                spacing_shape = (
                    image1.shape[0],
                    spacing_width,
                    max(image1.shape[2], image2.shape[2]),
                    image1.shape[-1],
                )

            spacing = torch.full(spacing_shape, 0.0, device=image1.device)
            if isinstance(color_val, tuple):
                for i, c in enumerate(color_val):
                    if i < spacing.shape[-1]:
                        spacing[..., i] = c
                if spacing.shape[-1] == 4:  # Add alpha
                    spacing[..., 3] = 1.0
            else:
                spacing[..., : min(3, spacing.shape[-1])] = color_val
                if spacing.shape[-1] == 4:
                    spacing[..., 3] = 1.0

        # Concatenate images
        images = [image2, image1] if direction in ["left", "up"] else [image1, image2]
        if spacing_width > 0:
            images.insert(1, spacing)

        concat_dim = 2 if direction in ["left", "right"] else 1
        return (torch.cat(images, dim=concat_dim),)


class ResizeAndPadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "step": 1
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": MAX_RESOLUTION,
                    "step": 1
                }),
                "padding_color": (["white", "black"],),
                "interpolation": (["area", "bicubic", "nearest-exact", "bilinear", "lanczos"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_and_pad"
    CATEGORY = "image/transform"

    def resize_and_pad(self, image, target_width, target_height, padding_color, interpolation):
        batch_size, orig_height, orig_width, channels = image.shape

        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        scale = min(scale_w, scale_h)

        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        image_permuted = image.permute(0, 3, 1, 2)

        resized = common_upscale(image_permuted, new_width, new_height, interpolation, "disabled")

        pad_value = 0.0 if padding_color == "black" else 1.0
        padded = torch.full(
            (batch_size, channels, target_height, target_width),
            pad_value,
            dtype=image.dtype,
            device=image.device
        )

        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2

        padded[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        output = padded.permute(0, 2, 3, 1)
        return (output,)


class SaveSVGNode:
    """Save SVG files on disk."""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    RETURN_TYPES = ()
    DESCRIPTION = cleandoc(__doc__ or "")
    FUNCTION = "save_svg"
    CATEGORY = "image/save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg": ("SVG",),
                "filename_prefix": ("STRING", {"default": "svg/ComfyUI", "tooltip": "The prefix for the file to save."})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    def save_svg(self, svg: SVG, filename_prefix="svg/ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        results = list()
        metadata_dict = {}
        if prompt is not None:
            metadata_dict["prompt"] = prompt
        if extra_pnginfo is not None:
            metadata_dict.update(extra_pnginfo)
        metadata_json = json.dumps(metadata_dict, indent=2) if metadata_dict else None

        for batch_number, svg_bytes in enumerate(svg.data):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.svg"
            svg_bytes.seek(0)
            svg_content = svg_bytes.read().decode('utf-8')

            if metadata_json:
                metadata_element = f'  <metadata><![CDATA[\n{metadata_json}\n]]></metadata>\n'
                svg_content = re.sub(r'(<svg[^>]*>)', lambda m: m.group(1) + '\n' + metadata_element, svg_content, flags=re.UNICODE)

            with open(os.path.join(full_output_folder, file), 'wb') as svg_file:
                svg_file.write(svg_content.encode('utf-8'))
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1
        return {"ui": {"images": results}}


class ImageRotate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": (IO.IMAGE,),
                              "rotation": (["none", "90 degrees", "180 degrees", "270 degrees"],),
                              }}
    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "rotate"

    CATEGORY = "image/transform"

    def rotate(self, image, rotation):
        rotate_by = 0
        if rotation.startswith("90"):
            rotate_by = 1
        elif rotation.startswith("180"):
            rotate_by = 2
        elif rotation.startswith("270"):
            rotate_by = 3

        image = torch.rot90(image, k=rotate_by, dims=[2, 1])
        return (image,)

class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": (IO.IMAGE,),
                              "flip_method": (["x-axis: vertically", "y-axis: horizontally"],),
                              }}
    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "flip"

    CATEGORY = "image/transform"

    def flip(self, image, flip_method):
        if flip_method.startswith("x"):
            image = torch.flip(image, dims=[1])
        elif flip_method.startswith("y"):
            image = torch.flip(image, dims=[2])

        return (image,)

class ImageScaleToMaxDimension:
    upscale_methods = ["area", "lanczos", "bilinear", "nearest-exact", "bilinear", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",),
                             "upscale_method": (s.upscale_methods,),
                             "largest_size": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, largest_size):
        height = image.shape[1]
        width = image.shape[2]

        if height > width:
            width = round((width / height) * largest_size)
            height = largest_size
        elif width > height:
            height = round((height / width) * largest_size)
            width = largest_size
        else:
            height = largest_size
            width = largest_size

        samples = image.movedim(-1, 1)
        s = common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return (s,)

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
    "ImageAddNoise": ImageAddNoise,
    "SaveSVGNode": SaveSVGNode,
    "ImageStitch": ImageStitch,
    "GetImageSize": GetImageSize,
    "ImageRotate": ImageRotate,
    "ImageFlip": ImageFlip,
    "ImageScaleToMaxDimension": ImageScaleToMaxDimension,
    "ResizeAndPadImage": ResizeAndPadImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "Fit Image to Diffusion Size",
    "ImageResize1": "Fit Image to Width Height",
    "ImageShape": "Get Image Shape",
    "GetImageSize": "Get Image Size",
    "ImageStitch": "Stitch Images",
    "SaveSVGNode": "Save SVG",
}

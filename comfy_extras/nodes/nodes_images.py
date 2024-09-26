import json
import os
from typing import Literal, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy.cli_args import args
from comfy.cmd import folder_paths
from comfy.component_model.tensor_types import ImageBatch
from comfy.nodes.common import MAX_RESOLUTION


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
                "resolutions": (["SDXL/SD3/Flux", "SD1.5", ], {"default": "SDXL/SD3/Flux"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"

    def resize_image(self, image: ImageBatch, resize_mode: Literal["cover", "contain", "auto"], resolutions: Literal["SDXL/SD3/Flux", "SD1.5",]) -> Tuple[ImageBatch]:
        if resolutions == "SDXL/SD3/Flux":
            supported_resolutions = [
                (640, 1536),
                (768, 1344),
                (832, 1216),
                (896, 1152),
                (1024, 1024),
                (1152, 896),
                (1216, 832),
                (1344, 768),
                (1536, 640),
            ]
        else:
            supported_resolutions = [
                (512, 512),
            ]

        resized_images = []
        for img in image:
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            h, w = img_np.shape[:2]
            current_aspect_ratio = w / h
            target_resolution = min(supported_resolutions,
                                    key=lambda res: abs(res[0] / res[1] - current_aspect_ratio))
            scale_w, scale_h = target_resolution[0] / w, target_resolution[1] / h

            if resize_mode == "cover":
                scale = max(scale_w, scale_h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                x1 = (new_w - target_resolution[0]) // 2
                y1 = (new_h - target_resolution[1]) // 2
                resized = resized[y1:y1 + target_resolution[1], x1:x1 + target_resolution[0]]
            elif resize_mode == "contain":
                scale = min(scale_w, scale_h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                canvas = np.zeros((target_resolution[1], target_resolution[0], 3), dtype=np.uint8)
                x1 = (target_resolution[0] - new_w) // 2
                y1 = (target_resolution[1] - new_h) // 2
                canvas[y1:y1 + new_h, x1:x1 + new_w] = resized
                resized = canvas
            else:
                if current_aspect_ratio > target_resolution[0] / target_resolution[1]:
                    scale = scale_w
                else:
                    scale = scale_h
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                if new_w > target_resolution[0] or new_h > target_resolution[1]:
                    x1 = (new_w - target_resolution[0]) // 2
                    y1 = (new_h - target_resolution[1]) // 2
                    resized = resized[y1:y1 + target_resolution[1], x1:x1 + target_resolution[0]]
                else:
                    canvas = np.zeros((target_resolution[1], target_resolution[0], 3), dtype=np.uint8)
                    x1 = (target_resolution[0] - new_w) // 2
                    y1 = (target_resolution[1] - new_h) // 2
                    canvas[y1:y1 + new_h, x1:x1 + new_w] = resized
                    resized = canvas

            resized_images.append(resized)

        return (torch.from_numpy(np.stack(resized_images)).float() / 255.0,)


NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResize,
    "ImageShape": ImageShape,
    "ImageCrop": ImageCrop,
    "RepeatImageBatch": RepeatImageBatch,
    "ImageFromBatch": ImageFromBatch,
    "SaveAnimatedWEBP": SaveAnimatedWEBP,
    "SaveAnimatedPNG": SaveAnimatedPNG,
}

from typing import Tuple
from math import ceil

import nodes
import folder_paths
from comfy.cli_args import args

from torch import Tensor
from torchvision.transforms.v2.functional import to_pil_image, to_image  # type: ignore
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from comfy.fonts import FontCollection, AnyFont

import numpy as np
import json
import os

MAX_RESOLUTION = nodes.MAX_RESOLUTION

class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
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
        img = image[:,y:to_y, x:to_x, :]
        return (img,)

class ImageLabel:
    fonts = FontCollection()

    @classmethod
    def INPUT_TYPES(s):
        font_names = list(s.fonts.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "font": (font_names, {"default": s.fonts.default_font_name}),
                "label": ("STRING", {"multiline": True}),
                "position": (["top", "bottom"],),
                "text_size": ("INT", {"default": 48, "min": 4}),
                "padding": ("INT", {"default": 24}),
                "line_spacing": ("INT", {"default": 5}),
                "text_color": ("STRING", {"default": "#fff"}),
                "background_color": ("STRING", {"default": "#000"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "label"

    CATEGORY = "image/transform"

    def label(
        self,
        image: Tensor,
        font: str,
        label: str,
        text_size: int,
        padding: int,
        line_spacing: int,
        position: str,
        text_color: str,
        background_color: str,
    ):
        """
        Extends an image at the top or bottom to add a label.

        Args:
            image (Tensor): The input image as a tensor with shape [1, H, W, C].
            font (str): The font name to be used for the label.
            label (str): The text of the label.
            text_size (int): The size of the label text in pixels.
            padding (int): Padding around the label text in pixels.
            line_spacing (int): Spacing between lines of the label.
            position (str): Position of the label, either 'top' or 'bottom'.
            text_color (str): Color of the label text as a hex reference.
            background_color (str): Background color of the label area as a hex reference.

        Returns:
            Tensor: The image with the label added, as a tensor with shape [1, H, W, C].

        Raises:
            ValueError: If an invalid position is provided.
        """

        original_image = to_pil_image(image.squeeze(0).permute(2, 0, 1))
        width, height = original_image.size
        font_obj: AnyFont = self.fonts[font].font_variant(size=text_size)

        _, label_height, text_size = self.calculate_label_dimensions(
            font_obj, label, text_size, line_spacing, padding, width
        )

        label_image = self.draw_label(
            font_obj, label, width, label_height, line_spacing, text_color, background_color
        )

        combined_image = Image.new("RGB", (width, height + label_height + line_spacing), (0, 0, 0))
        if position == "top":
            combined_image.paste(original_image, (0, label_height))
            combined_image.paste(label_image, (0, 0))
        elif position == "bottom":
            combined_image.paste(label_image, (0, height))
            combined_image.paste(original_image, (0, 0))
        else:
            raise ValueError(f"Unknown position: {position}")

        return (to_image(combined_image) / 255.0).permute(1, 2, 0)[None, None, ...]

    def calculate_label_dimensions(
        self, font: AnyFont, label: str, text_size: int, line_spacing: int, padding: int, max_width: float
    ) -> Tuple[int, int, int]:
        """
        Calculate the dimensions needed to draw a label within an image.

        This will reduce the font size where necessary to make the text fit.

        Args:
            font (AnyFont): The Pillow font to use.
            label (str): The text to calculate dimensions for.
            text_size (int): Starting font size for the label.
            line_spacing (int): Spacing between lines of text.
            padding (int): Padding around the text.
            max_width (float): Maximum allowed width for the text box.

        Returns:
            tuple[int, int, int]: The calculated width, height, and final font size.
        """

        while True:
            temp_image = Image.new("RGB", (1, 1))
            x1, y1, x2, y2 = ImageDraw.Draw(temp_image).textbbox(
                xy=(0, 0), text=label, font=font, spacing=line_spacing, align="center"
            )
            width = ceil(x2 - x1 + padding * 2)
            height = ceil(y2 - y1 + padding * 2)
            if width <= max_width:
                break
            text_size -= 1
            if text_size <= 8:
                break

        return width, height, text_size

    def draw_label(
        self,
        font: AnyFont,
        label: str,
        width: int,
        height: int,
        line_spacing: int,
        text_color: str,
        background_color: str,
    ) -> Image.Image:
        """
        Draws an image containing a label.

        Args:
            font (AnyFont): The Pillow font to use for text rendering.
            label (str): The text to use as the label.
            width (int): Width of the image in pixels.
            height (int): Height of the image in pixels.
            line_spacing (int): Spacing between lines of text.
            text_color (str): Color of the text as a hex reference.
            background_color (str): Background color of the image as a hex reference.

        Returns:
            Image: An image object with the label drawn on it.
        """

        image = Image.new("RGB", (width, height), background_color)
        draw = ImageDraw.Draw(image)
        draw.multiline_text(
            xy=(width / 2, height / 2),
            text=label,
            fill=text_color,
            font=font,
            anchor="mm",
            spacing=line_spacing,
            align="center",
        )
        return image

class RepeatImageBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
                              "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"

    CATEGORY = "image/batch"

    def repeat(self, image, amount):
        s = image.repeat((amount, 1,1,1))
        return (s,)

class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",),
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
                    {"images": ("IMAGE", ),
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
            i = 255. * image.cpu().numpy()
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
            pil_images[i].save(os.path.join(full_output_folder, file), save_all=True, duration=int(1000.0/fps), append_images=pil_images[i + 1:i + num_frames], exif=metadata, lossless=lossless, quality=quality, method=method)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        animated = num_frames != 1
        return { "ui": { "images": results, "animated": (animated,) } }

class SaveAnimatedPNG:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", ),
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
            i = 255. * image.cpu().numpy()
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
        pil_images[0].save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=compress_level, save_all=True, duration=int(1000.0/fps), append_images=pil_images[1:])
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })

        return { "ui": { "images": results, "animated": (True,)} }

NODE_CLASS_MAPPINGS = {
    "ImageCrop": ImageCrop,
    "ImageLabel": ImageLabel,
    "RepeatImageBatch": RepeatImageBatch,
    "ImageFromBatch": ImageFromBatch,
    "SaveAnimatedWEBP": SaveAnimatedWEBP,
    "SaveAnimatedPNG": SaveAnimatedPNG,
}

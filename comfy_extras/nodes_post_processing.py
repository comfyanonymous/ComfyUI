import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor
import re

import comfy.utils


class Blend:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

class Blur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur"

    CATEGORY = "image/postprocessing"

    def gaussian_kernel(self, kernel_size: int, sigma: float):
        x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size), torch.linspace(-1, 1, kernel_size), indexing="ij")
        d = torch.sqrt(x * x + y * y)
        g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
        return g / g.sum()

    def blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = self.gaussian_kernel(kernel_size, sigma).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        blurred = F.conv2d(image, kernel, padding=kernel_size // 2, groups=channels)
        blurred = blurred.permute(0, 2, 3, 1)

        return (blurred,)

class Quantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "dither": (["none", "floyd-steinberg"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"

    CATEGORY = "image/postprocessing"

    def quantize(self, image: torch.Tensor, colors: int = 256, dither: str = "FLOYDSTEINBERG"):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        dither_option = Image.Dither.FLOYDSTEINBERG if dither == "floyd-steinberg" else Image.Dither.NONE

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            palette = pil_image.quantize(colors=colors) # Required as described in https://github.com/python-pillow/Pillow/issues/5836
            quantized_image = pil_image.quantize(colors=colors, palette=palette, dither=dither_option)

            quantized_array = torch.tensor(np.array(quantized_image.convert("RGB"))).float() / 255
            result[b] = quantized_array

        return (result,)

class Sharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen_radius": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 31,
                    "step": 1
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "image/postprocessing"

    def sharpen(self, image: torch.Tensor, sharpen_radius: int, alpha: float):
        if sharpen_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape

        kernel_size = sharpen_radius * 2 + 1
        kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32) * -1
        center = kernel_size // 2
        kernel[center, center] = kernel_size**2
        kernel *= alpha
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return (result,)

class Transpose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": ([
                    "Flip horizontal",
                    "Flip vertical",
                    "Rotate 90°",
                    "Rotate 180°",
                    "Rotate 270°",
                    "Transpose",
                    "Transverse",
                ],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transpose"

    CATEGORY = "image/postprocessing"

    def transpose(self, image: torch.Tensor, method: str):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)
        if height != width and method in ("Transpose", "Transverse", "Rotate 90°", "Rotate 270°"):
            result = torch.permute(result, (0, 2, 1, 3))

        methods = {
            "Flip horizontal": (lambda x: torch.fliplr(x)),
            "Flip vertical": (lambda x: torch.flipud(x)),
            "Rotate 90°": (lambda x: torch.rot90(x)),
            "Rotate 180°": (lambda x: torch.rot90(x, 2)),
            "Rotate 270°": (lambda x: torch.rot90(x, 3)),
            "Transpose": (lambda x: torch.transpose(x, 0, 1)),
            "Transverse": (lambda x: torch.rot90(torch.transpose(x, 0, 1), 2)),
        }

        for b in range(batch_size):
            result[b] = methods[method](image[b])

        return (result,)

class Rotate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "angle": ("FLOAT", {
                    "default": 0, 
                    "min": 0,
                    "max": 360,
                    "step": 0.1
                }),
                "resample": ([
                    "Nearest Neighbor",
                    "Bilinear",
                    "Bicubic",
                ],),
                "expand": (["disabled", "enabled"],),
                "fill_color": ("STRING", {"default": "#000000"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"

    CATEGORY = "image/postprocessing"

    def rotate(self, image: torch.Tensor, angle: int, resample: str, expand: str, fill_color: str):
        batch_size, _, _, _ = image.shape

        resamplers = {
            "Nearest Neighbor": Image.Resampling.NEAREST,
            "Bilinear": Image.Resampling.BILINEAR,
            "Bicubic": Image.Resampling.BICUBIC,
        }

        tensor_image = image[0]
        img = (tensor_image * 255).to(torch.uint8).numpy()
        pil_image = Image.fromarray(img, mode='RGB')
        
        expand = True if expand == "enabled" else False
        fill_color = fill_color or "#000000"

        def parse_palette(color_str):
            if re.match(r'^#[a-fA-F0-9]{6}$', color_str) or color_str.lower() in ImageColor.colormap:
                return ImageColor.getrgb(color_str)

            color_rgb = re.match(r'^\(?(\d{1,3}),(\d{1,3}),(\d{1,3})\)?$', color_str)
            if color_rgb and int(color_rgb.group(1)) <= 255 and int(color_rgb.group(2)) <= 255 and int(color_rgb.group(3)) <= 255:
                return tuple(map(int, re.findall(r'\d{1,3}', color_str)))
            else:
                raise ValueError(f"Invalid color format: {color_str}")

        color = fill_color.replace(" ", "")
        color = parse_palette(color)
        rotated_image = pil_image.rotate(angle=angle, resample=resamplers[resample], expand=expand, fillcolor=color)
        height, width = rotated_image.size
        result = torch.zeros(batch_size, height, width, 3)
        rotated_array = torch.tensor(np.array(rotated_image.convert("RGB"))).float() / 255
        result[0] = rotated_array

        return (result,)

class GetChannel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": ([
                    "Red",
                    "Green",
                    "Blue",
                ],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "getchannel"

    CATEGORY = "image/postprocessing"

    def getchannel(self, image: torch.Tensor, channel: str):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            output_image = pil_image.getchannel(channel[0])

            output_array = torch.tensor(np.array(output_image.convert("RGB"))).float() / 255
            result[b] = output_array

        return (result,)

class Split:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Red", "Green", "Blue")

    FUNCTION = "split"

    CATEGORY = "image/postprocessing"

    def split(self, image: torch.Tensor):
        batch_size, height, width, _ = image.shape
        result_r = torch.zeros_like(image)
        result_g = torch.zeros_like(image)
        result_b = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b]
            img = (tensor_image * 255).to(torch.uint8).numpy()
            pil_image = Image.fromarray(img, mode='RGB')

            output_r, output_g, output_b = pil_image.split()

            output_array_r = torch.tensor(np.array(output_r.convert("RGB"))).float() / 255
            output_array_g = torch.tensor(np.array(output_g.convert("RGB"))).float() / 255
            output_array_b = torch.tensor(np.array(output_b.convert("RGB"))).float() / 255
            result_r[b], result_g[b], result_b[b] = output_array_r, output_array_g, output_array_b

        return (result_r, result_g, result_b)

class Merge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "red": ("IMAGE",),
                "green": ("IMAGE",),
                "blue": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"

    CATEGORY = "image/postprocessing"

    def merge(self, red: torch.Tensor, green: torch.Tensor, blue: torch.Tensor):
        batch_size, height, width, _ = red.shape
        result = torch.zeros_like(red)

        for b in range(batch_size):
            img_r = (red[b] * 255).to(torch.uint8).numpy()
            img_g = (green[b] * 255).to(torch.uint8).numpy()
            img_b = (blue[b] * 255).to(torch.uint8).numpy()
            pil_image_r = Image.fromarray(img_r, mode='RGB').convert("L")
            pil_image_g = Image.fromarray(img_g, mode='RGB').convert("L")
            pil_image_b = Image.fromarray(img_b, mode='RGB').convert("L")

            output_image = Image.merge("RGB", (pil_image_r, pil_image_g, pil_image_b))

            output_array = torch.tensor(np.array(output_image.convert("RGB"))).float() / 255
            result[b] = output_array

        return (result,)

class Composite:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"

    CATEGORY = "image/postprocessing"

    def composite(self, image_a: torch.Tensor, image_b: torch.Tensor, mask: torch.Tensor):
        batch_size, height, width, _ = image_a.shape
        result = torch.zeros_like(image_a)

        for b in range(batch_size):
            img_a = (image_a[b] * 255).to(torch.uint8).numpy()
            img_b = (image_b[b] * 255).to(torch.uint8).numpy()
            img_mask = (mask * 255).to(torch.uint8).numpy()
            pil_image_a = Image.fromarray(img_a, mode='RGB')
            pil_image_b = Image.fromarray(img_b, mode='RGB')
            pil_image_mask = Image.fromarray(img_mask, mode='L')

            output_image = Image.composite(pil_image_a, pil_image_b, pil_image_mask)

            output_array = torch.tensor(np.array(output_image.convert("RGB"))).float() / 255
            result[b] = output_array

        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageBlend": Blend,
    "ImageBlur": Blur,
    "ImageQuantize": Quantize,
    "ImageSharpen": Sharpen,
    "ImageTranspose": Transpose,
    "ImageRotate": Rotate,
    "ImageGetChannel": GetChannel,
    "ImageSplit": Split,
    "ImageMerge": Merge,
    "ImageComposite": Composite,
}

NODE_DISPLAY_NAME_MAPPINGS  = {
    "ImageBlend": "Blend Images",
    "ImageBlur": "Blur Image",
    "ImageQuantize": "Quantize Image",
    "ImageSharpen": "Sharpen Image",
    "ImageTranspose": "Transpose",
    "ImageRotate": "Rotate",
    "ImageGetChannel": "Extract Channel",
    "ImageSplit": "Split Channels",
    "ImageMerge": "Merge Channels",
    "ImageComposite": "Composite Images",
}

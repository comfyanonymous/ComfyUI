import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math

import comfy.utils
import comfy.model_management


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
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        image2 = image2.to(image1.device)
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
        elif mode == "difference":
            return img1 - img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))

def gaussian_kernel(kernel_size: int, sigma: float, device=None):
    x, y = torch.meshgrid(torch.linspace(-1, 1, kernel_size, device=device), torch.linspace(-1, 1, kernel_size, device=device), indexing="ij")
    d = torch.sqrt(x * x + y * y)
    g = torch.exp(-(d * d) / (2.0 * sigma * sigma))
    return g / g.sum()

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

    def blur(self, image: torch.Tensor, blur_radius: int, sigma: float):
        if blur_radius == 0:
            return (image,)

        image = image.to(comfy.model_management.get_torch_device())
        batch_size, height, width, channels = image.shape

        kernel_size = blur_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device).repeat(channels, 1, 1).unsqueeze(1)

        image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        padded_image = F.pad(image, (blur_radius,blur_radius,blur_radius,blur_radius), 'reflect')
        blurred = F.conv2d(padded_image, kernel, padding=kernel_size // 2, groups=channels)[:,:,blur_radius:-blur_radius, blur_radius:-blur_radius]
        blurred = blurred.permute(0, 2, 3, 1)

        return (blurred.to(comfy.model_management.intermediate_device()),)

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
                "dither": (["none", "floyd-steinberg", "bayer-2", "bayer-4", "bayer-8", "bayer-16"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "quantize"

    CATEGORY = "image/postprocessing"

    def bayer(im, pal_im, order):
        def normalized_bayer_matrix(n):
            if n == 0:
                return np.zeros((1,1), "float32")
            else:
                q = 4 ** n
                m = q * normalized_bayer_matrix(n - 1)
                return np.bmat(((m-1.5, m+0.5), (m+1.5, m-0.5))) / q

        num_colors = len(pal_im.getpalette()) // 3
        spread = 2 * 256 / num_colors
        bayer_n = int(math.log2(order))
        bayer_matrix = torch.from_numpy(spread * normalized_bayer_matrix(bayer_n) + 0.5)

        result = torch.from_numpy(np.array(im).astype(np.float32))
        tw = math.ceil(result.shape[0] / bayer_matrix.shape[0])
        th = math.ceil(result.shape[1] / bayer_matrix.shape[1])
        tiled_matrix = bayer_matrix.tile(tw, th).unsqueeze(-1)
        result.add_(tiled_matrix[:result.shape[0],:result.shape[1]]).clamp_(0, 255)
        result = result.to(dtype=torch.uint8)

        im = Image.fromarray(result.cpu().numpy())
        im = im.quantize(palette=pal_im, dither=Image.Dither.NONE)
        return im

    def quantize(self, image: torch.Tensor, colors: int, dither: str):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            im = Image.fromarray((image[b] * 255).to(torch.uint8).numpy(), mode='RGB')

            pal_im = im.quantize(colors=colors) # Required as described in https://github.com/python-pillow/Pillow/issues/5836

            if dither == "none":
                quantized_image = im.quantize(palette=pal_im, dither=Image.Dither.NONE)
            elif dither == "floyd-steinberg":
                quantized_image = im.quantize(palette=pal_im, dither=Image.Dither.FLOYDSTEINBERG)
            elif dither.startswith("bayer"):
                order = int(dither.split('-')[-1])
                quantized_image = Quantize.bayer(im, pal_im, order)

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
                "sigma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    CATEGORY = "image/postprocessing"

    def sharpen(self, image: torch.Tensor, sharpen_radius: int, sigma: float, alpha: float):
        if sharpen_radius == 0:
            return (image,)

        batch_size, height, width, channels = image.shape
        if not comfy.model_management.mps_mode():
            image = image.to(comfy.model_management.get_torch_device())

        kernel_size = sharpen_radius * 2 + 1
        kernel = gaussian_kernel(kernel_size, sigma, device=image.device) * -(alpha*10)
        center = kernel_size // 2
        kernel[center, center] = kernel[center, center] - kernel.sum() + 1.0
        kernel = kernel.repeat(channels, 1, 1).unsqueeze(1)

        tensor_image = image.permute(0, 3, 1, 2) # Torch wants (B, C, H, W) we use (B, H, W, C)
        tensor_image = F.pad(tensor_image, (sharpen_radius,sharpen_radius,sharpen_radius,sharpen_radius), 'reflect')
        sharpened = F.conv2d(tensor_image, kernel, padding=center, groups=channels)[:,:,sharpen_radius:-sharpen_radius, sharpen_radius:-sharpen_radius]
        sharpened = sharpened.permute(0, 2, 3, 1)

        result = torch.clamp(sharpened, 0, 1)

        return (result,) if comfy.model_management.mps_mode() else (result.to(comfy.model_management.intermediate_device()),)

class ImageScaleToTotalPixels:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "image": ("IMAGE",), "upscale_method": (s.upscale_methods,),
                              "megapixels": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),
                            }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, image, upscale_method, megapixels):
        samples = image.movedim(-1,1)
        total = int(megapixels * 1024 * 1024)

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1,-1)
        return (s,)

NODE_CLASS_MAPPINGS = {
    "ImageBlend": Blend,
    "ImageBlur": Blur,
    "ImageQuantize": Quantize,
    "ImageSharpen": Sharpen,
    "ImageScaleToTotalPixels": ImageScaleToTotalPixels,
}

import numpy as np
import cv2
import torch
from PIL import Image, ImageEnhance

class Dither:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bits": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dither"

    CATEGORY = "postprocessing"

    def dither(self, image: torch.Tensor, bits: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            img = (tensor_image * 255)
            height, width, _ = img.shape

            scale = 255 / (2**bits - 1)

            for y in range(height):
                for x in range(width):
                    old_pixel = img[y, x].copy()
                    new_pixel = np.round(old_pixel / scale) * scale
                    img[y, x] = new_pixel

                    quant_error = old_pixel - new_pixel

                    if x + 1 < width:
                        img[y, x + 1] += quant_error * 7 / 16
                    if y + 1 < height:
                        if x - 1 >= 0:
                            img[y + 1, x - 1] += quant_error * 3 / 16
                        img[y + 1, x] += quant_error * 5 / 16
                        if x + 1 < width:
                            img[y + 1, x + 1] += quant_error * 1 / 16

            dithered = img / 255
            tensor = torch.from_numpy(dithered).unsqueeze(0)
            result[b] = tensor

        return (result,)

class KMeansQuantize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "colors": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 256,
                    "step": 1
                }),
                "precision": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "kmeans_quantize"

    CATEGORY = "postprocessing"

    def kmeans_quantize(self, image: torch.Tensor, colors: int, precision: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy().astype(np.float32)
            img = tensor_image

            height, width, c = img.shape

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                precision * 5, 0.01
            )

            img_copy = img.reshape(-1, c)
            _, label, center = cv2.kmeans(
                img_copy, colors, None,
                criteria, 1, cv2.KMEANS_PP_CENTERS
            )

            img = center[label.flatten()].reshape(*img.shape)
            tensor = torch.from_numpy(img).unsqueeze(0)
            result[b] = tensor

        return (result,)

class GaussianBlur:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {
                    "default": 5,
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

    CATEGORY = "postprocessing"

    def blur(self, image: torch.Tensor, kernel_size: int, sigma: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            blurred = cv2.GaussianBlur(tensor_image, (kernel_size, kernel_size), sigma)
            tensor = torch.from_numpy(blurred).unsqueeze(0)
            result[b] = tensor

        return (result,)

class Sharpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {
                    "default": 5,
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

    CATEGORY = "postprocessing"

    def sharpen(self, image: torch.Tensor, kernel_size: int, alpha: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) * -1
            center = kernel_size // 2
            kernel[center, center] = kernel_size**2
            kernel *= alpha

            sharpened = cv2.filter2D(tensor_image, -1, kernel)

            tensor = torch.from_numpy(sharpened).unsqueeze(0)
            tensor = torch.clamp(tensor, 0, 1)
            result[b] = tensor

        return (result,)

class CannyEdgeDetection:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lower_threshold": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
                "upper_threshold": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "step": 10
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "canny"

    CATEGORY = "postprocessing"

    def canny(self, image: torch.Tensor, lower_threshold: int, upper_threshold: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros(batch_size, height, width)

        for b in range(batch_size):
            tensor_image = image[b].numpy().copy()
            gray_image = (cv2.cvtColor(tensor_image, cv2.COLOR_RGB2GRAY) * 255).astype(np.uint8)
            canny = cv2.Canny(gray_image, lower_threshold, upper_threshold)
            tensor = torch.from_numpy(canny)
            result[b] = tensor

        return (result,)

class ColorCorrect:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "hue": ("FLOAT", {
                    "default": 0,
                    "min": -90,
                    "max": 90,
                    "step": 5
                }),
                "brightness": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "contrast": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "saturation": ("FLOAT", {
                    "default": 0,
                    "min": -100,
                    "max": 100,
                    "step": 5
                }),
                "gamma": ("FLOAT", {
                    "default": 1,
                    "min": 0.2,
                    "max": 2.2,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_correct"

    CATEGORY = "postprocessing"

    def color_correct(self, image: torch.Tensor, temperature: float, hue: float, brightness: float, contrast: float, saturation: float, gamma: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            brightness /= 100
            contrast /= 100
            saturation /= 100
            temperature /= 100

            brightness = 1 + brightness
            contrast = 1 + contrast
            saturation = 1 + saturation

            modified_image = Image.fromarray((tensor_image * 255).astype(np.uint8))

            # brightness
            modified_image = ImageEnhance.Brightness(modified_image).enhance(brightness)

            # contrast
            modified_image = ImageEnhance.Contrast(modified_image).enhance(contrast)
            modified_image = np.array(modified_image).astype(np.float32)

            # temperature
            if temperature > 0:
                modified_image[:, :, 0] *= 1 + temperature
                modified_image[:, :, 1] *= 1 + temperature * 0.4
            elif temperature < 0:
                modified_image[:, :, 2] *= 1 - temperature
            modified_image = np.clip(modified_image, 0, 255)/255

            # gamma
            modified_image = np.clip(np.power(modified_image, gamma), 0, 1)

            # saturation
            hls_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HLS)
            hls_img[:, :, 2] = np.clip(saturation*hls_img[:, :, 2], 0, 1)
            modified_image = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB) * 255

            # hue
            hsv_img = cv2.cvtColor(modified_image, cv2.COLOR_RGB2HSV)
            hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue) % 360
            modified_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

            modified_image = modified_image.astype(np.uint8)
            modified_image = modified_image / 255
            modified_image = torch.from_numpy(modified_image).unsqueeze(0)
            result[b] = modified_image

        return (result, )


NODE_CLASS_MAPPINGS = {
    "Dither": Dither,
    "KMeansQuantize": KMeansQuantize,
    "GaussianBlur": GaussianBlur,
    "Sharpen": Sharpen,
    "CannyEdgeDetection": CannyEdgeDetection,
    "ColorCorrect": ColorCorrect,
}

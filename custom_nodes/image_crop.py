import math

import einops
import torch
import torchvision.transforms as T
from PIL import ImageFilter
from PIL.Image import Image

import nodes


class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "center_x": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 16,  # Slider's step
                }),
                "center_y": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 16,  # Slider's step
                }),
                "pixelradius": ("INT", {
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 16,  # Slider's step
                })
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)

    FUNCTION = "image_crop"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    def image_crop(self, vae, latent, center_x, center_y, pixelradius):
        tensor_img = vae.decode(latent["samples"])
        stripped_tensor_img = tensor_img[0]
        h, w, c = stripped_tensor_img.size()
        pil_img: Image = T.ToPILImage()(einops.rearrange(stripped_tensor_img, "h w c -> c h w"))
        nw, nh = center_x + pixelradius / 2, center_y + pixelradius / 2
        pil_img = pil_img.crop((center_x - pixelradius / 2, center_y - pixelradius / 2, nw, nh))
        new_tensor_img = einops.reduce(T.ToTensor()(pil_img), "c h w -> 1 h w c", "max")
        # new_tensor_img = new_stripped_tensor_img.permute(0, 1, 2, 3)
        pixels = nodes.VAEEncode.vae_encode_crop_pixels(new_tensor_img)
        new_latent = vae.encode(pixels[:, :, :, :3])
        return ({"samples": new_latent}, new_tensor_img)


NODE_CLASS_MAPPINGS = {
    "ImageCrop": ImageCrop
}

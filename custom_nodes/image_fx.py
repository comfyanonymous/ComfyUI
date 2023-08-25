import math

import torch
import torchvision.transforms as T
from PIL import ImageFilter
from PIL.Image import Image

import nodes


class ImageFX:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
                "vae": ("VAE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE",)

    FUNCTION = "image_fx"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    def image_fx(self, vae, latent):
        tensor_img = vae.decode(latent["samples"])
        stripped_tensor_img = tensor_img[0]
        h, w, c = stripped_tensor_img.size()
        pil_img: Image = T.ToPILImage()(stripped_tensor_img.reshape(c, h, w))
        pil_img = pil_img.filter(ImageFilter.ModeFilter(2))
        new_stripped_tensor_img = T.PILToTensor()(pil_img) / 255.0
        new_tensor_img = new_stripped_tensor_img.reshape(1, h, w, c)
        pixels = nodes.VAEEncode.vae_encode_crop_pixels(new_tensor_img)
        new_latent = vae.encode(pixels[:, :, :, :3])
        return ({"samples": new_latent}, new_tensor_img)


NODE_CLASS_MAPPINGS = {
    "ImageFX": ImageFX
}

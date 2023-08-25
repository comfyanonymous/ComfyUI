import math

import torch
import torchvision.transforms as T
from PIL.Image import Image


class DebugLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"latent": ("LATENT",), }
                }

    RETURN_TYPES = ("LATENT", "LATENT",)
    FUNCTION = "latent_space"
    OUTPUT_NODE = True

    CATEGORY = "inflamously"

    def latent_space(self, latent):
        x = latent["samples"]
        transformer = T.ToPILImage()
        img: Image = transformer(x[0])
        # img.show()
        # y = x * 0.75 - x * 0.25 + torch.rand(x.shape) * 0.1
        y = x * 0.5 + torch.rand(x.shape) * 0.5
        modified_latent = {"samples": y}
        return (latent, modified_latent)


NODE_CLASS_MAPPINGS = {
    "DebugLatent": DebugLatent
}

import torch

import comfy.model_management

import nodes

class EmptyChromaRadianceLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, width, height, batch_size=1):
        latent = torch.zeros((batch_size, 3, height, width), device=self.device)
        return ({"samples":latent}, )


class ChromaRadianceLatentToImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, latent):
        img = latent["samples"].to(device=self.device, dtype=torch.float32, copy=True)
        img = img.clamp_(-1, 1).movedim(1, -1).contiguous()
        img += 1.0
        img *= 0.5
        return (img.clamp_(0, 1),)

class ChromaRadianceImageToLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    def go(self, *, image):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError("Unexpected input image shape")
        h, w, c = image.shape[1:]
        if h < 16 or w < 16 or not (h / 16).is_integer() or not (w / 16).is_integer():
            raise ValueError("Chroma Radiance image inputs must have sizes that are multiples of 16.")
        if c > 3:
            image = image[..., :3]
        elif c == 1:
            image = image.expand(-1, -1, -1, 3)
        elif c != 3:
            raise ValueError("Unexpected number of channels in input image")
        latent = image.to(device=self.device, dtype=torch.float32, copy=True)
        latent = latent.clamp_(0, 1).movedim(-1, 1).contiguous()
        latent -= 0.5
        latent *= 2
        return ({"samples": latent.clamp_(-1, 1)},)

NODE_CLASS_MAPPINGS = {
    "EmptyChromaRadianceLatentImage": EmptyChromaRadianceLatentImage,
    "ChromaRadianceLatentToImage": ChromaRadianceLatentToImage,
    "ChromaRadianceImageToLatent": ChromaRadianceImageToLatent,
}

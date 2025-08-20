import torch

import comfy.model_management

import nodes

class EmptyChromaRadianceLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 2, "max": nodes.MAX_RESOLUTION}),
                              "height": ("INT", {"default": 1024, "min": 2, "max": nodes.MAX_RESOLUTION}),
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

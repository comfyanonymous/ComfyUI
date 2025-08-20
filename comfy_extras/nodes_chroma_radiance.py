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
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    @classmethod
    def go(cls, *, latent):
        img = latent["samples"].movedim(1, -1).clamp(-1, 1).contiguous()
        img = (img + 1.0) * 0.5
        return (img,)

class ChromaRadianceImageToLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "go"

    CATEGORY = "latent/chroma_radiance"

    @classmethod
    def go(cls, *, image):
        image = (image.clone().clamp(0, 1) - 0.5) * 2
        image = image.movedim(-1, 1).contiguous()
        return ({"samples": image},)

NODE_CLASS_MAPPINGS = {
    "EmptyChromaRadianceLatentImage": EmptyChromaRadianceLatentImage,
    "ChromaRadianceLatentToImage": ChromaRadianceLatentToImage,
    "ChromaRadianceImageToLatent": ChromaRadianceImageToLatent,
}

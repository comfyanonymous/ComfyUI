import nodes
import torch
import comfy.model_management

class EmptyCosmosLatentVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 704, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "length": ("INT", {"default": 121, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples":latent}, )

NODE_CLASS_MAPPINGS = {
    "EmptyCosmosLatentVideo": EmptyCosmosLatentVideo,
}

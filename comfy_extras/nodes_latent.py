import torch
from nodes import MAX_RESOLUTION

# diffusers library scale the random noise
default_vae_scaling_factor = 1.0/0.18215 

class NoisyLatentImage:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              "vae_scaling_factor": ("FLOAT", {"default": default_vae_scaling_factor, "min": 0.0, "max": 10, "step": 0.01}),
                              "width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, seed, vae_scaling_factor, width, height, batch_size=1):
        generator = torch.manual_seed(seed)
        latent = torch.randn([batch_size, 4, height // 8, width // 8], generator=generator, device=self.device) * vae_scaling_factor
        return ({"samples":latent}, )


NODE_CLASS_MAPPINGS = {
    "Noisy Latent Image": NoisyLatentImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoisyLatentImage": "Noisy Latent Image"
}

import nodes
import torch
import comfy.model_management
import comfy.utils


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
        return ({"samples": latent}, )


class CosmosImageToVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "width": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 704, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 121, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, vae, image, width, height, length, batch_size):
        pixels = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        pixel_len = min(pixels.shape[0], length)
        padded_length = min(length, (((pixel_len - 1) // 8) + 2) * 8 - 7)
        padded_pixels = torch.ones((padded_length, height, width, 3)) * 0.5
        padded_pixels[:pixel_len] = pixels[:pixel_len]

        latent_temp = vae.encode(padded_pixels)

        latent = torch.zeros([1, latent_temp.shape[1], ((length - 1) // 8) + 1, latent_temp.shape[-2], latent_temp.shape[-1]], device=comfy.model_management.intermediate_device())
        latent_len = ((pixel_len - 1) // 8) + 1
        latent[:, :, :latent_len] = latent_temp[:, :, :latent_len]

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 8) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())
        mask[:, :, :latent_len] *= 0.0

        out_latent = {}
        out_latent["samples"] = latent
        out_latent["noise_mask"] = mask
        return (out_latent,)


NODE_CLASS_MAPPINGS = {
    "EmptyCosmosLatentVideo": EmptyCosmosLatentVideo,
    "CosmosImageToVideoLatent": CosmosImageToVideoLatent,
}

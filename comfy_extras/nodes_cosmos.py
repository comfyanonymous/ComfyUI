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


def vae_encode_with_padding(vae, image, width, height, length, padding=0):
    pixels = comfy.utils.common_upscale(image[..., :3].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    pixel_len = min(pixels.shape[0], length)
    padded_length = min(length, (((pixel_len - 1) // 8) + 1 + padding) * 8 - 7)
    padded_pixels = torch.ones((padded_length, height, width, 3)) * 0.5
    padded_pixels[:pixel_len] = pixels[:pixel_len]
    latent_len = ((pixel_len - 1) // 8) + 1
    latent_temp = vae.encode(padded_pixels)
    return latent_temp[:, :, :latent_len]


class CosmosImageToVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE", ),
                             "width": ("INT", {"default": 1280, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 704, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 121, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}


    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, vae, width, height, length, batch_size, start_image=None, end_image=None):
        latent = torch.zeros([1, 16, ((length - 1) // 8) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is None and end_image is None:
            out_latent = {}
            out_latent["samples"] = latent
            return (out_latent,)

        mask = torch.ones([latent.shape[0], 1, ((length - 1) // 8) + 1, latent.shape[-2], latent.shape[-1]], device=comfy.model_management.intermediate_device())

        if start_image is not None:
            latent_temp = vae_encode_with_padding(vae, start_image, width, height, length, padding=1)
            latent[:, :, :latent_temp.shape[-3]] = latent_temp
            mask[:, :, :latent_temp.shape[-3]] *= 0.0

        if end_image is not None:
            latent_temp = vae_encode_with_padding(vae, end_image, width, height, length, padding=0)
            latent[:, :, -latent_temp.shape[-3]:] = latent_temp
            mask[:, :, -latent_temp.shape[-3]:] *= 0.0

        out_latent = {}
        out_latent["samples"] = latent.repeat((batch_size, ) + (1,) * (latent.ndim - 1))
        out_latent["noise_mask"] = mask.repeat((batch_size, ) + (1,) * (mask.ndim - 1))
        return (out_latent,)


NODE_CLASS_MAPPINGS = {
    "EmptyCosmosLatentVideo": EmptyCosmosLatentVideo,
    "CosmosImageToVideoLatent": CosmosImageToVideoLatent,
}

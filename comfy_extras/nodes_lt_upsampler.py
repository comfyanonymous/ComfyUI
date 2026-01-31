from comfy import model_management
import math

class LTXVLatentUpsampler:
    """
    Upsamples a video latent by a factor of 2.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample_latent"
    CATEGORY = "latent/video"
    EXPERIMENTAL = True

    def upsample_latent(
        self,
        samples: dict,
        upscale_model,
        vae,
    ) -> tuple:
        """
        Upsample the input latent using the provided model.

        Args:
            samples (dict): Input latent samples
            upscale_model (LatentUpsampler): Loaded upscale model
            vae: VAE model for normalization
            auto_tiling (bool): Whether to automatically tile the input for processing

        Returns:
            tuple: Tuple containing the upsampled latent
        """
        device = model_management.get_torch_device()
        memory_required = model_management.module_size(upscale_model)

        model_dtype = next(upscale_model.parameters()).dtype
        latents = samples["samples"]
        input_dtype = latents.dtype

        memory_required += math.prod(latents.shape) * 3000.0  # TODO: more accurate
        model_management.free_memory(memory_required, device)

        try:
            upscale_model.to(device)  # TODO: use the comfy model management system.

            latents = latents.to(dtype=model_dtype, device=device)

            """Upsample latents without tiling."""
            latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
            upsampled_latents = upscale_model(latents)
        finally:
            upscale_model.cpu()

        upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )
        upsampled_latents = upsampled_latents.to(dtype=input_dtype, device=model_management.intermediate_device())
        return_dict = samples.copy()
        return_dict["samples"] = upsampled_latents
        return_dict.pop("noise_mask", None)
        return (return_dict,)


def ltxLatentUpscalerBySizeWithModel(model, samples, upscale_method, width, height, crop):
    if width == 0 and height == 0:
        return io.NodeOutput(samples)
    else:
        if width == 0:
            height = max(64, height)
            width = max(64, round(samples.shape[-1] * height / samples.shape[-2]))
        elif height == 0:
            width = max(64, width)
            height = max(64, round(samples.shape[-2] * width / samples.shape[-1]))
        else:
            width = max(64, width)
            height = max(64, height)
        s = comfy.utils.common_upscale(samples, width // 64, height // 64, upscale_method, crop)
        s = model(s)
        
        return s


class LTXVLatentUpsamplerBySize: 
    methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]  
    options = ["disabled", "center"] 
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"samples": ("LATENT",),
                     "upscale_method": (s.methods, {"default": "bilinear"}),
                     "upscale_model": ("LATENT_UPSCALE_MODEL",),
                     "vae": ("VAE",),
                     "width": ("INT", {"default": 1280, "min": 0, "max": 16384, "step": 8}),
                     "height": ("INT", {"default": 720, "min": 0, "max": 16384, "step": 8}),
                     "crop": (s.options,),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upsample_latent"
    CATEGORY = "latent/video"
    DESCRIPTION = "Upscale latents to the desired size"

    def upsample_latent(cls, samples, upscale_method, upscale_model, vae, width, height, crop) -> tuple:
        
        #-------------------------------------------------------------------
        device = comfy.model_management.get_torch_device()
        memory_required = comfy.model_management.module_size(upscale_model)

        model_dtype = next(upscale_model.parameters()).dtype
        latents = samples["samples"]
        input_dtype = latents.dtype

        memory_required += math.prod(latents.shape) * 3000.0  # TODO: more accurate
        comfy.model_management.free_memory(memory_required, device)

        try:
            upscale_model.to(device)  # TODO: use the comfy model management system.

            latents = latents.to(dtype=model_dtype, device=device)

            """Upsample latents without tiling."""
            latents = vae.first_stage_model.per_channel_statistics.un_normalize(latents)
            upsampled_latents = ltxLatentUpscalerBySizeWithModel(upscale_model, latents, upscale_method, width, height, crop)
        finally:
            upscale_model.cpu()

        upsampled_latents = vae.first_stage_model.per_channel_statistics.normalize(
            upsampled_latents
        )
        upsampled_latents = upsampled_latents.to(dtype=input_dtype, device=comfy.model_management.intermediate_device())
        return_dict = samples.copy()
        return_dict["samples"] = upsampled_latents
        return_dict.pop("noise_mask", None)

        return (return_dict,)
        

NODE_CLASS_MAPPINGS = {
    "LTXVLatentUpsampler": LTXVLatentUpsampler,
    "LTXVLatentUpsamplerBySize": LTXVLatentUpsamplerBySize,
}

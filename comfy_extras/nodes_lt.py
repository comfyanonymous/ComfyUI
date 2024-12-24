import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.model_sampling
import math

class EmptyLTXVLatentVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 768, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                              "length": ("INT", {"default": 97, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video/ltxv"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        return ({"samples": latent}, )


class LTXVImgToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE",),
                             "image": ("IMAGE",),
                             "width": ("INT", {"default": 768, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 32}),
                             "length": ("INT", {"default": 97, "min": 9, "max": nodes.MAX_RESOLUTION, "step": 8}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "image_noise_scale": ("FLOAT", {"default": 0.15, "min": 0, "max": 1.0, "step": 0.01, "tooltip": "Amount of noise to apply on conditioning image latent."})
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")

    CATEGORY = "conditioning/video_models"
    FUNCTION = "generate"

    def generate(self, positive, negative, image, vae, width, height, length, batch_size, image_noise_scale):
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        positive = node_helpers.conditioning_set_values(positive, {"guiding_latent": t, "guiding_latent_noise_scale": image_noise_scale})
        negative = node_helpers.conditioning_set_values(negative, {"guiding_latent": t, "guiding_latent_noise_scale": image_noise_scale})

        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        latent[:, :, :t.shape[2]] = t
        return (positive, negative, {"samples": latent}, )


class LTXVConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "frame_rate": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "append"

    CATEGORY = "conditioning/video_models"

    def append(self, positive, negative, frame_rate):
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": frame_rate})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": frame_rate})
        return (positive, negative)


class ModelSamplingLTXV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step":0.01}),
                              "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step":0.01}),
                              },
                "optional": {"latent": ("LATENT",), }
                }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model, max_shift, base_shift, latent=None):
        m = model.clone()

        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (tokens) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)

        return (m, )


class LTXVScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step":0.01}),
                     "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step":0.01}),
                     "stretch": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Stretch the sigmas to be in the range [terminal, 1]."
                    }),
                     "terminal": (
                        "FLOAT",
                        {
                            "default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01,
                            "tooltip": "The terminal value of the sigmas after stretching."
                        },
                    ),
                    },
                "optional": {"latent": ("LATENT",), }
               }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, max_shift, base_shift, stretch, terminal, latent=None):
        if latent is None:
            tokens = 4096
        else:
            tokens = math.prod(latent["samples"].shape[2:])

        sigmas = torch.linspace(1.0, 0.0, steps + 1)

        x1 = 1024
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = (tokens) * mm + b

        power = 1
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
            0,
        )

        # Stretch sigmas so that its final value matches the given terminal value.
        if stretch:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            stretched = 1.0 - (one_minus_z / scale_factor)
            sigmas[non_zero_mask] = stretched

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "EmptyLTXVLatentVideo": EmptyLTXVLatentVideo,
    "LTXVImgToVideo": LTXVImgToVideo,
    "ModelSamplingLTXV": ModelSamplingLTXV,
    "LTXVConditioning": LTXVConditioning,
    "LTXVScheduler": LTXVScheduler,
}

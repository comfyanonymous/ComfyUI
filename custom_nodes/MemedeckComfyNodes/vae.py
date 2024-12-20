from copy import copy

import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import torch
from diffusers.image_processor import VaeImageProcessor
from ltx_video.models.autoencoders.vae_encode import (
    get_vae_size_scale_factor,
    vae_decode,
    vae_encode,
)


class MD_VideoVAE(comfy.sd.VAE):
    def __init__(self, decode_timestep=0.05, decode_noise_scale=0.025, seed=42):
        self.device = comfy.model_management.vae_device()
        self.offload_device = comfy.model_management.vae_offload_device()
        self.decode_timestep = decode_timestep
        self.decode_noise_scale = decode_noise_scale
        self.seed = seed

    @classmethod
    def from_pretrained(cls, vae_class, model_path, dtype=torch.bfloat16):
        instance = cls()
        model = vae_class.from_pretrained(
            pretrained_model_name_or_path=model_path,
            revision=None,
            torch_dtype=dtype,
            load_in_8bit=False,
        )
        instance._finalize_model(model)
        return instance

    @classmethod
    def from_config_and_state_dict(
        cls, vae_class, config, state_dict, dtype=torch.bfloat16
    ):
        instance = cls()
        model = vae_class.from_config(config)
        model.load_state_dict(state_dict)
        model.to(dtype)
        instance._finalize_model(model)
        return instance

    def _finalize_model(self, model):
        self.video_scale_factor, self.vae_scale_factor, _ = get_vae_size_scale_factor(
            model
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.first_stage_model = model.eval().to(self.device)

    # Assumes that the input samples have dimensions in following order
    # (batch, channels, frames, height, width)
    def decode(self, samples_in):
        is_video = samples_in.shape[2] > 1
        decode_timestep = self.decode_timestep
        if getattr(self.first_stage_model.decoder, "timestep_conditioning", False):
            samples_in = self.add_noise(
                decode_timestep, self.decode_noise_scale, self.seed, samples_in
            )
        else:
            decode_timestep = None

        result = vae_decode(
            samples_in.to(self.device),
            vae=self.first_stage_model,
            is_video=is_video,
            vae_per_channel_normalize=True,
            timestep=decode_timestep,
        )
        result = self.image_processor.postprocess(
            result, output_type="pt", do_denormalize=[True]
        )
        return result.squeeze(0).permute(1, 2, 3, 0).to(torch.float32)

    @staticmethod
    def add_noise(decode_timestep, decode_noise_scale, seed, latents):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            latents.size(),
            generator=generator,
            device=latents.device,
            dtype=latents.dtype,
        )
        if not isinstance(decode_timestep, list):
            decode_timestep = [decode_timestep] * latents.shape[0]
        if decode_noise_scale is None:
            decode_noise_scale = decode_timestep
        elif not isinstance(decode_noise_scale, list):
            decode_noise_scale = [decode_noise_scale] * latents.shape[0]

        decode_timestep = torch.tensor(decode_timestep).to(latents.device)
        decode_noise_scale = torch.tensor(decode_noise_scale).to(latents.device)[
            :, None, None, None, None
        ]
        latents = latents * (1 - decode_noise_scale) + noise * decode_noise_scale
        return latents

    # Underlying VAE expects b, c, n, h, w dimensions order and dtype specific dtype.
    # However in Comfy the convension is n, h, w, c.
    def encode(self, pixel_samples):
        preprocessed = self.image_processor.preprocess(
            pixel_samples.permute(3, 0, 1, 2)
        )
        input = preprocessed.unsqueeze(0).to(torch.bfloat16).to(self.device)
        latents = vae_encode(
            input, self.first_stage_model, vae_per_channel_normalize=True
        ).to(comfy.model_management.get_torch_device())
        return latents


# class DecoderNoise:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "vae": ("VAE",),
#                 "timestep": (
#                     "FLOAT",
#                     {
#                         "default": 0.05,
#                         "min": 0.0,
#                         "max": 1.0,
#                         "step": 0.01,
#                         "tooltip": "The timestep used for decoding the noise.",
#                     },
#                 ),
#                 "scale": (
#                     "FLOAT",
#                     {
#                         "default": 0.025,
#                         "min": 0.0,
#                         "max": 1.0,
#                         "step": 0.001,
#                         "tooltip": "The scale of the noise added to the decoder.",
#                     },
#                 ),
#                 "seed": (
#                     "INT",
#                     {
#                         "default": 42,
#                         "min": 0,
#                         "max": 0xFFFFFFFFFFFFFFFF,
#                         "tooltip": "The random seed used for creating the noise.",
#                     },
#                 ),
#             }
#         }

#     FUNCTION = "add_noise"
#     RETURN_TYPES = ("VAE",)
#     CATEGORY = "lightricks/LTXV"

#     def add_noise(self, vae, timestep, scale, seed):
#         result = copy(vae)
#         result.decode_timestep = timestep
#         result.decode_noise_scale = scale
#         result.seed = seed
#         return (result,)

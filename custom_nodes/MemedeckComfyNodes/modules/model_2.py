import math
from contextlib import nullcontext

import comfy.latent_formats
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.sd
import comfy.supported_models_base
import comfy.utils
import torch
import torch.nn as nn
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel


class LTXVModelConfig:
    def __init__(self, latent_channels, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.LatentFormat()
        self.latent_format.latent_channels = latent_channels
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.7
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True


class LTXVSampling(torch.nn.Module, comfy.model_sampling.CONST):
    def __init__(self, condition_mask, guiding_latent=None):
        super().__init__()
        self.condition_mask = condition_mask
        self.guiding_latent = guiding_latent
        self.set_parameters(shift=1.0, multiplier=1)

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.shift = shift
        self.multiplier = multiplier
        ts = self.sigma((torch.arange(0, timesteps + 1, 1) / timesteps) * multiplier)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return timestep

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return 1.0 - percent

    def calculate_input(self, sigma, noise):
        if self.guiding_latent is not None:
            noise = (
                noise * (1 - self.condition_mask)
                + self.guiding_latent * self.condition_mask
            )
        return noise

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        self.condition_mask = self.condition_mask.to(latent_image.device)
        scaled = latent_image * (1 - sigma) + noise * sigma
        result = latent_image * self.condition_mask + scaled * (1 - self.condition_mask)

        return result

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        result = model_input - model_output * sigma
        # In order to d * dT to be zero in euler step, we need to set result equal to input in first latent frame.
        if self.guiding_latent is not None:
            result = (
                result * (1 - self.condition_mask)
                + self.guiding_latent * self.condition_mask
            )
        else:
            result = (
                result * (1 - self.condition_mask) + model_input * self.condition_mask
            )
        return result


class LTXVModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_sampling = LTXVSampling(torch.zeros([1]))


class LTXVTransformer3D(nn.Module):
    def __init__(
        self,
        transformer: Transformer3DModel,
        patchifier: SymmetricPatchifier,
        conditioning_mask,
        latent_frame_rate,
        vae_scale_factor,
    ):
        super().__init__()

        self.dtype = transformer.dtype
        self.transformer = transformer
        self.patchifier = patchifier
        self.conditioning_mask = conditioning_mask
        self.latent_frame_rate = latent_frame_rate
        self.vae_scale_factor = vae_scale_factor

    def indices_grid(
        self,
        latent_shape,
        device,
    ):
        use_rope = self.transformer.use_rope
        scale_grid = (
            (1 / self.latent_frame_rate, self.vae_scale_factor, self.vae_scale_factor)
            if use_rope
            else None
        )

        indices_grid = self.patchifier.get_grid(
            orig_num_frames=latent_shape[2],
            orig_height=latent_shape[3],
            orig_width=latent_shape[4],
            batch_size=latent_shape[0],
            scale_grid=scale_grid,
            device=device,
        )

        return indices_grid

    def wrapped_transformer(
        self,
        latent,
        timesteps,
        context,
        indices_grid,
        skip_layer_mask=None,
        skip_layer_strategy=None,
        img_hw=None,
        aspect_ratio=None,
        mixed_precision=True,
        **kwargs,
    ):
        # infer mask from context padding, assumes padding vectors are all zero.
        latent = latent.to(self.transformer.dtype)
        latent_patchified = self.patchifier.patchify(latent)
        context_mask = (context != 0).any(dim=2).to(self.transformer.dtype)

        if mixed_precision:
            context_manager = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context_manager = nullcontext()
        with context_manager:
            noise_pred = self.transformer(
                latent_patchified.to(self.transformer.dtype).to(
                    self.transformer.device
                ),
                indices_grid.to(self.transformer.device),
                encoder_hidden_states=context.to(self.transformer.device),
                encoder_attention_mask=context_mask.to(self.transformer.device).to(
                    torch.int64
                ),
                timestep=timesteps,
                skip_layer_mask=skip_layer_mask,
                skip_layer_strategy=skip_layer_strategy,
                return_dict=False,
            )[0]

        result = self.patchifier.unpatchify(
            latents=noise_pred,
            output_height=latent.shape[3],
            output_width=latent.shape[4],
            output_num_frames=latent.shape[2],
            out_channels=latent.shape[1] // math.prod(self.patchifier.patch_size),
        )
        return result

    def forward(self, x, timesteps, context, img_hw=None, aspect_ratio=None, **kwargs):
        transformer_options = kwargs.get("transformer_options", {})
        ptb_index = transformer_options.get("ptb_index", None)
        mixed_precision = transformer_options.get("mixed_precision", False)
        cond_or_uncond = transformer_options.get("cond_or_uncond", [])
        skip_block_list = transformer_options.get("skip_block_list", [])
        skip_layer_strategy = transformer_options.get("skip_layer_strategy", None)
        mask = self.patchifier.patchify(self.conditioning_mask).squeeze(-1).to(x.device)
        ndim_mask = mask.ndimension()
        expanded_timesteps = timesteps.view(timesteps.size(0), *([1] * (ndim_mask - 1)))
        timesteps_masked = expanded_timesteps * (1 - mask)
        skip_layer_mask = None
        if ptb_index is not None and ptb_index in cond_or_uncond:
            skip_layer_mask = self.transformer.create_skip_layer_mask(
                skip_block_list,
                1,
                len(cond_or_uncond),
                len(cond_or_uncond) - 1 - cond_or_uncond.index(ptb_index),
            )

        result = self.wrapped_transformer(
            x,
            timesteps_masked,
            context,
            indices_grid=self.indices_grid(x.shape, x.device),
            mixed_precision=mixed_precision,
            skip_layer_mask=skip_layer_mask,
            skip_layer_strategy=skip_layer_strategy,
        )
        return result

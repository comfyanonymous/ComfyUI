#credit to huchenlei for this module
#from https://github.com/huchenlei/ComfyUI-IC-Light-Native
import torch
import numpy as np
from typing import Tuple, TypedDict, Callable

import comfy.model_management
from comfy.sd import load_unet
from comfy.ldm.models.autoencoder import AutoencoderKL
from comfy.model_base import BaseModel
from comfy.model_patcher import ModelPatcher
from PIL import Image
from nodes import VAEEncode
from ...libs.image import np2tensor, pil2tensor

class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor

class VAEEncodeArgMax(VAEEncode):
    def encode(self, vae, pixels):
        assert isinstance(
            vae.first_stage_model, AutoencoderKL
        ), "ArgMax only supported for AutoencoderKL"
        original_sample_mode = vae.first_stage_model.regularization.sample
        vae.first_stage_model.regularization.sample = False
        ret = super().encode(vae, pixels)
        vae.first_stage_model.regularization.sample = original_sample_mode
        return ret

class ICLight:

    @staticmethod
    def apply_c_concat(params: UnetParams, concat_conds) -> UnetParams:
        """Apply c_concat on unet call."""
        sample = params["input"]
        params["c"]["c_concat"] = torch.cat(
            (
                    [concat_conds.to(sample.device)]
                    * (sample.shape[0] // concat_conds.shape[0])
            ),
            dim=0,
        )
        return params

    @staticmethod
    def create_custom_conv(
        original_conv: torch.nn.Module,
        dtype: torch.dtype,
        device=torch.device,
    ) -> torch.nn.Module:
        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                original_conv.out_channels,
                original_conv.kernel_size,
                original_conv.stride,
                original_conv.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(original_conv.weight)
            new_conv_in.bias = original_conv.bias
            return new_conv_in.to(dtype=dtype, device=device)

    def generate_lighting_image(self, original_image, direction):
        _, image_height, image_width, _ = original_image.shape
        if direction == 'Left Light':
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif direction == 'Right Light':
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif direction == 'Top Light':
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif direction == 'Bottom Light':
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif direction == 'Circle Light':
            x = np.linspace(-1, 1, image_width)
            y = np.linspace(-1, 1, image_height)
            x, y = np.meshgrid(x, y)
            r = np.sqrt(x ** 2 + y ** 2)
            r = r / r.max()
            color1 = np.array([0, 0, 0])[np.newaxis, np.newaxis, :]
            color2 = np.array([255, 255, 255])[np.newaxis, np.newaxis, :]
            gradient = (color1 * r[..., np.newaxis] + color2 * (1 - r)[..., np.newaxis]).astype(np.uint8)
            image = pil2tensor(Image.fromarray(gradient))
            return image
        else:
            image = pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))
            return image

    def generate_source_image(self, original_image, source):
        batch_size, image_height, image_width, _ = original_image.shape
        if source == 'Use Flipped Background Image':
            if batch_size < 2:
                raise ValueError('Must be at least 2 image to use flipped background image.')
            original_image = [img.unsqueeze(0) for img in original_image]
            image = torch.flip(original_image[1], [2])
            return image
        elif source == 'Ambient':
            input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
            return np2tensor(input_bg)
        elif source == 'Left Light':
            gradient = np.linspace(224, 32, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif source == 'Right Light':
            gradient = np.linspace(32, 224, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif source == 'Top Light':
            gradient = np.linspace(224, 32, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        elif source == 'Bottom Light':
            gradient = np.linspace(32, 224, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
            return np2tensor(input_bg)
        else:
            image = pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))
            return image


    def apply(self, ic_model_path, model, c_concat: dict, ic_model=None) -> Tuple[ModelPatcher]:
        device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype()
        work_model = model.clone()

        # Apply scale factor.
        base_model: BaseModel = work_model.model
        scale_factor = base_model.model_config.latent_format.scale_factor

        # [B, 4, H, W]
        concat_conds: torch.Tensor = c_concat["samples"] * scale_factor
        # [1, 4 * B, H, W]
        concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

        def unet_dummy_apply(unet_apply: Callable, params: UnetParams):
            """A dummy unet apply wrapper serving as the endpoint of wrapper
            chain."""
            return unet_apply(x=params["input"], t=params["timestep"], **params["c"])

        existing_wrapper = work_model.model_options.get(
            "model_function_wrapper", unet_dummy_apply
        )

        def wrapper_func(unet_apply: Callable, params: UnetParams):
            return existing_wrapper(unet_apply, params=self.apply_c_concat(params, concat_conds))

        work_model.set_model_unet_function_wrapper(wrapper_func)
        if not ic_model:
            ic_model = load_unet(ic_model_path)
        ic_model_state_dict = ic_model.model.diffusion_model.state_dict()

        work_model.add_patches(
            patches={
                ("diffusion_model." + key): (
                    'diff',
                    [
                        value.to(dtype=dtype, device=device),
                        {"pad_weight": key == 'input_blocks.0.0.weight'}
                    ]
                )
                for key, value in ic_model_state_dict.items()
            }
        )

        return (work_model, ic_model)
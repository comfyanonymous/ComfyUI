"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
import logging
from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from comfy.ldm.cascade.stage_c import StageC
from comfy.ldm.cascade.stage_b import StageB
from comfy.ldm.modules.encoders.noise_aug_modules import CLIPEmbeddingNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation
from comfy.ldm.modules.diffusionmodules.mmdit import OpenAISignatureMMDITWrapper
import comfy.ldm.genmo.joint_model.asymm_models_joint
import comfy.ldm.aura.mmdit
import comfy.ldm.pixart.pixartms
import comfy.ldm.hydit.models
import comfy.ldm.audio.dit
import comfy.ldm.audio.embedders
import comfy.ldm.flux.model
import comfy.ldm.lightricks.model
import comfy.ldm.hunyuan_video.model
import comfy.ldm.cosmos.model
import comfy.ldm.cosmos.predict2
import comfy.ldm.lumina.model
import comfy.ldm.wan.model
import comfy.ldm.hunyuan3d.model
import comfy.ldm.hidream.model
import comfy.ldm.chroma.model
import comfy.ldm.ace.model

import comfy.model_management
import comfy.patcher_extension
import comfy.conds
import comfy.ops
from enum import Enum
from . import utils
import comfy.latent_formats
import comfy.model_sampling
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    V_PREDICTION_EDM = 3
    STABLE_CASCADE = 4
    EDM = 5
    FLOW = 6
    V_PREDICTION_CONTINUOUS = 7
    FLUX = 8
    IMG_TO_IMG = 9
    FLOW_COSMOS = 10


def model_sampling(model_config, model_type):
    s = comfy.model_sampling.ModelSamplingDiscrete

    if model_type == ModelType.EPS:
        c = comfy.model_sampling.EPS
    elif model_type == ModelType.V_PREDICTION:
        c = comfy.model_sampling.V_PREDICTION
    elif model_type == ModelType.V_PREDICTION_EDM:
        c = comfy.model_sampling.V_PREDICTION
        s = comfy.model_sampling.ModelSamplingContinuousEDM
    elif model_type == ModelType.FLOW:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingDiscreteFlow
    elif model_type == ModelType.STABLE_CASCADE:
        c = comfy.model_sampling.EPS
        s = comfy.model_sampling.StableCascadeSampling
    elif model_type == ModelType.EDM:
        c = comfy.model_sampling.EDM
        s = comfy.model_sampling.ModelSamplingContinuousEDM
    elif model_type == ModelType.V_PREDICTION_CONTINUOUS:
        c = comfy.model_sampling.V_PREDICTION
        s = comfy.model_sampling.ModelSamplingContinuousV
    elif model_type == ModelType.FLUX:
        c = comfy.model_sampling.CONST
        s = comfy.model_sampling.ModelSamplingFlux
    elif model_type == ModelType.IMG_TO_IMG:
        c = comfy.model_sampling.IMG_TO_IMG
    elif model_type == ModelType.FLOW_COSMOS:
        c = comfy.model_sampling.COSMOS_RFLOW
        s = comfy.model_sampling.ModelSamplingCosmosRFlow

    class ModelSampling(s, c):
        pass

    return ModelSampling(model_config)


def convert_tensor(extra, dtype):
    if hasattr(extra, "dtype"):
        if extra.dtype != torch.int and extra.dtype != torch.long:
            extra = extra.to(dtype)
    return extra


class BaseModel(torch.nn.Module):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, unet_model=UNetModel):
        super().__init__()

        unet_config = model_config.unet_config
        self.latent_format = model_config.latent_format
        self.model_config = model_config
        self.manual_cast_dtype = model_config.manual_cast_dtype
        self.device = device
        self.current_patcher: 'ModelPatcher' = None

        if not unet_config.get("disable_unet_model_creation", False):
            if model_config.custom_operations is None:
                fp8 = model_config.optimizations.get("fp8", False)
                operations = comfy.ops.pick_operations(unet_config.get("dtype", None), self.manual_cast_dtype, fp8_optimizations=fp8, scaled_fp8=model_config.scaled_fp8)
            else:
                operations = model_config.custom_operations
            self.diffusion_model = unet_model(**unet_config, device=device, operations=operations)
            if comfy.model_management.force_channels_last():
                self.diffusion_model.to(memory_format=torch.channels_last)
                logging.debug("using channels last mode for diffusion model")
            logging.info("model weight dtype {}, manual cast: {}".format(self.get_dtype(), self.manual_cast_dtype))
        self.model_type = model_type
        self.model_sampling = model_sampling(model_config, model_type)

        self.adm_channels = unet_config.get("adm_in_channels", None)
        if self.adm_channels is None:
            self.adm_channels = 0

        self.concat_keys = ()
        logging.info("model_type {}".format(model_type.name))
        logging.debug("adm {}".format(self.adm_channels))
        self.memory_usage_factor = model_config.memory_usage_factor
        self.memory_usage_factor_conds = ()

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._apply_model,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.APPLY_MODEL, transformer_options)
        ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.model_sampling.calculate_input(sigma, x)

        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.get_dtype()

        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        xc = xc.to(dtype)
        t = self.model_sampling.timestep(t).float()
        if context is not None:
            context = context.to(dtype)

        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]

            if hasattr(extra, "dtype"):
                extra = convert_tensor(extra, dtype)
            elif isinstance(extra, list):
                ex = []
                for ext in extra:
                    ex.append(convert_tensor(ext, dtype))
                extra = ex
            extra_conds[o] = extra

        t = self.process_timestep(t, x=x, **extra_conds)
        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        return self.model_sampling.calculate_denoised(sigma, model_output, x)

    def process_timestep(self, timestep, **kwargs):
        return timestep

    def get_dtype(self):
        return self.diffusion_model.dtype

    def encode_adm(self, **kwargs):
        return None

    def concat_cond(self, **kwargs):
        if len(self.concat_keys) > 0:
            cond_concat = []
            denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
            concat_latent_image = kwargs.get("concat_latent_image", None)
            if concat_latent_image is None:
                concat_latent_image = kwargs.get("latent_image", None)
            else:
                concat_latent_image = self.process_latent_in(concat_latent_image)

            noise = kwargs.get("noise", None)
            device = kwargs["device"]

            if concat_latent_image.shape[1:] != noise.shape[1:]:
                concat_latent_image = utils.common_upscale(concat_latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")
                if noise.ndim == 5:
                    if concat_latent_image.shape[-3] < noise.shape[-3]:
                        concat_latent_image = torch.nn.functional.pad(concat_latent_image, (0, 0, 0, 0, 0, noise.shape[-3] - concat_latent_image.shape[-3]), "constant", 0)
                    else:
                        concat_latent_image = concat_latent_image[:, :, :noise.shape[-3]]

            concat_latent_image = utils.resize_to_batch_size(concat_latent_image, noise.shape[0])

            if denoise_mask is not None:
                if len(denoise_mask.shape) == len(noise.shape):
                    denoise_mask = denoise_mask[:, :1]

                num_dim = noise.ndim - 2
                denoise_mask = denoise_mask.reshape((-1, 1) + tuple(denoise_mask.shape[-num_dim:]))
                if denoise_mask.shape[-2:] != noise.shape[-2:]:
                    denoise_mask = utils.common_upscale(denoise_mask, noise.shape[-1], noise.shape[-2], "bilinear", "center")
                denoise_mask = utils.resize_to_batch_size(denoise_mask.round(), noise.shape[0])

            for ck in self.concat_keys:
                if denoise_mask is not None:
                    if ck == "mask":
                        cond_concat.append(denoise_mask.to(device))
                    elif ck == "masked_image":
                        cond_concat.append(concat_latent_image.to(device))  # NOTE: the latent_image should be masked by the mask in pixel space
                    elif ck == "mask_inverted":
                        cond_concat.append(1.0 - denoise_mask.to(device))
                else:
                    if ck == "mask":
                        cond_concat.append(torch.ones_like(noise)[:, :1])
                    elif ck == "masked_image":
                        cond_concat.append(self.blank_inpaint_image_like(noise))
                    elif ck == "mask_inverted":
                        cond_concat.append(torch.zeros_like(noise)[:, :1])
                if ck == "concat_image":
                    if concat_latent_image is not None:
                        cond_concat.append(concat_latent_image.to(device))
                    else:
                        cond_concat.append(torch.zeros_like(noise))
            data = torch.cat(cond_concat, dim=1)
            return data
        return None

    def extra_conds(self, **kwargs):
        out = {}
        concat_cond = self.concat_cond(**kwargs)
        if concat_cond is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(concat_cond)

        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)

        cross_attn_cnet = kwargs.get("cross_attn_controlnet", None)
        if cross_attn_cnet is not None:
            out['crossattn_controlnet'] = comfy.conds.CONDCrossAttn(cross_attn_cnet)

        c_concat = kwargs.get("noise_concat", None)
        if c_concat is not None:
            out['c_concat'] = comfy.conds.CONDNoiseShape(c_concat)

        return out

    def load_model_weights(self, sd, unet_prefix=""):
        to_load = {}
        keys = list(sd.keys())
        for k in keys:
            if k.startswith(unet_prefix):
                to_load[k[len(unet_prefix):]] = sd.pop(k)

        to_load = self.model_config.process_unet_state_dict(to_load)
        m, u = self.diffusion_model.load_state_dict(to_load, strict=False)
        if len(m) > 0:
            logging.warning("unet missing: {}".format(m))

        if len(u) > 0:
            logging.warning("unet unexpected: {}".format(u))
        del to_load
        return self

    def process_latent_in(self, latent):
        return self.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.latent_format.process_out(latent)

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        extra_sds = []
        if clip_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_state_dict_for_saving(clip_state_dict))
        if vae_state_dict is not None:
            extra_sds.append(self.model_config.process_vae_state_dict_for_saving(vae_state_dict))
        if clip_vision_state_dict is not None:
            extra_sds.append(self.model_config.process_clip_vision_state_dict_for_saving(clip_vision_state_dict))

        unet_state_dict = self.diffusion_model.state_dict()

        if self.model_config.scaled_fp8 is not None:
            unet_state_dict["scaled_fp8"] = torch.tensor([], dtype=self.model_config.scaled_fp8)

        unet_state_dict = self.model_config.process_unet_state_dict_for_saving(unet_state_dict)

        if self.model_type == ModelType.V_PREDICTION:
            unet_state_dict["v_pred"] = torch.tensor([])

        for sd in extra_sds:
            unet_state_dict.update(sd)

        return unet_state_dict

    def set_inpaint(self):
        self.concat_keys = ("mask", "masked_image")
        def blank_inpaint_image_like(latent_image):
            blank_image = torch.ones_like(latent_image)
            # these are the values for "zero" in pixel space translated to latent space
            blank_image[:,0] *= 0.8223
            blank_image[:,1] *= -0.6876
            blank_image[:,2] *= 0.6364
            blank_image[:,3] *= 0.1380
            return blank_image
        self.blank_inpaint_image_like = blank_inpaint_image_like

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return self.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1)), noise, latent_image)

    def memory_required(self, input_shape, cond_shapes={}):
        input_shapes = [input_shape]
        for c in self.memory_usage_factor_conds:
            shape = cond_shapes.get(c, None)
            if shape is not None and len(shape) > 0:
                input_shapes += shape

        if comfy.model_management.xformers_enabled() or comfy.model_management.pytorch_attention_flash_attention():
            dtype = self.get_dtype()
            if self.manual_cast_dtype is not None:
                dtype = self.manual_cast_dtype
            #TODO: this needs to be tweaked
            area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))
            return (area * comfy.model_management.dtype_size(dtype) * 0.01 * self.memory_usage_factor) * (1024 * 1024)
        else:
            #TODO: this formula might be too aggressive since I tweaked the sub-quad and split algorithms to use less memory.
            area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))
            return (area * 0.15 * self.memory_usage_factor) * (1024 * 1024)

    def extra_conds_shapes(self, **kwargs):
        return {}


def unclip_adm(unclip_conditioning, device, noise_augmentor, noise_augment_merge=0.0, seed=None):
    adm_inputs = []
    weights = []
    noise_aug = []
    for unclip_cond in unclip_conditioning:
        for adm_cond in unclip_cond["clip_vision_output"].image_embeds:
            weight = unclip_cond["strength"]
            noise_augment = unclip_cond["noise_augmentation"]
            noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
            c_adm, noise_level_emb = noise_augmentor(adm_cond.to(device), noise_level=torch.tensor([noise_level], device=device), seed=seed)
            adm_out = torch.cat((c_adm, noise_level_emb), 1) * weight
            weights.append(weight)
            noise_aug.append(noise_augment)
            adm_inputs.append(adm_out)

    if len(noise_aug) > 1:
        adm_out = torch.stack(adm_inputs).sum(0)
        noise_augment = noise_augment_merge
        noise_level = round((noise_augmentor.max_noise_level - 1) * noise_augment)
        c_adm, noise_level_emb = noise_augmentor(adm_out[:, :noise_augmentor.time_embed.dim], noise_level=torch.tensor([noise_level], device=device))
        adm_out = torch.cat((c_adm, noise_level_emb), 1)

    return adm_out

class SD21UNCLIP(BaseModel):
    def __init__(self, model_config, noise_aug_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**noise_aug_config)

    def encode_adm(self, **kwargs):
        unclip_conditioning = kwargs.get("unclip_conditioning", None)
        device = kwargs["device"]
        if unclip_conditioning is None:
            return torch.zeros((1, self.adm_channels))
        else:
            return unclip_adm(unclip_conditioning, device, self.noise_augmentor, kwargs.get("unclip_noise_augment_merge", 0.05), kwargs.get("seed", 0) - 10)

def sdxl_pooled(args, noise_augmentor):
    if "unclip_conditioning" in args:
        return unclip_adm(args.get("unclip_conditioning", None), args["device"], noise_augmentor, seed=args.get("seed", 0) - 10)[:,:1280]
    else:
        return args["pooled_output"]

class SDXLRefiner(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)

        if kwargs.get("prompt_type", "") == "negative":
            aesthetic_score = kwargs.get("aesthetic_score", 2.5)
        else:
            aesthetic_score = kwargs.get("aesthetic_score", 6)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([aesthetic_score])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)

class SDXL(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)
        self.noise_augmentor = CLIPEmbeddingNoiseAugmentation(**{"noise_schedule_config": {"timesteps": 1000, "beta_schedule": "squaredcos_cap_v2"}, "timestep_dim": 1280})

    def encode_adm(self, **kwargs):
        clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        crop_w = kwargs.get("crop_w", 0)
        crop_h = kwargs.get("crop_h", 0)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out = []
        out.append(self.embedder(torch.Tensor([height])))
        out.append(self.embedder(torch.Tensor([width])))
        out.append(self.embedder(torch.Tensor([crop_h])))
        out.append(self.embedder(torch.Tensor([crop_w])))
        out.append(self.embedder(torch.Tensor([target_height])))
        out.append(self.embedder(torch.Tensor([target_width])))
        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1)
        return torch.cat((clip_pooled.to(flat.device), flat), dim=1)


class SVD_img2vid(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder = Timestep(256)

    def encode_adm(self, **kwargs):
        fps_id = kwargs.get("fps", 6) - 1
        motion_bucket_id = kwargs.get("motion_bucket_id", 127)
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.Tensor([fps_id])))
        out.append(self.embedder(torch.Tensor([motion_bucket_id])))
        out.append(self.embedder(torch.Tensor([augmentation])))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat

    def extra_conds(self, **kwargs):
        out = {}
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = utils.common_upscale(latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        latent_image = utils.resize_to_batch_size(latent_image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)

        if "time_conditioning" in kwargs:
            out["time_context"] = comfy.conds.CONDCrossAttn(kwargs["time_conditioning"])

        out['num_video_frames'] = comfy.conds.CONDConstant(noise.shape[0])
        return out

class SV3D_u(SVD_img2vid):
    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0)
        return flat

class SV3D_p(SVD_img2vid):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION_EDM, device=None):
        super().__init__(model_config, model_type, device=device)
        self.embedder_512 = Timestep(512)

    def encode_adm(self, **kwargs):
        augmentation = kwargs.get("augmentation_level", 0)
        elevation = kwargs.get("elevation", 0) #elevation and azimuth are in degrees here
        azimuth = kwargs.get("azimuth", 0)
        noise = kwargs.get("noise", None)

        out = []
        out.append(self.embedder(torch.flatten(torch.Tensor([augmentation]))))
        out.append(self.embedder_512(torch.deg2rad(torch.fmod(torch.flatten(90 - torch.Tensor([elevation])), 360.0))))
        out.append(self.embedder_512(torch.deg2rad(torch.fmod(torch.flatten(torch.Tensor([azimuth])), 360.0))))

        out = list(map(lambda a: utils.resize_to_batch_size(a, noise.shape[0]), out))
        return torch.cat(out, dim=1)


class Stable_Zero123(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None, cc_projection_weight=None, cc_projection_bias=None):
        super().__init__(model_config, model_type, device=device)
        self.cc_projection = comfy.ops.manual_cast.Linear(cc_projection_weight.shape[1], cc_projection_weight.shape[0], dtype=self.get_dtype(), device=device)
        self.cc_projection.weight.copy_(cc_projection_weight)
        self.cc_projection.bias.copy_(cc_projection_bias)

    def extra_conds(self, **kwargs):
        out = {}

        latent_image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)

        if latent_image is None:
            latent_image = torch.zeros_like(noise)

        if latent_image.shape[1:] != noise.shape[1:]:
            latent_image = utils.common_upscale(latent_image, noise.shape[-1], noise.shape[-2], "bilinear", "center")

        latent_image = utils.resize_to_batch_size(latent_image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(latent_image)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            if cross_attn.shape[-1] != 768:
                cross_attn = self.cc_projection(cross_attn)
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out

class SD_X4Upscaler(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device)
        self.noise_augmentor = ImageConcatWithNoiseAugmentation(noise_schedule_config={"linear_start": 0.0001, "linear_end": 0.02}, max_noise_level=350)

    def extra_conds(self, **kwargs):
        out = {}

        image = kwargs.get("concat_image", None)
        noise = kwargs.get("noise", None)
        noise_augment = kwargs.get("noise_augmentation", 0.0)
        device = kwargs["device"]
        seed = kwargs["seed"] - 10

        noise_level = round((self.noise_augmentor.max_noise_level) * noise_augment)

        if image is None:
            image = torch.zeros_like(noise)[:,:3]

        if image.shape[1:] != noise.shape[1:]:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        noise_level = torch.tensor([noise_level], device=device)
        if noise_augment > 0:
            image, noise_level = self.noise_augmentor(image.to(device), noise_level=noise_level, seed=seed)

        image = utils.resize_to_batch_size(image, noise.shape[0])

        out['c_concat'] = comfy.conds.CONDNoiseShape(image)
        out['y'] = comfy.conds.CONDRegular(noise_level)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out

class IP2P:
    def concat_cond(self, **kwargs):
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        if image.shape[1:] != noise.shape[1:]:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")

        image = utils.resize_to_batch_size(image, noise.shape[0])
        return self.process_ip2p_image_in(image)


class SD15_instructpix2pix(IP2P, BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        self.process_ip2p_image_in = lambda image: image


class SDXL_instructpix2pix(IP2P, SDXL):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device)
        if model_type == ModelType.V_PREDICTION_EDM:
            self.process_ip2p_image_in = lambda image: comfy.latent_formats.SDXL().process_in(image) #cosxl ip2p
        else:
            self.process_ip2p_image_in = lambda image: image #diffusers ip2p

class Lotus(BaseModel):
    def extra_conds(self, **kwargs):
        out = {}
        cross_attn = kwargs.get("cross_attn", None)
        out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        device = kwargs["device"]
        task_emb = torch.tensor([1, 0]).float().to(device)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)]).unsqueeze(0)
        out['y'] = comfy.conds.CONDRegular(task_emb)
        return out

    def __init__(self, model_config, model_type=ModelType.IMG_TO_IMG, device=None):
        super().__init__(model_config, model_type, device=device)

class StableCascade_C(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageC)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out['clip_text_pooled'] = comfy.conds.CONDRegular(clip_text_pooled)

        if "unclip_conditioning" in kwargs:
            embeds = []
            for unclip_cond in kwargs["unclip_conditioning"]:
                weight = unclip_cond["strength"]
                embeds.append(unclip_cond["clip_vision_output"].image_embeds.unsqueeze(0) * weight)
            clip_img = torch.cat(embeds, dim=1)
        else:
            clip_img = torch.zeros((1, 1, 768))
        out["clip_img"] = comfy.conds.CONDRegular(clip_img)
        out["sca"] = comfy.conds.CONDRegular(torch.zeros((1,)))
        out["crp"] = comfy.conds.CONDRegular(torch.zeros((1,)))

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['clip_text'] = comfy.conds.CONDCrossAttn(cross_attn)
        return out


class StableCascade_B(BaseModel):
    def __init__(self, model_config, model_type=ModelType.STABLE_CASCADE, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=StageB)
        self.diffusion_model.eval().requires_grad_(False)

    def extra_conds(self, **kwargs):
        out = {}
        noise = kwargs.get("noise", None)

        clip_text_pooled = kwargs["pooled_output"]
        if clip_text_pooled is not None:
            out['clip'] = comfy.conds.CONDRegular(clip_text_pooled)

        #size of prior doesn't really matter if zeros because it gets resized but I still want it to get batched
        prior = kwargs.get("stable_cascade_prior", torch.zeros((1, 16, (noise.shape[2] * 4) // 42, (noise.shape[3] * 4) // 42), dtype=noise.dtype, layout=noise.layout, device=noise.device))

        out["effnet"] = comfy.conds.CONDRegular(prior)
        out["sca"] = comfy.conds.CONDRegular(torch.zeros((1,)))
        return out


class SD3(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=OpenAISignatureMMDITWrapper)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out


class AuraFlow(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.aura.mmdit.MMDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out


class StableAudio1(BaseModel):
    def __init__(self, model_config, seconds_start_embedder_weights, seconds_total_embedder_weights, model_type=ModelType.V_PREDICTION_CONTINUOUS, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.audio.dit.AudioDiffusionTransformer)
        self.seconds_start_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_total_embedder = comfy.ldm.audio.embedders.NumberConditioner(768, min_val=0, max_val=512)
        self.seconds_start_embedder.load_state_dict(seconds_start_embedder_weights)
        self.seconds_total_embedder.load_state_dict(seconds_total_embedder_weights)

    def extra_conds(self, **kwargs):
        out = {}

        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        seconds_start = kwargs.get("seconds_start", 0)
        seconds_total = kwargs.get("seconds_total", int(noise.shape[-1] / 21.53))

        seconds_start_embed = self.seconds_start_embedder([seconds_start])[0].to(device)
        seconds_total_embed = self.seconds_total_embedder([seconds_total])[0].to(device)

        global_embed = torch.cat([seconds_start_embed, seconds_total_embed], dim=-1).reshape((1, -1))
        out['global_embed'] = comfy.conds.CONDRegular(global_embed)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            cross_attn = torch.cat([cross_attn.to(device), seconds_start_embed.repeat((cross_attn.shape[0], 1, 1)), seconds_total_embed.repeat((cross_attn.shape[0], 1, 1))], dim=1)
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

    def state_dict_for_saving(self, clip_state_dict=None, vae_state_dict=None, clip_vision_state_dict=None):
        sd = super().state_dict_for_saving(clip_state_dict=clip_state_dict, vae_state_dict=vae_state_dict, clip_vision_state_dict=clip_vision_state_dict)
        d = {"conditioner.conditioners.seconds_start.": self.seconds_start_embedder.state_dict(), "conditioner.conditioners.seconds_total.": self.seconds_total_embedder.state_dict()}
        for k in d:
            s = d[k]
            for l in s:
                sd["{}{}".format(k, l)] = s[l]
        return sd


class HunyuanDiT(BaseModel):
    def __init__(self, model_config, model_type=ModelType.V_PREDICTION, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hydit.models.HunYuanDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['text_embedding_mask'] = comfy.conds.CONDRegular(attention_mask)

        conditioning_mt5xl = kwargs.get("conditioning_mt5xl", None)
        if conditioning_mt5xl is not None:
            out['encoder_hidden_states_t5'] = comfy.conds.CONDRegular(conditioning_mt5xl)

        attention_mask_mt5xl = kwargs.get("attention_mask_mt5xl", None)
        if attention_mask_mt5xl is not None:
            out['text_embedding_mask_t5'] = comfy.conds.CONDRegular(attention_mask_mt5xl)

        width = kwargs.get("width", 768)
        height = kwargs.get("height", 768)
        target_width = kwargs.get("target_width", width)
        target_height = kwargs.get("target_height", height)

        out['image_meta_size'] = comfy.conds.CONDRegular(torch.FloatTensor([[height, width, target_height, target_width, 0, 0]]))
        return out

class PixArt(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EPS, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.pixart.pixartms.PixArtMS)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        width = kwargs.get("width", None)
        height = kwargs.get("height", None)
        if width is not None and height is not None:
            out["c_size"] = comfy.conds.CONDRegular(torch.FloatTensor([[height, width]]))
            out["c_ar"] = comfy.conds.CONDRegular(torch.FloatTensor([[kwargs.get("aspect_ratio", height/width)]]))

        return out

class Flux(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None, unet_model=comfy.ldm.flux.model.Flux):
        super().__init__(model_config, model_type, device=device, unet_model=unet_model)

    def concat_cond(self, **kwargs):
        try:
            #Handle Flux control loras dynamically changing the img_in weight.
            num_channels = self.diffusion_model.img_in.weight.shape[1] // (self.diffusion_model.patch_size * self.diffusion_model.patch_size)
        except:
            #Some cases like tensorrt might not have the weights accessible
            num_channels = self.model_config.unet_config["in_channels"]

        out_channels = self.model_config.unet_config["out_channels"]

        if num_channels <= out_channels:
            return None

        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]

        if image is None:
            image = torch.zeros_like(noise)

        image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = utils.resize_to_batch_size(image, noise.shape[0])
        image = self.process_latent_in(image)
        if num_channels <= out_channels * 2:
            return image

        #inpaint model
        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.ones_like(noise)[:, :1]

        mask = torch.mean(mask, dim=1, keepdim=True)
        mask = utils.common_upscale(mask.to(device), noise.shape[-1] * 8, noise.shape[-2] * 8, "bilinear", "center")
        mask = mask.view(mask.shape[0], mask.shape[2] // 8, 8, mask.shape[3] // 8, 8).permute(0, 2, 4, 1, 3).reshape(mask.shape[0], -1, mask.shape[2] // 8, mask.shape[3] // 8)
        mask = utils.resize_to_batch_size(mask, noise.shape[0])
        return torch.cat((image, mask), dim=1)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        # upscale the attention mask, since now we
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            shape = kwargs["noise"].shape
            mask_ref_size = kwargs["attention_mask_img_shape"]
            # the model will pad to the patch size, and then divide
            # essentially dividing and rounding up
            (h_tok, w_tok) = (math.ceil(shape[2] / self.diffusion_model.patch_size), math.ceil(shape[3] / self.diffusion_model.patch_size))
            attention_mask = utils.upscale_dit_mask(attention_mask, mask_ref_size, (h_tok, w_tok))
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        guidance = kwargs.get("guidance", 3.5)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class GenmoMochi(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.genmo.joint_model.asymm_models_joint.AsymmDiTJoint)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
            out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class LTXV(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.lightricks.model.LTXVModel) #TODO

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        out['frame_rate'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", 25))

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)

        keyframe_idxs = kwargs.get("keyframe_idxs", None)
        if keyframe_idxs is not None:
            out['keyframe_idxs'] = comfy.conds.CONDRegular(keyframe_idxs)

        return out

    def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
        if denoise_mask is None:
            return timestep
        return self.diffusion_model.patchifier.patchify(((denoise_mask) * timestep.view([timestep.shape[0]] + [1] * (denoise_mask.ndim - 1)))[:, :1])[0]

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image

class HunyuanVideo(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan_video.model.HunyuanVideo)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 6.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))

        guiding_frame_index = kwargs.get("guiding_frame_index", None)
        if guiding_frame_index is not None:
            out['guiding_frame_index'] = comfy.conds.CONDRegular(torch.FloatTensor([guiding_frame_index]))

        ref_latent = kwargs.get("ref_latent", None)
        if ref_latent is not None:
            out['ref_latent'] = comfy.conds.CONDRegular(self.process_latent_in(ref_latent))

        return out

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return latent_image

class HunyuanVideoI2V(HunyuanVideo):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)
        self.concat_keys = ("concat_image", "mask_inverted")

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return super().scale_latent_inpaint(latent_image=latent_image, **kwargs)

class HunyuanVideoSkyreelsI2V(HunyuanVideo):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device)
        self.concat_keys = ("concat_image",)

    def scale_latent_inpaint(self, latent_image, **kwargs):
        return super().scale_latent_inpaint(latent_image=latent_image, **kwargs)

class CosmosVideo(BaseModel):
    def __init__(self, model_config, model_type=ModelType.EDM, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.cosmos.model.GeneralDIT)
        self.image_to_video = image_to_video
        if self.image_to_video:
            self.concat_keys = ("mask_inverted",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        out['fps'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", None))
        return out

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1))
        sigma_noise_augmentation = 0 #TODO
        if sigma_noise_augmentation != 0:
            latent_image = latent_image + noise
        latent_image = self.model_sampling.calculate_input(torch.tensor([sigma_noise_augmentation], device=latent_image.device, dtype=latent_image.dtype), latent_image)
        return latent_image * ((sigma ** 2 + self.model_sampling.sigma_data ** 2) ** 0.5)

class CosmosPredict2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW_COSMOS, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.cosmos.predict2.MiniTrainDIT)
        self.image_to_video = image_to_video
        if self.image_to_video:
            self.concat_keys = ("mask_inverted",)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        denoise_mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if denoise_mask is not None:
            out["denoise_mask"] = comfy.conds.CONDRegular(denoise_mask)

        out['fps'] = comfy.conds.CONDConstant(kwargs.get("frame_rate", None))
        return out

    def process_timestep(self, timestep, x, denoise_mask=None, **kwargs):
        if denoise_mask is None:
            return timestep
        if denoise_mask.ndim <= 4:
            return timestep
        condition_video_mask_B_1_T_1_1 = denoise_mask.mean(dim=[1, 3, 4], keepdim=True)
        c_noise_B_1_T_1_1 = 0.0 * (1.0 - condition_video_mask_B_1_T_1_1) + timestep.reshape(timestep.shape[0], 1, 1, 1, 1) * condition_video_mask_B_1_T_1_1
        out = c_noise_B_1_T_1_1.squeeze(dim=[1, 3, 4])
        return out

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        sigma = sigma.reshape([sigma.shape[0]] + [1] * (len(noise.shape) - 1))
        sigma_noise_augmentation = 0 #TODO
        if sigma_noise_augmentation != 0:
            latent_image = latent_image + noise
        latent_image = self.model_sampling.calculate_input(torch.tensor([sigma_noise_augmentation], device=latent_image.device, dtype=latent_image.dtype), latent_image)
        sigma = (sigma / (sigma + 1))
        return latent_image / (1.0 - sigma)

class Lumina2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.lumina.model.NextDiT)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            if torch.numel(attention_mask) != attention_mask.sum():
                out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)
            out['num_tokens'] = comfy.conds.CONDConstant(max(1, torch.sum(attention_mask).item()))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        return out

class WAN21(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.WanModel)
        self.image_to_video = image_to_video

    def concat_cond(self, **kwargs):
        noise = kwargs.get("noise", None)
        extra_channels = self.diffusion_model.patch_embedding.weight.shape[1] - noise.shape[1]
        if extra_channels == 0:
            return None

        image = kwargs.get("concat_latent_image", None)
        device = kwargs["device"]

        if image is None:
            shape_image = list(noise.shape)
            shape_image[1] = extra_channels
            image = torch.zeros(shape_image, dtype=noise.dtype, layout=noise.layout, device=noise.device)
        else:
            image = utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            for i in range(0, image.shape[1], 16):
                image[:, i: i + 16] = self.process_latent_in(image[:, i: i + 16])
            image = utils.resize_to_batch_size(image, noise.shape[0])

        if not self.image_to_video or extra_channels == image.shape[1]:
            return image

        if image.shape[1] > (extra_channels - 4):
            image = image[:, :(extra_channels - 4)]

        mask = kwargs.get("concat_mask", kwargs.get("denoise_mask", None))
        if mask is None:
            mask = torch.zeros_like(noise)[:, :4]
        else:
            if mask.shape[1] != 4:
                mask = torch.mean(mask, dim=1, keepdim=True)
            mask = 1.0 - mask
            mask = utils.common_upscale(mask.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
            if mask.shape[-3] < noise.shape[-3]:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, noise.shape[-3] - mask.shape[-3]), mode='constant', value=0)
            if mask.shape[1] == 1:
                mask = mask.repeat(1, 4, 1, 1, 1)
            mask = utils.resize_to_batch_size(mask, noise.shape[0])

        return torch.cat((mask, image), dim=1)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        clip_vision_output = kwargs.get("clip_vision_output", None)
        if clip_vision_output is not None:
            out['clip_fea'] = comfy.conds.CONDRegular(clip_vision_output.penultimate_hidden_states)

        time_dim_concat = kwargs.get("time_dim_concat", None)
        if time_dim_concat is not None:
            out['time_dim_concat'] = comfy.conds.CONDRegular(self.process_latent_in(time_dim_concat))

        return out


class WAN21_Vace(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.VaceWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        noise = kwargs.get("noise", None)
        noise_shape = list(noise.shape)
        vace_frames = kwargs.get("vace_frames", None)
        if vace_frames is None:
            noise_shape[1] = 32
            vace_frames = [torch.zeros(noise_shape, device=noise.device, dtype=noise.dtype)]

        mask = kwargs.get("vace_mask", None)
        if mask is None:
            noise_shape[1] = 64
            mask = [torch.ones(noise_shape, device=noise.device, dtype=noise.dtype)] * len(vace_frames)

        vace_frames_out = []
        for j in range(len(vace_frames)):
            vf = vace_frames[j].clone()
            for i in range(0, vf.shape[1], 16):
                vf[:, i:i + 16] = self.process_latent_in(vf[:, i:i + 16])
            vf = torch.cat([vf, mask[j]], dim=1)
            vace_frames_out.append(vf)

        vace_frames = torch.stack(vace_frames_out, dim=1)
        out['vace_context'] = comfy.conds.CONDRegular(vace_frames)

        vace_strength = kwargs.get("vace_strength", [1.0] * len(vace_frames_out))
        out['vace_strength'] = comfy.conds.CONDConstant(vace_strength)
        return out

class WAN21_Camera(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, image_to_video=False, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=comfy.ldm.wan.model.CameraWanModel)
        self.image_to_video = image_to_video

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        camera_conditions = kwargs.get("camera_conditions", None)
        if camera_conditions is not None:
            out['camera_conditions'] = comfy.conds.CONDRegular(camera_conditions)
        return out

class Hunyuan3Dv2(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hunyuan3d.model.Hunyuan3Dv2)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        guidance = kwargs.get("guidance", 5.0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class HiDream(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.hidream.model.HiDreamImageTransformer2DModel)

    def encode_adm(self, **kwargs):
        return kwargs["pooled_output"]

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)
        conditioning_llama3 = kwargs.get("conditioning_llama3", None)
        if conditioning_llama3 is not None:
            out['encoder_hidden_states_llama3'] = comfy.conds.CONDRegular(conditioning_llama3)
        image_cond = kwargs.get("concat_latent_image", None)
        if image_cond is not None:
            out['image_cond'] = comfy.conds.CONDNoiseShape(self.process_latent_in(image_cond))
        return out

class Chroma(Flux):
    def __init__(self, model_config, model_type=ModelType.FLUX, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.chroma.model.Chroma)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)

        guidance = kwargs.get("guidance", 0)
        if guidance is not None:
            out['guidance'] = comfy.conds.CONDRegular(torch.FloatTensor([guidance]))
        return out

class ACEStep(BaseModel):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super().__init__(model_config, model_type, device=device, unet_model=comfy.ldm.ace.model.ACEStepTransformer2DModel)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        noise = kwargs.get("noise", None)

        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDRegular(cross_attn)

        conditioning_lyrics = kwargs.get("conditioning_lyrics", None)
        if cross_attn is not None:
            out['lyric_token_idx'] = comfy.conds.CONDRegular(conditioning_lyrics)
        out['speaker_embeds'] = comfy.conds.CONDRegular(torch.zeros(noise.shape[0], 512, device=noise.device, dtype=noise.dtype))
        out['lyrics_strength'] = comfy.conds.CONDConstant(kwargs.get("lyrics_strength", 1.0))
        return out

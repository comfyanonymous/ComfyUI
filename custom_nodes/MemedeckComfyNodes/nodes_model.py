import json
import math
from pathlib import Path

import comfy
import comfy.model_management
import comfy.model_patcher

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import latent_preview
from .modules.model_2 import LTXVModel, LTXVModelConfig, LTXVTransformer3D


from ltx_video.models.autoencoders.causal_video_autoencoder import (
    CausalVideoAutoencoder,
)
from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

import folder_paths
import node_helpers
import torch
import safetensors.torch
from safetensors import safe_open


from .vae import MD_VideoVAE
from .stg import STGGuider
from ltx_video.models.autoencoders.vae_encode import get_vae_size_scale_factor

from .modules.img2vid import encode_media_conditioning
from .modules.model_2 import LTXVSampling

def get_normal_shift(
    n_tokens: int,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    m = (max_shift - min_shift) / (max_tokens - min_tokens)
    b = min_shift - m * min_tokens
    return m * n_tokens + b

class MD_LoadVideoModel:
    """
    Loads the DIT model for video generation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chkpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "default": "genesis-dit-video-2b.safetensors",
                    "tooltip": "The name of the checkpoint (model) to load."
                }),
                "clip_name": (folder_paths.get_filename_list("text_encoders"), {
                    "default": "t5xxl_fp16.safetensors",
                    "tooltip": "The name of the clip (model) to load."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "MemeDeck"
    
    # def load_model(self, chkpt_name, clip_name):
    #     ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", chkpt_name)
    #     out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
    #     model = out[0]
    #     vae = out[2]
        
    #     clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    #     clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=8)
        
    #     # modify model
    #     model.model.diffusion_model = inject_model(model.model.diffusion_model)
        
    #     return (model, clip, vae, )
    

    def load_model(self, chkpt_name, clip_name):
        dtype = torch.float32
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()

        ckpt_path = Path(folder_paths.get_full_path("checkpoints", chkpt_name))
        
        vae_config = None
        unet_config = None
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata is not None:
                config_metadata = metadata.get("config", None)
                if config_metadata is not None:
                    config_metadata = json.loads(config_metadata)
                    vae_config = config_metadata.get("vae", None)
                    unet_config = config_metadata.get("transformer", None)

        weights = safetensors.torch.load_file(ckpt_path, device="cpu")

        vae = self._load_vae(weights, vae_config)
        num_latent_channels = vae.first_stage_model.config.latent_channels

        model = self._load_unet(
            load_device,
            offload_device,
            weights,
            num_latent_channels,
            dtype=dtype,
            config=unet_config,
        )
        
        # Load clip
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=8)
        
        # modify model
        # model.model.diffusion_model = inject_model(model.model.diffusion_model)
        
        return (model, clip, vae, )
    
    # -----------------------------
    def _load_vae(self, weights, config=None):
        if config is None:
            config = {
                "_class_name": "CausalVideoAutoencoder",
                "dims": 3,
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 128,
                "blocks": [
                    ["res_x", 4],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x_y", 1],
                    ["res_x", 3],
                    ["compress_all", 1],
                    ["res_x", 3],
                    ["res_x", 4],
                ],
                "scaling_factor": 1.0,
                "norm_layer": "pixel_norm",
                "patch_size": 4,
                "latent_log_var": "uniform",
                "use_quant_conv": False,
                "causal_decoder": False,
            }
        vae_prefix = "vae."
        vae = MD_VideoVAE.from_config_and_state_dict(
            vae_class=CausalVideoAutoencoder,
            config=config,
            state_dict={
                key.removeprefix(vae_prefix): value
                for key, value in weights.items()
                if key.startswith(vae_prefix)
            },
        )
        return vae

    def _load_unet(
        self,
        load_device,
        offload_device,
        weights,
        num_latent_channels,
        dtype,
        config=None,
    ):
        if config is None:
            config = {
                "_class_name": "Transformer3DModel",
                "_diffusers_version": "0.25.1",
                "_name_or_path": "PixArt-alpha/PixArt-XL-2-256x256",
                "activation_fn": "gelu-approximate",
                "attention_bias": True,
                "attention_head_dim": 64,
                "attention_type": "default",
                "caption_channels": 4096,
                "cross_attention_dim": 2048,
                "double_self_attention": False,
                "dropout": 0.0,
                "in_channels": 128,
                "norm_elementwise_affine": False,
                "norm_eps": 1e-06,
                "norm_num_groups": 32,
                "num_attention_heads": 32,
                "num_embeds_ada_norm": 1000,
                "num_layers": 28,
                "num_vector_embeds": None,
                "only_cross_attention": False,
                "out_channels": 128,
                "project_to_2d_pos": True,
                "upcast_attention": False,
                "use_linear_projection": False,
                "qk_norm": "rms_norm",
                "standardization_norm": "rms_norm",
                "positional_embedding_type": "rope",
                "positional_embedding_theta": 10000.0,
                "positional_embedding_max_pos": [20, 2048, 2048],
                "timestep_scale_multiplier": 1000,
            }

        transformer = Transformer3DModel.from_config(config)
        unet_prefix = "model.diffusion_model."
        transformer.load_state_dict(
            {
                key.removeprefix(unet_prefix): value
                for key, value in weights.items()
                if key.startswith(unet_prefix)
            }
        )
        transformer.to(dtype).to(load_device).eval()
        patchifier = SymmetricPatchifier(1)
        diffusion_model = LTXVTransformer3D(transformer, patchifier, None, None, None)
        model = LTXVModel(
            LTXVModelConfig(num_latent_channels, dtype=dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=comfy.model_management.get_torch_device(),
        )
        model.diffusion_model = diffusion_model

        patcher = comfy.model_patcher.ModelPatcher(model, load_device, offload_device)

        return patcher

class LatentGuide(torch.nn.Module):
    def __init__(self, latent: torch.Tensor, index) -> None:
        super().__init__()
        self.index = index
        self.register_buffer('latent', latent)
        
class MD_ImgToVideo:
    """
    Sets the conditioning and dimensions for video generation.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 832,
                    "description": "The width of the video."
                }),
                "height": ("INT", {
                    "default": 832,
                    "description": "The height of the video."
                }),
                "length": ("INT", {
                    "default": 97,
                    "description": "The length of the video."
                }),
                "fps": ("INT", {
                    "default": 24,
                    "description": "The fps of the video."
                }),
                "crf": ("FLOAT", {
                    "default": 28.0,
                    "description": "The cfg of the image."
                }),
                # SCHEDULER INPUTS
                "steps": ("INT", {
                    "default": 40,
                    "description": "Number of steps to generate the video."
                }),
                "max_shift": ("FLOAT", {
                    "default": 1.5,
                    "step": 0.01,
                    "description": "The maximum shift of the video."
                }),
                "base_shift": ("FLOAT", {
                    "default": 0.95,
                    "step": 0.01,
                    "description": "The base shift of the video."
                }),
                "stretch": ("BOOLEAN", {
                    "default": True,
                    "description": "Stretch the sigmas to be in the range [terminal, 1]."
                }),
                "terminal": ("FLOAT", {
                    "default": 0.1,
                    "step": 0.01,
                    "description": "The terminal values of the sigmas after stretching."
                }),
                # ATTENTION OVERRIDE INPUTS
                "attention_override": ("STRING", {
                    "default": 14,
                    "description": "The amount of attention to override the model with."
                }),
                "attention_adjustment_scale": ("FLOAT", {
                    "default": 1.0,
                    "description": "The scale of the attention adjustment."
                }),
                "attention_adjustment_rescale": ("FLOAT", {
                    "default": 0.5,
                    "description": "The scale of the attention adjustment."
                }),
                "attention_adjustment_cfg": ("FLOAT", {
                    "default": 3.0,
                    "description": "The scale of the attention adjustment."
                }),
            },
        }
        
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SIGMAS", "LATENT", "STRING", "GUIDER")
    RETURN_NAMES = ("model", "positive", "negative", "sigmas", "latent", "img2vid_metadata", "guider")
    FUNCTION = "img_to_video"
    CATEGORY = "MemeDeck"

    def img_to_video(self, model, positive, negative, vae, image, width, height, length, fps, crf, steps, max_shift, base_shift, stretch, terminal, attention_override, attention_adjustment_scale, attention_adjustment_rescale, attention_adjustment_cfg):        
        batch_size = 1
        # pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        # # get pixels from image 
        # # encode_pixels = image[:, :, :, :3]
        # encode_pixels = pixels[:, :, :, :3]
        # t = vae.encode(encode_pixels)
        # positive = node_helpers.conditioning_set_values(positive, {"guiding_latent": t})
        # negative = node_helpers.conditioning_set_values(negative, {"guiding_latent": t})

        # # latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        # # latent[:, :, :t.shape[2]] = t
        # # latent_samples = {"samples": latent}
        
        # positive = node_helpers.conditioning_set_values(positive, {"frame_rate": fps})
        # negative = node_helpers.conditioning_set_values(negative, {"frame_rate": fps})
                
        # 1. apply stg
        model = self.apply_stg(model, "attention", attention_override)
        
        # 2. configure sizes
        model, latent_updated, sigma_shift = self.configure_sizes(model, vae, "Custom", width, height, length, fps, batch_size, mixed_precision=True, img_compression=crf, conditioning=image, initial_latent=None)

        # 3. shift sigmas - model, scheduler, steps, denoise
        scheduler_sigmas = self.get_sigmas(model, "normal", steps, 1.0)
        sigmas = self.shift_sigmas(scheduler_sigmas, sigma_shift, stretch, terminal)
        
        # 4. get guider
        guider = self.get_guider(model, positive, negative, cfg=attention_adjustment_cfg, stg=attention_adjustment_scale, rescale=attention_adjustment_rescale)
        
        # all parameters starting with width, height, fps, crf, etc
        img2vid_metadata = {
            "width": width,
            "height": height,
            "length": length,
            "fps": fps,
            "steps": steps,
            "max_shift": max_shift,
            "base_shift": base_shift,
            "stretch": stretch,
            "terminal": terminal,
            "attention_override": attention_override,
            "attention_adjustment_scale": attention_adjustment_scale,
            "attention_adjustment_rescale": attention_adjustment_rescale,
            "attention_adjustment_cfg": attention_adjustment_cfg,
        }

        json_img2vid_metadata = json.dumps(img2vid_metadata)
        return (model, positive, negative, sigmas, latent_updated, json_img2vid_metadata, guider)
    
    # 1 ------------------------------------------------------------
    def apply_stg(self, model, stg_mode: str, block_indices: str):
        skip_block_list = [int(i.strip()) for i in block_indices.split(",")]
        stg_mode = (
            SkipLayerStrategy.Attention
            if stg_mode == "attention"
            else SkipLayerStrategy.Residual
        )
        new_model = model.clone()

        new_model.model_options["transformer_options"]["skip_layer_strategy"] = stg_mode
        if "skip_block_list" in new_model.model_options["transformer_options"]:
            skip_block_list.extend(
                new_model.model_options["transformer_options"]["skip_block_list"]
            )
        new_model.model_options["transformer_options"][
            "skip_block_list"
        ] = skip_block_list

        return new_model
    
    # 2 ------------------------------------------------------------
    def configure_sizes(
        self,
        model,
        vae,
        preset,
        width,
        height,
        frames_number,
        frame_rate,
        batch,
        mixed_precision,
        img_compression,
        conditioning=None,
        initial_latent=None,
    ):
        load_device = comfy.model_management.get_torch_device()
        if preset != "Custom":
            preset = preset.split("|")
            width, height = map(int, preset[0].strip().split("x"))
            frames_number = int(preset[1].strip())
        latent_shape, latent_frame_rate = self.latent_shape_and_frame_rate(
            vae, batch, height, width, frames_number, frame_rate
        )
        mask_shape = [
            latent_shape[0],
            1,
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
        ]
        conditioning_mask = torch.zeros(mask_shape, device=load_device)
        initial_latent = (
            None
            if initial_latent is None
            else initial_latent["samples"].to(load_device)
        )
        guiding_latent = None
        if conditioning is not None:
            latent = encode_media_conditioning(
                conditioning,
                vae,
                width,
                height,
                frames_number,
                image_compression=img_compression,
                initial_latent=initial_latent,
            )
            conditioning_mask[:, :, 0] = 1.0
            guiding_latent = latent[:, :, :1, ...]
        else:
            latent = torch.zeros(latent_shape, dtype=torch.float32, device=load_device)
            if initial_latent is not None:
                latent[:, :, : initial_latent.shape[2], ...] = initial_latent

        _, vae_scale_factor, _ = get_vae_size_scale_factor(vae.first_stage_model)

        patcher = model.clone()
        patcher.add_object_patch("diffusion_model.conditioning_mask", conditioning_mask)
        patcher.add_object_patch("diffusion_model.latent_frame_rate", latent_frame_rate)
        patcher.add_object_patch("diffusion_model.vae_scale_factor", vae_scale_factor)
        patcher.add_object_patch(
            "model_sampling", LTXVSampling(conditioning_mask, guiding_latent)
        )
        patcher.model_options.setdefault("transformer_options", {})[
            "mixed_precision"
        ] = mixed_precision

        num_latent_patches = latent_shape[2] * latent_shape[3] * latent_shape[4]
        return (patcher, {"samples": latent}, get_normal_shift(num_latent_patches))
    
    def latent_shape_and_frame_rate(
        self, vae, batch, height, width, frames_number, frame_rate
    ):
        video_scale_factor, vae_scale_factor, _ = get_vae_size_scale_factor(
            vae.first_stage_model
        )
        video_scale_factor = video_scale_factor if frames_number > 1 else 1

        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latent_channels = vae.first_stage_model.config.latent_channels
        latent_num_frames = math.floor(frames_number / video_scale_factor) + 1
        latent_frame_rate = frame_rate / video_scale_factor

        latent_shape = [
            batch,
            latent_channels,
            latent_num_frames,
            latent_height,
            latent_width,
        ]
        return latent_shape, latent_frame_rate
    
    # 3 ------------------------------------------------------------
    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return sigmas
    
    def shift_sigmas(self, sigmas, sigma_shift, stretch, terminal):
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

        return sigmas
    
    # 4 ------------------------------------------------------------
    def get_guider(self, model, positive, negative, cfg, stg, rescale):
        guider = STGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg, stg, rescale)
        return guider
    
    # def img_to_video(self, model, positive, negative, vae, image, width, height, length, fps, add_latent_guide_index, add_latent_guide_insert, steps, max_shift, base_shift, stretch, terminal, attention_override, attention_adjustment_scale, attention_adjustment_rescale, attention_adjustment_cfg):        
    #     batch_size = 1
    #     pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    #     encode_pixels = pixels[:, :, :, :3]
    #     t = vae.encode(encode_pixels)
    #     positive = node_helpers.conditioning_set_values(positive, {"guiding_latent": t})
    #     negative = node_helpers.conditioning_set_values(negative, {"guiding_latent": t})

    #     latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
    #     latent[:, :, :t.shape[2]] = t
    #     latent_samples = {"samples": latent}
        
    #     positive = node_helpers.conditioning_set_values(positive, {"frame_rate": fps})
    #     negative = node_helpers.conditioning_set_values(negative, {"frame_rate": fps})
                
    #     # 2. add latent guide
    #     model, latent_updated = self.add_latent_guide(model, latent_samples, latent_samples, add_latent_guide_index, add_latent_guide_insert)
        
    #     # 3. apply attention override
    #     attn_override_layers = self.attention_override(attention_override)
    #     model = self.apply_attention_override(model, attention_adjustment_scale, attention_adjustment_rescale, attention_adjustment_cfg, attn_override_layers)
                
    #     # 4. configure scheduler
    #     sigmas = self.get_sigmas(steps, max_shift, base_shift, stretch, terminal, latent_updated)
        
    #     # all parameters starting with width, height, fps, crf, etc
    #     img2vid_metadata = {
    #         "width": width,
    #         "height": height,
    #         "length": length,
    #         "fps": fps,
    #         "steps": steps,
    #         "max_shift": max_shift,
    #         "base_shift": base_shift,
    #         "stretch": stretch,
    #         "terminal": terminal,
    #         "attention_override": attention_override,
    #         "attention_adjustment_scale": attention_adjustment_scale,
    #         "attention_adjustment_rescale": attention_adjustment_rescale,
    #         "attention_adjustment_cfg": attention_adjustment_cfg,
    #     }

    #     json_img2vid_metadata = json.dumps(img2vid_metadata)
    #     return (model, positive, negative, sigmas, latent_updated, json_img2vid_metadata)
    
    # -----------------------------
    # Attention functions
    # -----------------------------  
    # 1. Add latent guide
    # def add_latent_guide(self, model, latent, image_latent, index, insert):
    #     image_latent = image_latent['samples']
    #     latent = latent['samples'].clone()
        
    #     if insert:
    #         # Handle insertion
    #         if index == 0:
    #             # Insert at beginning
    #             latent = torch.cat([image_latent[:,:,0:1], latent], dim=2)
    #         elif index >= latent.shape[2] or index < 0:
    #             # Append to end
    #             latent = torch.cat([latent, image_latent[:,:,0:1]], dim=2)
    #         else:
    #             # Insert in middle
    #             latent = torch.cat([
    #                 latent[:,:,:index],
    #                 image_latent[:,:,0:1],
    #                 latent[:,:,index:]
    #             ], dim=2)
    #     else:
    #         # Original replacement behavior
    #         latent[:,:,index] = image_latent[:,:,0]
        
    #     model = model.clone()
    #     guiding_latent = LatentGuide(image_latent, index)
    #     model.set_model_patch(guiding_latent, 'guiding_latents')
        
    #     return (model, {"samples": latent},)
      
    # 2. Apply attention override
    # def is_integer(self, string):
    #     try:
    #         int(string)
    #         return True
    #     except ValueError:
    #         return False
    
    # def attention_override(self, layers: str = "14"):
    #     try:
    #         return set(map(int, layers.split(','))) 
    #     except ValueError:
    #         return set()
    
    # def apply_attention_override(self, model, scale, rescale, cfg, attention_override: set):
    #     m = model.clone()

    #     def pag_fn(q, k,v, heads, attn_precision=None, transformer_options=None):
    #         return v

    #     def post_cfg_function(args):
    #         model = args["model"]

    #         cond_pred = args["cond_denoised"]
    #         uncond_pred = args["uncond_denoised"]

    #         len_conds = 1 if args.get('uncond', None) is None else 2 
            
    #         cond = args["cond"]
    #         sigma = args["sigma"]
    #         model_options = args["model_options"].copy()
    #         x = args["input"]

    #         if scale == 0:
    #             if len_conds == 1:
    #                 return cond_pred
    #             return uncond_pred + (cond_pred - uncond_pred)
            
    #         for block_idx in attention_override:
    #             model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, pag_fn, f"layer", "self_attn", int(block_idx))

    #         (perturbed,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

    #         output = uncond_pred + cfg * (cond_pred - uncond_pred) \
    #             + scale * (cond_pred - perturbed)
    #         if rescale > 0:
    #             factor = cond_pred.std() / output.std()
    #             factor = rescale * factor + (1 - rescale)
    #             output = output * factor

    #         return output


    #     m.set_model_sampler_post_cfg_function(post_cfg_function)

    #     return m    
    
    # -----------------------------
    # Scheduler
    # -----------------------------  
    # def get_sigmas(self, steps, max_shift, base_shift, stretch, terminal, latent=None):
    #     if latent is None:
    #         tokens = 4096
    #     else:
    #         tokens = math.prod(latent["samples"].shape[2:])

    #     sigmas = torch.linspace(1.0, 0.0, steps + 1)

    #     x1 = 1024
    #     x2 = 4096
    #     mm = (max_shift - base_shift) / (x2 - x1)
    #     b = base_shift - mm * x1
    #     sigma_shift = (tokens) * mm + b

    #     power = 1
    #     sigmas = torch.where(
    #         sigmas != 0,
    #         math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1) ** power),
    #         0,
    #     )

    #     # Stretch sigmas so that its final value matches the given terminal value.
    #     if stretch:
    #         non_zero_mask = sigmas != 0
    #         non_zero_sigmas = sigmas[non_zero_mask]
    #         one_minus_z = 1.0 - non_zero_sigmas
    #         scale_factor = one_minus_z[-1] / (1.0 - terminal)
    #         stretched = 1.0 - (one_minus_z / scale_factor)
    #         sigmas[non_zero_mask] = stretched

    #     return sigmas

        
KSAMPLER_NAMES = ["euler", "ddim", "euler_ancestral", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis"]

class MD_VideoSampler:
    """
    Samples the video.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "sampler": (KSAMPLER_NAMES, ),
                "noise_seed": ("INT", {
                    "default": 42,
                    "description": "The seed of the noise."
                }),
                "cfg": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 30.0,
                    "step": 0.01,
                    "description": "The cfg of the video."
                }),
            },
        }
    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("output", "denoised_output", "img2vid_metadata")
    FUNCTION = "video_sampler"
    CATEGORY = "MemeDeck"
    
    def video_sampler(self, model, positive, negative, guider, sigmas, latent_image, sampler, noise_seed, cfg):
        # latent = latent_image
        # latent_image = latent["samples"]
        # latent = latent.copy()
        # latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        # latent["samples"] = latent_image
        
        # sampler_name = sampler
        # noise = Noise_RandomNoise(noise_seed).generate_noise(latent)
        # sampler = comfy.samplers.sampler_object(sampler)

        # noise_mask = None
        # if "noise_mask" in latent:
        #     noise_mask = latent["noise_mask"]

        # x0_output = {}
        # callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        # disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        # samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        # out = latent.copy()
        # out["samples"] = samples
        # if "x0" in x0_output:
        #     out_denoised = latent.copy()
        #     out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        # else:
        #     out_denoised = out
        
        noise = Noise_RandomNoise(noise_seed)
        
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        sampler_name = sampler
        sampler = comfy.samplers.sampler_object(sampler)
        
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        # return (out, out_denoised)
        
        sampler_metadata = {
            "sampler": sampler_name,
            "noise_seed": noise_seed,
            "cfg": cfg,
        }
        
        json_sampler_metadata = json.dumps(sampler_metadata)
        return (out, out_denoised, json_sampler_metadata)
import json
import math

from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
import latent_preview
from .modules.video_model import inject_model
import folder_paths
import node_helpers
import torch
import comfy

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
    
    def load_model(self, chkpt_name, clip_name):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", chkpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        vae = out[2]
        
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=8)
        
        # modify model
        model.model.diffusion_model = inject_model(model.model.diffusion_model)
        
        return (model, clip, vae, )

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
                # LATENT GUIDE INPUTS
                "add_latent_guide_index": ("INT", {
                    "default": 0,
                    "description": "The index of the latent to add to the guide."
                }),
                "add_latent_guide_insert": ("BOOLEAN", {
                    "default": False,
                    "description": "Whether to add the latent to the guide."
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
        
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "SIGMAS", "LATENT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "sigmas", "latent", "img2vid_metadata")
    FUNCTION = "img_to_video"
    CATEGORY = "MemeDeck"
    
    def img_to_video(self, model, positive, negative, vae, image, width, height, length, fps, add_latent_guide_index, add_latent_guide_insert, steps, max_shift, base_shift, stretch, terminal, attention_override, attention_adjustment_scale, attention_adjustment_rescale, attention_adjustment_cfg):        
        batch_size = 1
        pixels = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        positive = node_helpers.conditioning_set_values(positive, {"guiding_latent": t})
        negative = node_helpers.conditioning_set_values(negative, {"guiding_latent": t})

        latent = torch.zeros([batch_size, 128, ((length - 1) // 8) + 1, height // 32, width // 32], device=comfy.model_management.intermediate_device())
        latent[:, :, :t.shape[2]] = t
        latent_samples = {"samples": latent}
        
        positive = node_helpers.conditioning_set_values(positive, {"frame_rate": fps})
        negative = node_helpers.conditioning_set_values(negative, {"frame_rate": fps})
                
        # 2. add latent guide
        model, latent_updated = self.add_latent_guide(model, latent_samples, latent_samples, add_latent_guide_index, add_latent_guide_insert)
        
        # 3. apply attention override
        attn_override_layers = self.attention_override(attention_override)
        model = self.apply_attention_override(model, attention_adjustment_scale, attention_adjustment_rescale, attention_adjustment_cfg, attn_override_layers)
                
        # 5. configure scheduler
        sigmas = self.get_sigmas(steps, max_shift, base_shift, stretch, terminal, latent_updated)
        
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
        return (model, positive, negative, sigmas, latent_updated, json_img2vid_metadata)
    
    # -----------------------------
    # Attention functions
    # -----------------------------  
    # 1. Add latent guide
    def add_latent_guide(self, model, latent, image_latent, index, insert):
        image_latent = image_latent['samples']
        latent = latent['samples'].clone()
        
        # # Convert negative index to positive
        # if insert:
        #     index = max(0, min(index, latent.shape[2]))  # Clamp index
        #     latent = torch.cat([
        #         latent[:,:,:index],
        #         image_latent[:,:,0:1],
        #         latent[:,:,index:]
        #     ], dim=2)
        # else:
        #     latent[:,:,index] = image_latent[:,:,0]
        if insert:
            # Handle insertion
            if index == 0:
                # Insert at beginning
                latent = torch.cat([image_latent[:,:,0:1], latent], dim=2)
            elif index >= latent.shape[2] or index < 0:
                # Append to end
                latent = torch.cat([latent, image_latent[:,:,0:1]], dim=2)
            else:
                # Insert in middle
                latent = torch.cat([
                    latent[:,:,:index],
                    image_latent[:,:,0:1],
                    latent[:,:,index:]
                ], dim=2)
        else:
            # Original replacement behavior
            latent[:,:,index] = image_latent[:,:,0]
        
        model = model.clone()
        guiding_latent = LatentGuide(image_latent, index)
        model.set_model_patch(guiding_latent, 'guiding_latents')
        
        return (model, {"samples": latent},)
      
    # 2. Apply attention override
    def is_integer(self, string):
        try:
            int(string)
            return True
        except ValueError:
            return False
    
    def attention_override(self, layers: str = "14"):
        try:
            return set(map(int, layers.split(','))) 
        except ValueError:
            return set()
        
        # layers_map = set([])
        # return set(map(int, layers.split(','))) 
        # for block in layers.split(','):
        #     block = block.strip()
        #     if self.is_integer(block):
        #         layers_map.add(block)

        # return layers_map
    
    def apply_attention_override(self, model, scale, rescale, cfg, attention_override: set):
        m = model.clone()

        def pag_fn(q, k,v, heads, attn_precision=None, transformer_options=None):
            return v

        def post_cfg_function(args):
            model = args["model"]

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            len_conds = 1 if args.get('uncond', None) is None else 2 
            
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                if len_conds == 1:
                    return cond_pred
                return uncond_pred + (cond_pred - uncond_pred)
            
            for block_idx in attention_override:
                model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, pag_fn, f"layer", "self_attn", int(block_idx))

            (perturbed,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            output = uncond_pred + cfg * (cond_pred - uncond_pred) \
                + scale * (cond_pred - perturbed)
            if rescale > 0:
                factor = cond_pred.std() / output.std()
                factor = rescale * factor + (1 - rescale)
                output = output * factor

            return output


        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return m    
    
    # -----------------------------
    # Scheduler
    # -----------------------------  
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

        return sigmas

        
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
    
    def video_sampler(self, model, positive, negative, sigmas, latent_image, sampler, noise_seed, cfg):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image
        
        sampler_name = sampler
        noise = Noise_RandomNoise(noise_seed).generate_noise(latent)
        sampler = comfy.samplers.sampler_object(sampler)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        
        sampler_metadata = {
            "sampler": sampler_name,
            "noise_seed": noise_seed,
            "cfg": cfg,
        }
        
        json_sampler_metadata = json.dumps(sampler_metadata)
        return (out, out_denoised, json_sampler_metadata)
import torch
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models
from comfy.sd import CLIP, VAE
from comfy.model_patcher import ModelPatcher
from PIL import Image

from nodes import MAX_RESOLUTION, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS, ConditioningConcat, CLIPTextEncode, ConditioningZeroOut

from ..libs.log import log_node_info, log_node_error, log_node_warn
from ..libs.wildcards import process_with_loras
from ..libs.utils import find_wildcards_seed, is_linked_styles_selector, get_sd_version, AlwaysEqualProxy
from ..libs.sampler import easySampler
from ..libs.controlnet import easyControlnet, union_controlnet_types
from ..libs.conditioning import prompt_to_cond
from ..libs.easing import EasingBase
from ..libs.translate import has_chinese, zh_to_en

from ..config import *

from .. import easyCache, sampler

any_type = AlwaysEqualProxy("*")
# 简易加载器完整
resolution_strings = [f"{width} x {height} (custom)" if width == 'width' and height == 'height' else f"{width} x {height}" for width, height in BASE_RESOLUTIONS]
class fullLoader:

    @classmethod
    def INPUT_TYPES(cls):
        a1111_prompt_style_default = False

        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints") + ['None'],),
            "config_name": (["Default", ] + folder_paths.get_filename_list("configs"), {"default": "Default"}),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings, {"default": "512 x 512"}),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
            "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
            "positive_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
            "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
            "negative_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "batch_size": (
            "INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
        },
            "optional": {"model_override": ("MODEL",), "clip_override": ("CLIP",), "vae_override": ("VAE",), "optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",), "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default})},
            "hidden": {"video_length": "INT", "prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE", "CLIP", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("pipe", "model", "vae", "clip", "positive", "negative", "latent")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, positive_token_normalization, positive_weight_interpretation,
                       negative, negative_token_normalization, negative_weight_interpretation,
                       batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, video_length=25, prompt=None,
                       my_unique_id=None
                       ):

        if ckpt_name == 'None' and model_override is None:
            raise Exception("Please select a checkpoint or provide a model override.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Load models
        log_node_warn("Loading models...")
        model, clip, vae, clip_vision, lora_stack = easyCache.load_main(ckpt_name, config_name, vae_name, lora_name, lora_model_strength, lora_clip_strength, optional_lora_stack, model_override, clip_override, vae_override, prompt)

        # Create Empty Latent
        model_type = get_sd_version(model)
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, model_type=model_type, video_length=video_length)

        # Prompt to Conditioning
        positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip, lora_stack, positive, positive_token_normalization, positive_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache, model_type=model_type)
        negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip, clip_skip, lora_stack, negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache, model_type=model_type)

        if negative_embeddings_final is None:
            negative_embeddings_final, = ConditioningZeroOut().zero_out(positive_embeddings_final)

        # Conditioning add controlnet
        if optional_controlnet_stack is not None and len(optional_controlnet_stack) > 0:
            for controlnet in optional_controlnet_stack:
                positive_embeddings_final, negative_embeddings_final = easyControlnet().apply(controlnet[0], controlnet[5], positive_embeddings_final, negative_embeddings_final, controlnet[1], start_percent=controlnet[2], end_percent=controlnet[3], control_net=None, scale_soft_weights=controlnet[4], mask=None, easyCache=easyCache, use_cache=True, model=model, vae=vae)

        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": None,

            "loader_settings": {
                "ckpt_name": ckpt_name,
                "vae_name": vae_name,
                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,
                "lora_stack": lora_stack,

                "clip_skip": clip_skip,
                "a1111_prompt_style": a1111_prompt_style,
                "positive": positive,
                "positive_token_normalization": positive_token_normalization,
                "positive_weight_interpretation": positive_weight_interpretation,
                "negative": negative,
                "negative_token_normalization": negative_token_normalization,
                "negative_weight_interpretation": negative_weight_interpretation,
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt}, "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}

# A1111简易加载器
class a1111Loader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        a1111_prompt_style_default = False
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (checkpoints,),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                "lora_name": (loras,),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "a1111loader"
    CATEGORY = "EasyUse/Loaders"

    def a1111loader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None):

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'mean', 'A1111',
             negative,'mean','A1111',
             batch_size, None, None,  None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack,a1111_prompt_style=a1111_prompt_style, prompt=prompt,
             my_unique_id=my_unique_id
        )

# Comfy简易加载器
class comfyLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            },
            "optional": {"optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "comfyloader"
    CATEGORY = "EasyUse/Loaders"

    def comfyloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, prompt=None,
                      my_unique_id=None):
        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative, 'none', 'comfy',
             batch_size, None, None, None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack, a1111_prompt_style=False, prompt=prompt,
             my_unique_id=my_unique_id
         )

# hydit简易加载器
class hunyuanDiTLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {"optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "hyditloader"
    CATEGORY = "EasyUse/Loaders"

    def hyditloader(self, ckpt_name, vae_name,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, prompt=None,
                      my_unique_id=None):

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, 0,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative, 'none', 'comfy',
             batch_size, None, None, None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack, a1111_prompt_style=False, prompt=prompt,
             my_unique_id=my_unique_id
         )

# stable Cascade
class cascadeLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {"required": {
            "stage_c": (folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("checkpoints"),),
            "stage_b": (folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("checkpoints"),),
            "stage_a": (["Baked VAE"]+folder_paths.get_filename_list("vae"),),
            "clip_name": (["None"] + folder_paths.get_filename_list("clip"),),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings, {"default": "1024 x 1024"}),
            "empty_latent_width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "compression": ("INT", {"default": 42, "min": 32, "max": 64, "step": 1}),

            "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
            "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"optional_lora_stack": ("LORA_STACK",), },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "LATENT", "VAE")
    RETURN_NAMES = ("pipe", "model_c", "latent_c", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def is_ckpt(self, name):
        is_ckpt = False
        path = folder_paths.get_full_path("checkpoints", name)
        if path is not None:
            is_ckpt = True
        return is_ckpt

    def adv_pipeloader(self, stage_c, stage_b, stage_a, clip_name, lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height, compression,
                       positive, negative, batch_size, optional_lora_stack=None,prompt=None,
                       my_unique_id=None):

        vae: VAE | None = None
        model_c: ModelPatcher | None = None
        model_b: ModelPatcher | None = None
        clip: CLIP | None = None
        can_load_lora = True
        pipe_lora_stack = []

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Create Empty Latent
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, compression)

        if self.is_ckpt(stage_c):
            model_c, clip, vae_c, clip_vision = easyCache.load_checkpoint(stage_c)
        else:
            model_c = easyCache.load_unet(stage_c)
            vae_c = None
        if self.is_ckpt(stage_b):
            model_b, clip, vae_b, clip_vision = easyCache.load_checkpoint(stage_b)
        else:
            model_b = easyCache.load_unet(stage_b)
            vae_b = None

        if optional_lora_stack is not None and can_load_lora:
            for lora in optional_lora_stack:
                lora = {"lora_name": lora[0], "model": model_c, "clip": clip, "model_strength": lora[1], "clip_strength": lora[2]}
                model_c, clip = easyCache.load_lora(lora)
                lora['model'] = model_c
                lora['clip'] = clip
                pipe_lora_stack.append(lora)

        if lora_name != "None" and can_load_lora:
            lora = {"lora_name": lora_name, "model": model_c, "clip": clip, "model_strength": lora_model_strength,
                    "clip_strength": lora_clip_strength}
            model_c, clip = easyCache.load_lora(lora)
            pipe_lora_stack.append(lora)

        model = (model_c, model_b)
        # Load clip
        if clip_name != 'None':
            clip = easyCache.load_clip(clip_name, "stable_cascade")
        # Load vae
        if stage_a not in ["Baked VAE", "Baked-VAE"]:
            vae_b = easyCache.load_vae(stage_a)

        vae = (vae_c, vae_b)
        # 判断是否连接 styles selector
        is_positive_linked_styles_selector = is_linked_styles_selector(prompt, my_unique_id, 'positive')
        is_negative_linked_styles_selector = is_linked_styles_selector(prompt, my_unique_id, 'negative')

        positive_seed = find_wildcards_seed(my_unique_id, positive, prompt)
        # Translate cn to en
        if has_chinese(positive):
            positive = zh_to_en([positive])[0]
        model_c, clip, positive, positive_decode, show_positive_prompt, pipe_lora_stack = process_with_loras(positive,
                                                                                                           model_c, clip,
                                                                                                           "positive",
                                                                                                           positive_seed,
                                                                                                           can_load_lora,
                                                                                                           pipe_lora_stack,
                                                                                                           easyCache)
        positive_wildcard_prompt = positive_decode if show_positive_prompt or is_positive_linked_styles_selector else ""
        negative_seed = find_wildcards_seed(my_unique_id, negative, prompt)
        # Translate cn to en
        if has_chinese(negative):
            negative = zh_to_en([negative])[0]
        model_c, clip, negative, negative_decode, show_negative_prompt, pipe_lora_stack = process_with_loras(negative,
                                                                                                           model_c, clip,
                                                                                                           "negative",
                                                                                                           negative_seed,
                                                                                                           can_load_lora,
                                                                                                           pipe_lora_stack,
                                                                                                           easyCache)
        negative_wildcard_prompt = negative_decode if show_negative_prompt or is_negative_linked_styles_selector else ""

        tokens = clip.tokenize(positive)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        positive_embeddings_final = [[cond, {"pooled_output": pooled}]]

        tokens = clip.tokenize(negative)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        negative_embeddings_final = [[cond, {"pooled_output": pooled}]]

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": 0,

            "loader_settings": {
                "vae_name": stage_a,
                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,
                "lora_stack": pipe_lora_stack,

                "positive": positive,
                "positive_token_normalization": 'none',
                "positive_weight_interpretation": 'comfy',
                "negative": negative,
                "negative_token_normalization": 'none',
                "negative_weight_interpretation": 'comfy',
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
                "compression": compression
            }
        }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt},
                "result": (pipe, model_c, model_b, vae)}

# Zero123简易加载器 (3D)
try:
    from comfy_extras.nodes_stable3d import camera_embeddings
except FileNotFoundError:
    log_node_error("EasyUse[zero123Loader]", "请更新ComfyUI到最新版本")

class zero123Loader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "zero123" in file.lower()]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),

            "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
            "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, elevation, azimuth, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    }
                }

        return (pipe, model, vae)

# SV3D加载器
class sv3dLoader(EasingBase):

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "sv3d" in file]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 576, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 21, "min": 1, "max": 4096}),
            "interp_easing": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
            "easing_mode": (["azimuth", "elevation", "custom"], {"default": "azimuth"}),
        },
            "optional": {"scheduler": ("STRING", {"default": "",  "multiline": True})},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "STRING")
    RETURN_NAMES = ("pipe", "model", "interp_log")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, interp_easing, easing_mode, scheduler='',prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1,
                                                                                                                    -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)

        azimuth_points = []
        elevation_points = []
        if easing_mode == 'azimuth':
            azimuth_points = [(0, 0), (batch_size-1, 360)]
            elevation_points = [(0, 0)] * batch_size
        elif easing_mode == 'elevation':
            azimuth_points = [(0, 0)] * batch_size
            elevation_points = [(0, -90), (batch_size-1, 90)]
        else:
            schedulers = scheduler.rstrip('\n')
            for line in schedulers.split('\n'):
                frame_str, point_str = line.split(':')
                point_str = point_str.strip()[1:-1]
                point = point_str.split(',')
                azimuth_point = point[0]
                elevation_point = point[1] if point[1] else 0.0
                frame = int(frame_str.strip())
                azimuth = float(azimuth_point)
                azimuth_points.append((frame, azimuth))
                elevation_val = float(elevation_point)
                elevation_points.append((frame, elevation_val))
            azimuth_points.sort(key=lambda x: x[0])
            elevation_points.sort(key=lambda x: x[0])

        #interpolation
        next_point = 1
        next_elevation_point = 1
        elevations = []
        azimuths = []
        # For azimuth interpolation
        for i in range(batch_size):
            # Find the interpolated azimuth for the current frame
            while next_point < len(azimuth_points) and i >= azimuth_points[next_point][0]:
                next_point += 1
            if next_point == len(azimuth_points):
                next_point -= 1
            prev_point = max(next_point - 1, 0)

            if azimuth_points[next_point][0] != azimuth_points[prev_point][0]:
                timing = (i - azimuth_points[prev_point][0]) / (
                            azimuth_points[next_point][0] - azimuth_points[prev_point][0])
                interpolated_azimuth = self.ease(azimuth_points[prev_point][1], azimuth_points[next_point][1], self.easing(timing, interp_easing))
            else:
                interpolated_azimuth = azimuth_points[prev_point][1]

            # Interpolate the elevation
            next_elevation_point = 1
            while next_elevation_point < len(elevation_points) and i >= elevation_points[next_elevation_point][0]:
                next_elevation_point += 1
            if next_elevation_point == len(elevation_points):
                next_elevation_point -= 1
            prev_elevation_point = max(next_elevation_point - 1, 0)

            if elevation_points[next_elevation_point][0] != elevation_points[prev_elevation_point][0]:
                timing = (i - elevation_points[prev_elevation_point][0]) / (
                            elevation_points[next_elevation_point][0] - elevation_points[prev_elevation_point][0])
                interpolated_elevation = self.ease(elevation_points[prev_point][1], elevation_points[next_point][1], self.easing(timing, interp_easing))
            else:
                interpolated_elevation = elevation_points[prev_elevation_point][1]

            azimuths.append(interpolated_azimuth)
            elevations.append(interpolated_elevation)

        log_node_info("easy sv3dLoader", "azimuths:" + str(azimuths))
        log_node_info("easy sv3dLoader", "elevations:" + str(elevations))

        log = 'azimuths:' + str(azimuths) + '\n\n' + "elevations:" + str(elevations)
        # Structure the final output
        positive = [[pooled, {"concat_latent_image": t, "elevation": elevations, "azimuth": azimuths}]]
        negative = [[torch.zeros_like(pooled),
                           {"concat_latent_image": torch.zeros_like(t), "elevation": elevations, "azimuth": azimuths}]]

        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))


        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    }
                }

        return (pipe, model, log)

#svd加载器
class svdLoader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "svd" in file.lower()]

        return {"required": {
                "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_name": (["None"] + folder_paths.get_filename_list("clip"),),

                "init_image": ("IMAGE",),
                "resolution": (resolution_strings, {"default": "1024 x 576"}),
                "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

                "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "optional_positive": ("STRING", {"default": "", "multiline": True}),
                "optional_negative": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_name, init_image, resolution, empty_latent_width, empty_latent_height, video_frames, motion_bucket_id, fps, augmentation_level, optional_positive=None, optional_negative=None, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled,
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled),
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": torch.zeros_like(t)}]]
        if optional_positive is not None and optional_positive != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when positive is not empty")
            clip = easyCache.load_clip(clip_name)
            if has_chinese(optional_positive):
                optional_positive = zh_to_en([optional_positive])[0]
            positive_embeddings_final, = CLIPTextEncode().encode(clip, optional_positive)
            positive, = ConditioningConcat().concat(positive, positive_embeddings_final)
        if optional_negative is not None and optional_negative != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when negative is not empty")
            if has_chinese(optional_negative):
                optional_positive = zh_to_en([optional_negative])[0]
            negative_embeddings_final, = CLIPTextEncode().encode(clip, optional_negative)
            negative, = ConditioningConcat().concat(negative, negative_embeddings_final)

        latent = torch.zeros([video_frames, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "resolution": resolution,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                     }
                }

        return (pipe, model, vae)


# kolors Loader
from ..modules.kolors.text_encode import chatglm3_adv_text_encode
class kolorsLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "chatglm3_name": (folder_paths.get_filename_list("llm"),),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "resolution": (resolution_strings, {"default": "1024 x 576"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "model_override": ("MODEL",),
                "vae_override": ("VAE",),
                "optional_lora_stack": ("LORA_STACK",),
                "auto_clean_gpu": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, unet_name, vae_name, chatglm3_name, lora_name, lora_model_strength, lora_clip_strength, resolution, empty_latent_width, empty_latent_height, positive, negative, batch_size, model_override=None, optional_lora_stack=None, vae_override=None, auto_clean_gpu=False, prompt=None, my_unique_id=None):
        # load unet
        if model_override:
           model = model_override
        else:
           model = easyCache.load_kolors_unet(unet_name)
        # load vae
        if vae_override:
           vae = vae_override
        else:
           vae = easyCache.load_vae(vae_name)
        # load chatglm3
        chatglm3_model = easyCache.load_chatglm3(chatglm3_name)
        # load lora
        lora_stack = []
        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                lora = {"lora_name": lora[0], "model": model, "clip": None, "model_strength": lora[1],
                        "clip_strength": lora[2]}
                model, _ = easyCache.load_lora(lora)
                lora['model'] = model
                lora['clip'] = None
                lora_stack.append(lora)

        if lora_name != "None":
            lora = {"lora_name": lora_name, "model": model, "clip": None, "model_strength": lora_model_strength,
                    "clip_strength": lora_clip_strength}
            model, _ = easyCache.load_lora(lora)
            lora_stack.append(lora)


        # text encode
        log_node_warn("Positive encoding...")
        positive_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, positive, auto_clean_gpu)
        log_node_warn("Negative encoding...")
        negative_embeddings_final = chatglm3_adv_text_encode(chatglm3_model, negative, auto_clean_gpu)

        # empty latent
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size)

        pipe = {
            "model": model,
            "chatglm3_model": chatglm3_model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": None,

            "samples": samples,
            "images": None,

            "loader_settings": {
                "unet_name": unet_name,
                "vae_name": vae_name,
                "chatglm3_name": chatglm3_name,

                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,

                "positive": positive,
                "negative": negative,
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
                "auto_clean_gpu": auto_clean_gpu,
            }
        }

        return {"ui": {},
                "result": (pipe, model, vae, chatglm3_model, positive_embeddings_final, negative_embeddings_final, samples)}


        return (chatglm3_model, None, None)

# Flux Loader
class fluxLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (checkpoints + ['None'],),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "lora_name": (loras,),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "resolution": (resolution_strings, {"default": "1024 x 1024"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "model_override": ("MODEL",),
                "clip_override": ("CLIP",),
                "vae_override": ("VAE",),
                "optional_lora_stack": ("LORA_STACK",),
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "fluxloader"
    CATEGORY = "EasyUse/Loaders"

    def fluxloader(self, ckpt_name, vae_name,
                    lora_name, lora_model_strength, lora_clip_strength,
                    resolution, empty_latent_width, empty_latent_height,
                    positive, batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, optional_controlnet_stack=None,
                    a1111_prompt_style=False, prompt=None,
                    my_unique_id=None):

        if positive == '':
            positive = None

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, 0,
                                      lora_name, lora_model_strength, lora_clip_strength,
                                      resolution, empty_latent_width, empty_latent_height,
                                      positive, 'none', 'comfy',
                                      None, 'none', 'comfy',
                                      batch_size, model_override, clip_override, vae_override, optional_lora_stack=optional_lora_stack,
                                      optional_controlnet_stack=optional_controlnet_stack,
                                      a1111_prompt_style=a1111_prompt_style, prompt=prompt,
                                      my_unique_id=my_unique_id)


# Dit Loader
from ..modules.dit.pixArt.config import pixart_conf, pixart_res

class pixArtLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_name":(list(pixart_conf.keys()),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "t5_type": (['sd3'],),
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "padding": ("INT", {"default": 1, "min": 1, "max": 300}),
                "t5_name": (folder_paths.get_filename_list("t5"),),
                "device": (["auto", "cpu", "gpu"], {"default": "cpu"}),
                "dtype": (["default", "auto (comfy)", "FP32", "FP16", "BF16"],),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "ratio": (["custom"] + list(pixart_res["PixArtMS_XL_2"].keys()), {"default":"1.00"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional":{
              "optional_lora_stack": ("LORA_STACK",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")
    FUNCTION = "pixart_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def pixart_pipeloader(self, ckpt_name, model_name, vae_name, t5_type, clip_name, padding, t5_name, device, dtype, lora_name, lora_model_strength, ratio, empty_latent_width, empty_latent_height, positive, negative, batch_size, optional_lora_stack=None, prompt=None, my_unique_id=None):
        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # load checkpoint
        model = easyCache.load_dit_ckpt(ckpt_name=ckpt_name, model_name=model_name, pixart_conf=pixart_conf,
                                        model_type='PixArt')
        # load vae
        vae = easyCache.load_vae(vae_name)

        # load t5
        if t5_type == 'sd3':
            clip = easyCache.load_clip(clip_name=clip_name,type='sd3')
            clip = easyCache.load_t5_from_sd3_clip(sd3_clip=clip, padding=padding)
            lora_stack = None
            if optional_lora_stack is not None:
                for lora in optional_lora_stack:
                    lora = {"lora_name": lora[0], "model": model, "clip": clip, "model_strength": lora[1],
                            "clip_strength": lora[2]}
                    model, _ = easyCache.load_lora(lora, type='PixArt')
                    lora['model'] = model
                    lora['clip'] = clip
                    lora_stack.append(lora)

            if lora_name != "None":
                lora = {"lora_name": lora_name, "model": model, "clip": clip, "model_strength": lora_model_strength,
                        "clip_strength": 1}
                model, _ = easyCache.load_lora(lora, type='PixArt')
                lora_stack.append(lora)

            positive_embeddings_final, = CLIPTextEncode().encode(clip, positive)
            negative_embeddings_final, = CLIPTextEncode().encode(clip, negative)
        else:
            # todo t5v11
            positive_embeddings_final, negative_embeddings_final = None, None
            clip = None
            pass

        # Create Empty Latent
        if ratio != 'custom':
            if model_name in ['ControlPixArtMSHalf','PixArtMS_Sigma_XL_2_900M']:
                res_name = 'PixArtMS_XL_2'
            elif model_name in ['ControlPixArtHalf']:
                res_name = 'PixArt_XL_2'
            else:
                res_name = model_name
            width, height = pixart_res[res_name][ratio]
            empty_latent_width = width
            empty_latent_height = height

        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8], device=sampler.device)
        samples = {"samples": latent}

        log_node_warn("加载完毕...")
        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": None,

            "loader_settings": {
                "ckpt_name": ckpt_name,
                "clip_name": clip_name,
                "vae_name": vae_name,
                "t5_name": t5_name,

                "positive": positive,
                "negative": negative,
                "ratio": ratio,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {},
                "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}


# Mochi加载器
class mochiLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name": (checkpoints,),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"), {"default": "mochi_vae.safetensors"}),

                "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),

                "resolution": (resolution_strings, {"default": "width x height (custom)"}),
                "empty_latent_width": ("INT", {"default": 848, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 480, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "length": ("INT", {"default": 25, "min": 7, "max": MAX_RESOLUTION, "step": 6}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            },
            "optional": {
                "model_override": ("MODEL",), "clip_override": ("CLIP",), "vae_override": ("VAE",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "mochiLoader"
    CATEGORY = "EasyUse/Loaders"

    def mochiLoader(self, ckpt_name, vae_name,
                       positive, negative,
                       resolution, empty_latent_width, empty_latent_height,
                       length, batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None):

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, 0,
             "None", 1.0, 1.0,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative,'none','comfy',
             batch_size, model_override, clip_override,  vae_override, a1111_prompt_style=False, video_length=length, prompt=prompt,
             my_unique_id=my_unique_id
        )
# lora
class loraSwitcher:
    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 50
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "on", "label_off": "off"}),
                "select": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num + 1):
            inputs["optional"][f"lora_{i}_name"] = (
                ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})

        return inputs

    RETURN_TYPES = ("LORA_STACK", any_type)
    RETURN_NAMES = ("lora_stack", "lora_name")
    FUNCTION = "stack"

    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, select,num_loras, lora_strength, optional_lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,'')

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

        # Import Lora values
        lora_name = kwargs.get(f"lora_{select}_name")

        if not lora_name or lora_name == "None":
            return (None,'')

        loras.append((lora_name, lora_strength, lora_strength))

        name = os.path.splitext(os.path.basename(str(lora_name)))[0]
        return (loras, name)


class loraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "on", "label_off": "off"}),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_model_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_loras, optional_lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        return (loras,)

class controlnetStack:


    @classmethod
    def INPUT_TYPES(s):
        max_cn_num = 3
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_controlnet": ("INT", {"default": 1, "min": 1, "max": max_cn_num}),
            },
            "optional": {
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
            }
        }

        for i in range(1, max_cn_num+1):
            inputs["optional"][f"controlnet_{i}"] = (["None"] + folder_paths.get_filename_list("controlnet"), {"default": "None"})
            inputs["optional"][f"controlnet_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},)
            inputs["optional"][f"start_percent_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"end_percent_{i}"] = ("FLOAT",{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"scale_soft_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("controlnet_stack",)
    FUNCTION = "stack"
    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_controlnet, optional_controlnet_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        controlnets = []

        # Import Stack values
        if optional_controlnet_stack is not None:
            controlnets.extend([l for l in optional_controlnet_stack if l[0] != "None"])

        # Import Controlnet values
        for i in range(1, num_controlnet+1):
            controlnet_name = kwargs.get(f"controlnet_{i}")

            if not controlnet_name or controlnet_name == "None":
                continue

            controlnet_strength = float(kwargs.get(f"controlnet_{i}_strength"))
            start_percent = float(kwargs.get(f"start_percent_{i}")) if mode == "advanced" else 0
            end_percent = float(kwargs.get(f"end_percent_{i}")) if mode == "advanced" else 1.0
            scale_soft_weights = float(kwargs.get(f"scale_soft_weight_{i}"))
            image = kwargs.get(f"image_{i}")

            controlnets.append((controlnet_name, controlnet_strength, start_percent, end_percent, scale_soft_weights, image, True))

        return (controlnets,)
# controlnet
class controlnetSimple:
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"

    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, scale_soft_weights=1, union_type=None):

        positive, negative = easyControlnet().apply(control_net_name, image, pipe["positive"], pipe["negative"], strength, 0, 1, control_net, scale_soft_weights, mask=None, easyCache=easyCache, model=pipe['model'], vae=pipe['vae'])

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe
        return (new_pipe, positive, negative)

# controlnetADV
class controlnetAdvanced:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"


    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, start_percent=0, end_percent=1, scale_soft_weights=1):
        positive, negative = easyControlnet().apply(control_net_name, image, pipe["positive"], pipe["negative"],
                                                    strength, start_percent, end_percent, control_net, scale_soft_weights, union_type=None, mask=None, easyCache=easyCache, model=pipe['model'], vae=pipe['vae'])

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": image,
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe

        return (new_pipe, positive, negative)

# controlnetPlusPlus
class controlnetPlusPlus:

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
            },
            "optional": {
                "control_net": ("CONTROL_NET",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_soft_weights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                "union_type": (list(union_controlnet_types.keys()),)
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("pipe", "positive", "negative")

    FUNCTION = "controlnetApply"
    CATEGORY = "EasyUse/Loaders"


    def controlnetApply(self, pipe, image, control_net_name, control_net=None, strength=1, start_percent=0, end_percent=1, scale_soft_weights=1, union_type=None):
        if scale_soft_weights < 1:
            if "ScaledSoftControlNetWeights" in ALL_NODE_CLASS_MAPPINGS:
                soft_weight_cls = ALL_NODE_CLASS_MAPPINGS['ScaledSoftControlNetWeights']
                (weights, timestep_keyframe) = soft_weight_cls().load_weights(scale_soft_weights, False)
                cn_adv_cls = ALL_NODE_CLASS_MAPPINGS['ACN_ControlNet++LoaderSingle']
                if union_type == 'auto':
                    union_type = 'none'
                elif union_type == 'canny/lineart/anime_lineart/mlsd':
                    union_type = 'canny/lineart/mlsd'
                elif union_type == 'repaint':
                    union_type = 'inpaint/outpaint'
                control_net, = cn_adv_cls().load_controlnet_plusplus(control_net_name, union_type)
                apply_adv_cls = ALL_NODE_CLASS_MAPPINGS['ACN_AdvancedControlNetApply']
                positive, negative, _ = apply_adv_cls().apply_controlnet(pipe["positive"], pipe["negative"], control_net, image, strength, start_percent, end_percent, timestep_kf=timestep_keyframe,)
            else:
                raise Exception(
                    f"[Advanced-ControlNet Not Found] you need to install 'COMFYUI-Advanced-ControlNet'")
        else:
            positive, negative = easyControlnet().apply(control_net_name, image, pipe["positive"], pipe["negative"],
                                                        strength, start_percent, end_percent, control_net, scale_soft_weights, union_type=union_type, mask=None, easyCache=easyCache, model=pipe['model'])

        new_pipe = {
            "model": pipe['model'],
            "positive": positive,
            "negative": negative,
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": pipe["samples"],
            "images": pipe["images"],
            "seed": 0,

            "loader_settings": pipe["loader_settings"]
        }

        del pipe

        return (new_pipe, positive, negative)

# LLLiteLoader
from ..libs.lllite import load_control_net_lllite_patch
class LLLiteLoader:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "lllite" in file]

        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_file_list(folder_paths.get_filename_list("controlnet")),),
                "cond_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "end_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lllite"
    CATEGORY = "EasyUse/Loaders"

    def load_lllite(self, model, model_name, cond_image, strength, steps, start_percent, end_percent):
        # cond_image is b,h,w,3, 0-1

        model_path = os.path.join(folder_paths.get_full_path("controlnet", model_name))

        model_lllite = model.clone()
        patch = load_control_net_lllite_patch(model_path, cond_image, strength, steps, start_percent, end_percent)
        if patch is not None:
            model_lllite.set_model_attn1_patch(patch)
            model_lllite.set_model_attn2_patch(patch)

        return (model_lllite,)


NODE_CLASS_MAPPINGS = {
    "easy fullLoader": fullLoader,
    "easy a1111Loader": a1111Loader,
    "easy comfyLoader": comfyLoader,
    "easy svdLoader": svdLoader,
    "easy sv3dLoader": sv3dLoader,
    "easy zero123Loader": zero123Loader,
    "easy cascadeLoader": cascadeLoader,
    "easy kolorsLoader": kolorsLoader,
    "easy fluxLoader": fluxLoader,
    "easy hunyuanDiTLoader": hunyuanDiTLoader,
    "easy pixArtLoader": pixArtLoader,
    "easy mochiLoader": mochiLoader,
    "easy loraSwitcher": loraSwitcher,
    "easy loraStack": loraStack,
    "easy controlnetStack": controlnetStack,
    "easy controlnetLoader": controlnetSimple,
    "easy controlnetLoaderADV": controlnetAdvanced,
    "easy controlnetLoader++": controlnetPlusPlus,
    "easy LLLiteLoader": LLLiteLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy fullLoader": "EasyLoader (Full)",
    "easy a1111Loader": "EasyLoader (A1111)",
    "easy comfyLoader": "EasyLoader (Comfy)",
    "easy svdLoader": "EasyLoader (SVD)",
    "easy sv3dLoader": "EasyLoader (SV3D)",
    "easy zero123Loader": "EasyLoader (Zero123)",
    "easy cascadeLoader": "EasyCascadeLoader",
    "easy kolorsLoader": "EasyLoader (Kolors)",
    "easy fluxLoader": "EasyLoader (Flux)",
    "easy hunyuanDiTLoader": "EasyLoader (HunyuanDiT)",
    "easy pixArtLoader": "EasyLoader (PixArt)",
    "easy mochiLoader": "EasyLoader (Mochi)",
    "easy loraSwitcher": "EasyLoraSwitcher",
    "easy loraStack": "EasyLoraStack",
    "easy controlnetStack": "EasyControlnetStack",
    "easy controlnetLoader": "EasyControlnet",
    "easy controlnetLoaderADV": "EasyControlnet (Advanced)",
    "easy controlnetLoader++": "EasyControlnet++",
    "easy LLLiteLoader": "EasyLLLite"
}
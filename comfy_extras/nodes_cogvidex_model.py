import os
from typing import List, Optional, Union
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline
from datetime import datetime, timedelta

from diffusers.image_processor import VaeImageProcessor
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.models.t5.tokenization_t5 import T5Tokenizer

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 

class CogVideoModelLoader:
    @classmethod
    def INPUT_TYPES(s):  
        cog_videox_base = folder_paths.get_folder_paths("cog_videox")
        model_paths = []
        for path_ in cog_videox_base:
            if path_ is None:
                continue 
            for x in os.listdir(path_):
                model_paths.append(os.path.join(path_, x))
 
        return {
            "required": { 
                "base_path": (model_paths,),
            },
        }
 
    RETURN_TYPES = ("COGVIDEOPIPE",)
    RETURN_NAMES = ("cogvideo_pipe",)
    FUNCTION = "cogvideo_loader"
    CATEGORY = "loaders"

    def cogvideo_loader(self, base_path): 
        precision = "bf16"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
  
  
        transformer = CogVideoXTransformer3DModel.from_pretrained(base_path, subfolder="transformer")
        transformer.to(dtype) 
        vae = AutoencoderKLCogVideoX.from_pretrained(base_path, subfolder="vae")
        vae.to(dtype)
        text_encoder = T5EncoderModel.from_pretrained(base_path, subfolder="text_encoder")
        t5_tokenizer = T5Tokenizer.from_pretrained(base_path, subfolder="tokenizer") 
        scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        pipe = CogVideoXPipeline(t5_tokenizer, text_encoder, vae, transformer, scheduler)

        pipeline = {
            "pipe": pipe,
            "dtype": dtype, 
        }
        # TODO execution.py:get_output_data output parse to dict error, so will't return dict here but return tuple 
        return (pipeline,)
  
     

class CogVideoPipeExtra:
    @classmethod
    def INPUT_TYPES(s):  
        return {
            "required": { 
                "pipeline": ("COGVIDEOPIPE",),
            },
        }
 
    RETURN_TYPES = ("T5_TOKENIZER", "TEXT_ENCODER", "VAE3D","3DTRANSFORMER", "SCHEDULER", "DTYPE")
    RETURN_NAMES = ("t5_tokenizer", "text_encoder", "vae3d", "transformer","scheduler" "dtype")
    FUNCTION = "extra"
    CATEGORY = "loaders"
 
    def extra(self, pipeline): 
        t5_tokenizer = pipeline["pipe"].tokenizer
        vae3d = pipeline["pipe"].vae
        text_encoder = pipeline["pipe"].text_encoder
        transformer = pipeline["pipe"].transformer
        scheduler = pipeline["pipe"].scheduler
        dtype = pipeline["dtype"]

        return (t5_tokenizer, text_encoder, vae3d, transformer, scheduler, dtype)
     


class CogVideoEncodePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "t5_tokenizer": ("T5_TOKENIZER",),
            "text_encoder": ("TEXT_ENCODER",),
            "dtype": ("DTYPE",),
            "prompt": ("STRING", {"default": "", "multiline": True} ),
            "negative_prompt": ("STRING", {"default": "", "multiline": True} ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "process"
    CATEGORY = "latent"


    def _get_t5_prompt_embeds(
        self,
        t5_tokenizer,
        text_encoder,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ): 
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = t5_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = t5_tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def _encode_prompt(
        self,
        t5_tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument. 
        """ 

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                t5_tokenizer,
                text_encoder,
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                t5_tokenizer,
                text_encoder,
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds


    def process(self, t5_tokenizer, text_encoder, dtype, prompt, negative_prompt):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device() 

        text_encoder.to(device)  
        
        positive, negative = self._encode_prompt(
            t5_tokenizer,
            text_encoder,
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=device,
            dtype=dtype,
        )
        text_encoder.to(offload_device) 

        return (positive, negative)
    
 


class CogVideoImageEncodeSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae3d": ("VAE3D",),
            "image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "latent"

    def encode(self, vae3d, image):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        vae3d.to(device)
  
        input_image = image.clone() * 2.0 - 1.0
        input_image = input_image.to(vae3d.dtype).to(device)
        input_image = input_image.unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
        B, C, T, H, W = input_image.shape
        chunk_size = 16
        latents_list = []
        # Loop through the temporal dimension in chunks of 16
        for i in range(0, T, chunk_size):
            # Get the chunk of 16 frames (or remaining frames if less than 16 are left)
            end_index = min(i + chunk_size, T)
            image_chunk = input_image[:, :, i:end_index, :, :]  # Shape: [B, C, chunk_size, H, W]

            # Encode the chunk of images
            latents = vae3d.encode(image_chunk)

            sample_mode = "sample"
            if hasattr(latents, "latent_dist") and sample_mode == "sample":
                latents = latents.latent_dist.sample(generator)
            elif hasattr(latents, "latent_dist") and sample_mode == "argmax":
                latents = latents.latent_dist.mode()
            elif hasattr(latents, "latents"):
                latents = latents.latents

            latents = vae3d.config.scaling_factor * latents
            latents = latents.permute(0, 2, 1, 3, 4)  # B, T_chunk, C, H, W
            latents_list.append(latents)
        vae3d._clear_fake_context_parallel_cache()

        # Concatenate all the chunks along the temporal dimension
        final_latents = torch.cat(latents_list, dim=1)
        print("final latents: ", final_latents.shape)
        
        vae3d.to(offload_device)
        
        return ({"samples": final_latents}, )


class CogVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("COGVIDEOPIPE",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 8}),
                "width": ("INT", {"default": 720, "min": 128, "max": 2048, "step": 8}),
                "num_frames": ("INT", {"default": 49, "min": 16, "max": 1024, "step": 1}),
                "steps": ("INT", {"default": 50, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["DDIM", "DPM"], {"tooltip": "5B likes DPM, but it doesn't support temporal tiling"}),
                "t_tile_length": ("INT", {"default": 16, "min": 2, "max": 128, "step": 1, "tooltip": "Length of temporal tiling, use same alue as num_frames to disable, disabled automatically for DPM"}),
                "t_tile_overlap": ("INT", {"default": 8, "min": 2, "max": 128, "step": 1, "tooltip": "Overlap of temporal tiling"}),
            },
            "optional": {
                "samples": ("LATENT", ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("COGVIDEOPIPE", "LATENT",)
    RETURN_NAMES = ("cogvideo_pipe", "samples",)
    FUNCTION = "process"
    CATEGORY = "CogVideoWrapper"

    def process(self, pipeline, positive, negative, steps, cfg, seed, height, width, num_frames, scheduler, t_tile_length, t_tile_overlap, samples=None, denoise_strength=1.0):
        mm.soft_empty_cache()

        assert t_tile_length > t_tile_overlap, "t_tile_length must be greater than t_tile_overlap"
        assert t_tile_length <= num_frames, "t_tile_length must be equal or less than num_frames"
        t_tile_length = t_tile_length // 4
        t_tile_overlap = t_tile_overlap // 4

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"]
        base_path = pipeline["base_path"]
        
        pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)

        if scheduler == "DDIM":
            pipe.scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        elif scheduler == "DPM":
            pipe.scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        with torch.autocast(mm.get_autocast_device(device)):
            latents = pipeline["pipe"](
                num_inference_steps=steps,
                height = height,
                width = width,
                num_frames = num_frames,
                t_tile_length = t_tile_length,
                t_tile_overlap = t_tile_overlap,
                guidance_scale=cfg,
                latents=samples["samples"] if samples is not None else None,
                denoise_strength=denoise_strength,
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                generator=generator,
                device=device
            )
        pipe.transformer.to(offload_device)
        mm.soft_empty_cache()
        print(latents.shape)

        return (pipeline, {"samples": latents})
    
class CogVideoSamplerDecodeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae3d": ("VAE3D",),
            "samples": ("LATENT", ),
            "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            "tile_sample_min_height": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
            "tile_sample_min_width": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
            "tile_overlap_factor_height": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
            "tile_overlap_factor_width": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("BARCHIMAGE",)
    RETURN_NAMES = ("b_images",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, vae3d, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width):
        """
            vae3d: VAE3D
            samples: dict of samples,  torch.Tensor  # B, T_chunk, C, H, W
        """
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        latents = samples["samples"] 
        vae3d.to(device)
        if enable_vae_tiling:
            vae3d.enable_tiling(
                tile_sample_min_height=tile_sample_min_height,
                tile_sample_min_width=tile_sample_min_width,
                tile_overlap_factor_height=tile_overlap_factor_height,
                tile_overlap_factor_width=tile_overlap_factor_width,
            )
        latents = latents.to(vae3d.dtype)
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / vae3d.config.scaling_factor * latents

        frames = vae3d.decode(latents).sample # B, C, F, H, W
        vae3d.to(offload_device)
        mm.soft_empty_cache()
         
        batch_size = frames.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            pt_image = frames[batch_idx].permute(1, 0, 2, 3)  # (to [f, c, w, h])

            outputs.append(torch.stack(
                [VaeImageProcessor.denormalize(pt_image[i]) for i in range(pt_image.shape[0])]
            ))
            

        return (torch.stack(outputs),)



class CogVideoProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "b_images": ("BARCHIMAGE", ), 
                "output_type": (["pt", "np", "plt"], {"default": "pt"}), 
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
     

    RETURN_TYPES = ("IMAGE","INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("image", "width", "height", "count", "batch_index", "batch_size",)
    FUNCTION = "process"
    CATEGORY = "video"

    DESCRIPTION = """Converts CogVideo tensor, pt image to np image or plt image,
and returns the width, height, count, batch_index, and batch_size.
"""
    def process(self, b_images, output_type, batch_index):
        batch_size = b_images.shape[0]
        pt_image = b_images[batch_index] # ([f, c, w, h])
        width = pt_image.shape[2]
        height = pt_image.shape[3]
        count = pt_image.shape[0]

        if output_type == "plt":
            image_np = VaeImageProcessor.pt_to_numpy(pt_image) # (to [f,w, h, c])
            image = VaeImageProcessor.numpy_to_pil(image_np)
        elif output_type == "np":
            image = VaeImageProcessor.pt_to_numpy(pt_image).cpu().float() # (to [f, w, h, c])
        elif output_type == "pt":
            image =  pt_image.permute( 0, 2, 3, 1).cpu().float() # (to [f, w, h, c])
  
        return {
        # TODO javascript app.registerExtension
        #    "ui": {
        #         "text": [f"{batch_size}x{batch_index}x{count}x{width}x{height}"]
        #     }, 
            "result": (image, width, height, count, batch_index, batch_size) 
        }
    



NODE_CLASS_MAPPINGS = {
    "CogVideoModelLoader": CogVideoModelLoader,
    "CogVideoPipeExtra": CogVideoPipeExtra,
    "CogVideoEncodePrompt": CogVideoEncodePrompt,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoImageEncodeSampler": CogVideoImageEncodeSampler,
    "CogVideoSamplerDecodeImages": CogVideoSamplerDecodeImages, 
    "CogVideoProcessor": CogVideoProcessor, 
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoModelLoader": "CogVideo ModelLoader",
    "CogVideoPipeExtra": "CogVideo PipeExtra",
    "CogVideoEncodePrompt": "CogVideo EncodePrompt",
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode", 
    "CogVideoImageEncodeSampler": "CogVideo ImageEncodeSampler",
    "CogVideoSamplerDecodeImages": "CogVideo SamplerDecodeImages",
    "CogVideoVideoProcessor": "CogVideo VideoProcessor",
}
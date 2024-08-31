import os
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps, get_resize_crop_region_for_grid, get_3d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback

from diffusers.image_processor import VaeImageProcessor
from transformers.models.t5.modeling_t5 import T5EncoderModel
from transformers.models.t5.tokenization_t5 import T5Tokenizer
import math
from tqdm.auto import tqdm
import inspect
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_timesteps(
    cogvideo_pipeline: CogVideoXPipeline,
    num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = cogvideo_pipeline.scheduler.timesteps[t_start * cogvideo_pipeline.scheduler.order :]
    if hasattr(cogvideo_pipeline.scheduler, "set_begin_index"):
        cogvideo_pipeline.scheduler.set_begin_index(t_start * cogvideo_pipeline.scheduler.order)

    return timesteps.to(device), num_inference_steps - t_start


def _prepare_rotary_positional_embeddings(
    cogvideo_pipeline: CogVideoXPipeline,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (cogvideo_pipeline.vae_scale_factor_spatial * cogvideo_pipeline.transformer.config.patch_size)
    grid_width = width // (cogvideo_pipeline.vae_scale_factor_spatial * cogvideo_pipeline.transformer.config.patch_size)
    base_size_width = 720 // (cogvideo_pipeline.vae_scale_factor_spatial * cogvideo_pipeline.transformer.config.patch_size)
    base_size_height = 480 // (cogvideo_pipeline.vae_scale_factor_spatial * cogvideo_pipeline.transformer.config.patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=cogvideo_pipeline.transformer.config.attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
        use_real=True,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(
    cogvideo_pipeline: CogVideoXPipeline,
    generator, eta
):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(cogvideo_pipeline.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(cogvideo_pipeline.scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs



def prepare_latents(
    cogvideo_pipeline: CogVideoXPipeline,
    batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
):
    shape = (
            batch_size,
            (num_frames - 1) // cogvideo_pipeline.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // cogvideo_pipeline.vae_scale_factor_spatial,
            width // cogvideo_pipeline.vae_scale_factor_spatial,
        )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * cogvideo_pipeline.scheduler.init_noise_sigma
    return latents
 
@torch.no_grad()
def cogvideox_sampler(
    cogvideo_pipeline: CogVideoXPipeline,
    height: int = 480,
    width: int = 720,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    timesteps: Optional[List[int]] = None,
    guidance_scale: float = 6, 
    use_dynamic_cfg: bool = False,
    denoise_strength: float = 1.0,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None, 
    device = torch.device("cuda"),
    noise_rectification_period: Optional[list] = None,
    noise_rectification_weight_start_omega = 1.0,
    noise_rectification_weight_end_omega = 0.5,
):
    """
    Function invoked when calling the pipeline for generation.

    Args:
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_frames (`int`, defaults to `49`):
            Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
            contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
            num_seconds is 6 and fps is 4. However, since videos can be saved at any fps, the only condition that
            needs to be satisfied is that of divisibility mentioned above.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        denoise_strength (`float`, *optional*, defaults to 1.0): This parameter ranges from 0.1 to 1.0 and
            controls the fidelity and diversity of the generated images. denoise_strength is typically used 
            in models like diffusion models or similar generative models. It influences the balance between preserving image
            details and achieving diversity during the generation process. A higher denoise strength means the image will
            be closer to real images from the training data, while a lower denoise strength may retain more noise,
            leading to more abstract or creative results
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of videos to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        noise_rectification_period (`List[int]`, *optional*): https://arxiv.org/pdf/2403.02827
        
    """
    if num_frames > 49:
        raise ValueError(
            "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
        )

    height = height or cogvideo_pipeline.transformer.config.sample_size * cogvideo_pipeline.vae_scale_factor_spatial
    width = width or cogvideo_pipeline.transformer.config.sample_size * cogvideo_pipeline.vae_scale_factor_spatial
    num_videos_per_prompt = 1

    cogvideo_pipeline._guidance_scale = guidance_scale
    cogvideo_pipeline._interrupt = False

    # 2. Default call parameters
    
    batch_size = prompt_embeds.shape[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    prompt_embeds = prompt_embeds.to(cogvideo_pipeline.transformer.dtype)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(cogvideo_pipeline.scheduler, num_inference_steps, device, timesteps)
    cogvideo_pipeline._num_timesteps = len(timesteps)

    # 5. Prepare latents.
    current_frames = latents.shape[1] if latents is not None else (num_frames - 1) // cogvideo_pipeline.vae_scale_factor_temporal + 1
    latent_channels = cogvideo_pipeline.transformer.config.in_channels
    latents = prepare_latents(
        cogvideo_pipeline,
        batch_size * num_videos_per_prompt,
        latent_channels,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
 

    timesteps, num_inference_steps = get_timesteps(cogvideo_pipeline, num_inference_steps, denoise_strength, device)
    latent_timestep = timesteps[:1]
    
    # images to video latents
    # Check if the number of frames needed matches the current frames
    frames_shape = (
        batch_size,
        (num_frames - 1) // cogvideo_pipeline.vae_scale_factor_temporal + 1,
        latent_channels,
        height // cogvideo_pipeline.vae_scale_factor_spatial,
        width // cogvideo_pipeline.vae_scale_factor_spatial,
    )
 
    frames_needed = frames_shape[1]
    
    # https://arxiv.org/pdf/2403.02827 Our method first adds noise to the input image and keep the added noise for latter rectification.
    noise = latents.clone()
    init_latents = None
    video_length = (num_frames - 1) // cogvideo_pipeline.vae_scale_factor_temporal + 1
    if frames_needed > current_frames:
        # images lanets frame is index 0 ，get the noise latents 
        init_latents = latents[:, 0, :, :, :, :]
    elif frames_needed < current_frames:
        latents = latents[:, :frames_needed, :, :, :]
    
    if init_latents is not None:
        latents = cogvideo_pipeline.scheduler.add_noise(init_latents, noise, latent_timestep)
    latents = latents.to(cogvideo_pipeline.transformer.dtype)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(cogvideo_pipeline, generator, eta)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        _prepare_rotary_positional_embeddings(cogvideo_pipeline, height, width, latents.size(1), device)
        if cogvideo_pipeline.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * cogvideo_pipeline.scheduler.order, 0)

    comfy_pbar = ProgressBar(num_inference_steps)
    with tqdm(total=num_inference_steps) as progress_bar:
        # for DPM-solver++
        old_pred_original_sample = None
        for i, t in enumerate(timesteps):
            if cogvideo_pipeline.interrupt:
                continue

            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = cogvideo_pipeline.scheduler.scale_model_input(latent_model_input, t)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # predict noise model_output
            noise_pred = cogvideo_pipeline.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.float()

            # perform guidance
            if use_dynamic_cfg:
                cogvideo_pipeline._guidance_scale = 1 + guidance_scale * (
                    (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                )
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cogvideo_pipeline.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # [The core code of our method.]
            # https://arxiv.org/pdf/2403.02827 our method rectifies the predicted noise with the GT noise to realize image-to-video.
            if noise_rectification_period is not None:
                assert len(noise_rectification_period) == 2
                noise_rectification_weight = torch.cat([torch.linspace(noise_rectification_weight_start_omega, noise_rectification_weight_end_omega, video_length//2), 
                                                        torch.linspace(noise_rectification_weight_end_omega, noise_rectification_weight_end_omega, video_length//2)])
                noise_rectification_weight = noise_rectification_weight.view(1, video_length, 1, 1, 1)
                noise_rectification_weight = noise_rectification_weight.to(latent_model_input.dtype).to(latent_model_input.device)

                if i >= len(timesteps) * noise_rectification_period[0] and i < len(timesteps) * noise_rectification_period[1]:
                    delta_frames = noise - noise_pred
                    delta_noise_adjust = noise_rectification_weight * (delta_frames[:,:,[0],:,:].repeat((1, video_length, 1, 1, 1))) + \
                                        (1 - noise_rectification_weight) * delta_frames
                    noise_pred = noise_pred + delta_noise_adjust

            # compute the previous noisy sample x_t -> x_t-1
            if not isinstance(cogvideo_pipeline.scheduler, CogVideoXDPMScheduler):
                latents = cogvideo_pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            else:
                latents, old_pred_original_sample = cogvideo_pipeline.scheduler.step(
                    noise_pred,
                    old_pred_original_sample,
                    t,
                    timesteps[i - 1] if i > 0 else None,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )
            latents = latents.to(prompt_embeds.dtype) 

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % cogvideo_pipeline.scheduler.order == 0):
                progress_bar.update()
                comfy_pbar.update(1)

    # Offload all models
    cogvideo_pipeline.maybe_free_model_hooks()

    return latents



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
    DESCRIPTION = """CogVideoX model loader"""

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


class CogVideoSchedulerLoader:
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
                "scheduler": (["DDIM", "DPM"], {"tooltip": "5B likes DPM, but it doesn't support temporal tiling"}),
            },
        }
 
    RETURN_TYPES = ("COGVIDEOSCHEDULER",)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "cogvideo_scheduler_loader"
    CATEGORY = "loaders"
    DESCRIPTION = """CogVideoX scheduler loader"""


    def cogvideo_scheduler_loader(self, base_path, scheduler):  
        pipe_scheduler = None
        if scheduler == "DDIM":
            pipe_scheduler = CogVideoXDDIMScheduler.from_pretrained(base_path, subfolder="scheduler")
        elif scheduler == "DPM":
            pipe_scheduler = CogVideoXDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
        else:
            raise ValueError(f"Invalid scheduler {scheduler}")
        
        return (pipe_scheduler,)
  

class CogVideoPipeExtra:
    @classmethod
    def INPUT_TYPES(s):  
        return {
            "required": { 
                "pipeline": ("COGVIDEOPIPE",),
            },
        }
 
    RETURN_TYPES = ("T5_TOKENIZER", "TEXT_ENCODER", "VAE3D","3DTRANSFORMER", "COGVIDEOSCHEDULER", "DTYPE",)
    RETURN_NAMES = ("t5_tokenizer", "text_encoder", "vae3d", "transformer","scheduler" "dtype",)
    FUNCTION = "extra"
    CATEGORY = "loaders"
    DESCRIPTION = """CogVideoX pipeline extra loader, return t5_tokenizer, text_encoder, vae3d, transformer, scheduler, dtype"""
 
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
    DESCRIPTION = """VideoEncode prompt, eg: Also use clip to convert CONDITIONING"""


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
        return {
            "required": {
                "vae3d": ("VAE3D",),
                "image": ("IMAGE", ),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "tile_sample_min_height": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
                "tile_sample_min_width": ("INT", {"default": 96, "min": 16, "max": 2048, "step": 8}),
                "tile_overlap_factor_height": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
                "tile_overlap_factor_width": ("FLOAT", {"default": 0.083, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "latent"
    DESCRIPTION = """3DVAE image Encode to latent"""

    def encode(self, vae3d, image, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, tile_overlap_factor_width):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        generator = torch.Generator(device=device).manual_seed(0)
        vae3d.to(device)
        if enable_vae_tiling:
            vae3d.enable_tiling(
                tile_sample_min_height=tile_sample_min_height,
                tile_sample_min_width=tile_sample_min_width,
                tile_overlap_factor_height=tile_overlap_factor_height,
                tile_overlap_factor_width=tile_overlap_factor_width,
            )
        else:
            vae3d.disable_tiling()
    
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
                "steps": ("INT", {"default": 50, "min": 1}),
                "num_frames": ("INT", {"default": 49, "min": 8, "max": 49}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "samples": ("LATENT", ),
                "scheduler": ("COGVIDEOSCHEDULER", ), 
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "sampler"
    DESCRIPTION = """CogVideoX sampler"""

    def process(self, pipeline, positive, negative, steps, num_frames, cfg, denoise_strength, seed, height, width, samples=None, scheduler=None):
        mm.soft_empty_cache()
 
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        pipe = pipeline["pipe"]
        dtype = pipeline["dtype"] 
        pipe.transformer.to(device)
        generator = torch.Generator(device=device).manual_seed(seed)
        if scheduler is not None:
            pipe.scheduler = scheduler
 
        with torch.autocast(mm.get_autocast_device(device)):
            latents = cogvideox_sampler(
                pipe,
                num_inference_steps=steps,
                height = height,
                width = width,
                num_frames=(num_frames-1), # frames - 1 step zero frame start
                guidance_scale=cfg,
                denoise_strength=denoise_strength,
                latents=samples["samples"] if samples is not None else None,
                prompt_embeds=positive.to(dtype).to(device),
                negative_prompt_embeds=negative.to(dtype).to(device),
                generator=generator,
                device=device
            )
        pipe.transformer.to(offload_device)
        mm.soft_empty_cache() 

        return ({"samples": latents}, )
    
class CogVideoSamplerDecodeImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
    DESCRIPTION = """CogVideoX sampler 3DVAE decode images"""

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
        else:
            vae3d.disable_tiling()

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
    CATEGORY = "latent"

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
    "CogVideoSchedulerLoader": CogVideoSchedulerLoader,
    "CogVideoPipeExtra": CogVideoPipeExtra,
    "CogVideoEncodePrompt": CogVideoEncodePrompt,
    "CogVideoSampler": CogVideoSampler,
    "CogVideoImageEncodeSampler": CogVideoImageEncodeSampler,
    "CogVideoSamplerDecodeImages": CogVideoSamplerDecodeImages, 
    "CogVideoProcessor": CogVideoProcessor, 
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoModelLoader": "CogVideo ModelLoader",
    "CogVideoSchedulerLoader": "CogVideo SchedulerLoader",
    "CogVideoPipeExtra": "CogVideo PipeExtra",
    "CogVideoEncodePrompt": "CogVideo EncodePrompt",
    "CogVideoSampler": "CogVideo Sampler",
    "CogVideoDecode": "CogVideo Decode", 
    "CogVideoImageEncodeSampler": "CogVideo ImageEncodeSampler",
    "CogVideoSamplerDecodeImages": "CogVideo SamplerDecodeImages",
    "CogVideoVideoProcessor": "CogVideo VideoProcessor",
}
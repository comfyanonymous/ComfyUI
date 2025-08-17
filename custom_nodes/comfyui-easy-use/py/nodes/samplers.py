import sys, re, time
import torch
import comfy.utils, comfy.sample, comfy.samplers, comfy.controlnet, comfy.model_base, comfy.model_management, comfy.sampler_helpers, comfy.supported_models
from comfy.model_patcher import ModelPatcher
from comfy_extras.nodes_mask import GrowMask
import comfy_extras.nodes_custom_sampler as custom_samplers
from tqdm import trange

from server import PromptServer
from nodes import RepeatLatentBatch, NODE_CLASS_MAPPINGS as ALL_NODE_CLASS_MAPPINGS, VAEEncodeForInpaint, InpaintModelConditioning
from ..modules.layer_diffuse import LayerDiffuse
from ..config import *

from ..libs.log import log_node_warn
from ..libs.utils import  easySave, get_local_filepath, get_sd_version
from ..libs.sampler import alignYourStepsScheduler, gitsScheduler
from ..libs.xyplot import easyXYPlot

from .. import easyCache, sampler

class samplerFull:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS+NEW_SCHEDULERS,),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save", "None"],),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "xyPlot": ("XYPLOT",),
                    "image": ("IMAGE",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe",  "image", "model", "positive", "negative", "latent", "vae", "clip", "seed",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def ip2p(self, positive, negative, vae=None, pixels=None, latent=None):
        if latent is not None:
            concat_latent = latent
        else:
            x = (pixels.shape[1] // 8) * 8
            y = (pixels.shape[2] // 8) * 8

            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]

            concat_latent = vae.encode(pixels)

        out_latent = {}
        out_latent["samples"] = torch.zeros_like(concat_latent)

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], out_latent)

    def get_inversed_euler_sampler(self):
        @torch.no_grad()
        def sample_inversed_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0.,s_tmax=float('inf'), s_noise=1.):
            """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            for i in trange(1, len(sigmas), disable=disable):
                sigma_in = sigmas[i - 1]

                if i == 1:
                    sigma_t = sigmas[i]
                else:
                    sigma_t = sigma_in

                denoised = model(x, sigma_t * s_in, **extra_args)

                if i == 1:
                    d = (x - denoised) / (2 * sigmas[i])
                else:
                    d = (x - denoised) / sigmas[i - 1]

                dt = sigmas[i] - sigmas[i - 1]
                x = x + d * dt
                if callback is not None:
                    callback(
                        {'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
            return x / sigmas[-1]

        ksampler = comfy.samplers.KSAMPLER(sample_inversed_euler)
        return (ksampler,)

    def get_custom_cls(self, sampler_name):
        try:
            cls = custom_samplers.__dict__[sampler_name]
            return cls()
        except:
            raise Exception(f"Custom sampler {sampler_name} not found, Please updated your ComfyUI")

    def add_model_patch_option(self, model):
        if 'transformer_options' not in model.model_options:
            model.model_options['transformer_options'] = {}
        to = model.model_options['transformer_options']
        if "model_patch" not in to:
            to["model_patch"] = {}
        return to

    def get_sampler_custom(self, model, positive, negative, loader_settings):
        _guider = None
        middle = loader_settings['middle'] if "middle" in loader_settings else negative
        steps = loader_settings['steps'] if "steps" in loader_settings else 20
        cfg = loader_settings['cfg'] if "cfg" in loader_settings else 8.0
        cfg_negative = loader_settings['cfg_negative'] if "cfg_negative" in loader_settings else 8.0
        sampler_name = loader_settings['sampler_name'] if "sampler_name" in loader_settings else "euler"
        scheduler = loader_settings['scheduler'] if "scheduler" in loader_settings else "normal"
        guider = loader_settings['custom']['guider'] if "guider" in loader_settings['custom'] else "CFG"
        beta_d = loader_settings['custom']['beta_d'] if "beta_d" in loader_settings['custom'] else 0.1
        beta_min = loader_settings['custom']['beta_min'] if "beta_min" in loader_settings['custom'] else 0.1
        eps_s = loader_settings['custom']['eps_s'] if "eps_s" in loader_settings['custom'] else 0.1
        sigma_max = loader_settings['custom']['sigma_max'] if "sigma_max" in loader_settings['custom'] else 14.61
        sigma_min = loader_settings['custom']['sigma_min'] if "sigma_min" in loader_settings['custom'] else 0.03
        rho = loader_settings['custom']['rho'] if "rho" in loader_settings['custom'] else 7.0
        coeff = loader_settings['custom']['coeff'] if "coeff" in loader_settings['custom'] else 1.2
        flip_sigmas = loader_settings['custom']['flip_sigmas'] if "flip_sigmas" in loader_settings['custom'] else False
        denoise = loader_settings['denoise'] if "denoise" in loader_settings else 1.0
        optional_sigmas = loader_settings['optional_sigmas'] if "optional_sigmas" in loader_settings else None
        optional_sampler = loader_settings['optional_sampler'] if "optional_sampler" in loader_settings else None

        # sigmas
        if optional_sigmas is not None:
            sigmas = optional_sigmas
        else:
            if scheduler == 'vp':
                sigmas, = self.get_custom_cls('VPScheduler').get_sigmas(steps, beta_d, beta_min, eps_s)
            elif scheduler == 'karrasADV':
                sigmas, = self.get_custom_cls('KarrasScheduler').get_sigmas(steps, sigma_max, sigma_min, rho)
            elif scheduler == 'exponentialADV':
                sigmas, = self.get_custom_cls('ExponentialScheduler').get_sigmas(steps, sigma_max, sigma_min)
            elif scheduler == 'polyExponential':
                sigmas, = self.get_custom_cls('PolyexponentialScheduler').get_sigmas(steps, sigma_max, sigma_min, rho)
            elif scheduler == 'sdturbo':
                sigmas, = self.get_custom_cls('SDTurboScheduler').get_sigmas(model, steps, denoise)
            elif scheduler == 'alignYourSteps':
                model_type = get_sd_version(model)
                if model_type == 'unknown':
                    model_type = 'sdxl'
                sigmas, = alignYourStepsScheduler().get_sigmas(model_type.upper(), steps, denoise)
            elif scheduler == 'gits':
                sigmas, = gitsScheduler().get_sigmas(coeff, steps, denoise)
            else:
                sigmas, = self.get_custom_cls('BasicScheduler').get_sigmas(model, scheduler, steps, denoise)

        # filp_sigmas
        if flip_sigmas:
            sigmas, = self.get_custom_cls('FlipSigmas').get_sigmas(sigmas)

        #######################################################################################
        # brushnet
        to = None
        transformer_options = model.model_options['transformer_options'] if "transformer_options" in model.model_options else {}
        if 'model_patch' in transformer_options and 'brushnet' in transformer_options['model_patch']:
            to = self.add_model_patch_option(model)
            mp = to['model_patch']
            if isinstance(model.model.model_config, comfy.supported_models.SD15):
                mp['SDXL'] = False
            elif isinstance(model.model.model_config, comfy.supported_models.SDXL):
                mp['SDXL'] = True
            else:
                print('Base model type: ', type(model.model.model_config))
                raise Exception("Unsupported model type: ", type(model.model.model_config))

            mp['all_sigmas'] = sigmas
            mp['unet'] = model.model.diffusion_model
            mp['step'] = 0
            mp['total_steps'] = 1
        #######################################################################################
        # guider
        if cfg > 0 and get_sd_version(model) == 'flux':
            c = []
            for t in positive:
                n = [t[0], t[1]]
                n[1]['guidance'] = cfg
                c.append(n)
            positive = c

        if guider in ['CFG', 'IP2P+CFG']:
            _guider, = self.get_custom_cls('CFGGuider').get_guider(model, positive, negative, cfg)
        elif guider in ['DualCFG', 'IP2P+DualCFG']:
            _guider, = self.get_custom_cls('DualCFGGuider').get_guider(model, positive, middle,
                                                                       negative, cfg, cfg_negative)
        else:
            _guider, = self.get_custom_cls('BasicGuider').get_guider(model, positive)

        # sampler
        if optional_sampler:
            _sampler = optional_sampler
        else:
            if sampler_name == 'inversed_euler':
                _sampler, = self.get_inversed_euler_sampler()
            else:
                _sampler, = self.get_custom_cls('KSamplerSelect').get_sampler(sampler_name)


        return (_guider, _sampler, sigmas)

    def run(self, pipe, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed=None, model=None, positive=None, negative=None, latent=None, vae=None, clip=None, xyPlot=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False, downscale_options=None, image=None):

        samp_model = model if model is not None else pipe["model"]
        samp_positive = positive if positive is not None else pipe["positive"]
        samp_negative = negative if negative is not None else pipe["negative"]
        samp_samples = latent if latent is not None else pipe["samples"]
        samp_vae = vae if vae is not None else pipe["vae"]
        samp_clip = clip if clip is not None else pipe["clip"]

        samp_seed = seed if seed is not None else pipe['seed']

        samp_custom = pipe["loader_settings"] if "custom" in pipe["loader_settings"] else None

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        add_noise = pipe['loader_settings']['add_noise'] if 'add_noise' in pipe['loader_settings'] else 'enabled'
        force_full_denoise = pipe['loader_settings']['force_full_denoise'] if 'force_full_denoise' in pipe['loader_settings'] else True
        noise_device = 'GPU' if ('a1111_prompt_style' in pipe['loader_settings'] and pipe['loader_settings']['a1111_prompt_style']) or add_noise == 'enable (GPU=A1111)' else 'CPU'

        if image is not None and latent is None:
            samp_samples = {"samples": samp_vae.encode(image[:, :, :, :3])}

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        def downscale_model_unet(samp_model):
            # 获取Unet参数
            if "PatchModelAddDownscale" in ALL_NODE_CLASS_MAPPINGS:
                cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                # 自动收缩Unet
                if downscale_options['downscale_factor'] is None:
                    unet_config = samp_model.model.model_config.unet_config
                    if unet_config is not None and "samples" in samp_samples:
                        height = samp_samples['samples'].shape[2] * 8
                        width = samp_samples['samples'].shape[3] * 8
                        context_dim = unet_config.get('context_dim')
                        longer_side = width if width > height else height
                        if context_dim is not None and longer_side > context_dim:
                            width_downscale_factor = float(width / context_dim)
                            height_downscale_factor = float(height / context_dim)
                            if width_downscale_factor > 1.75:
                                log_node_warn("Patch model unet add downscale...")
                                log_node_warn("Downscale factor:" + str(width_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], width_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                            elif height_downscale_factor > 1.25:
                                log_node_warn("Patch model unet add downscale....")
                                log_node_warn("Downscale factor:" + str(height_downscale_factor))
                                (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], height_downscale_factor, 0, 0.35, True, "bicubic",
                                                            "bicubic")
                else:
                    cls = ALL_NODE_CLASS_MAPPINGS['PatchModelAddDownscale']
                    log_node_warn("Patch model unet add downscale....")
                    log_node_warn("Downscale factor:" + str(downscale_options['downscale_factor']))
                    (samp_model,) = cls().patch(samp_model, downscale_options['block_number'], downscale_options['downscale_factor'], downscale_options['start_percent'], downscale_options['end_percent'], downscale_options['downscale_after_skip'], downscale_options['downscale_method'], downscale_options['upscale_method'])
            return samp_model

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                 samp_negative,
                                 steps, start_step, last_step, cfg, sampler_name, scheduler, denoise,
                                 image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                 preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise, samp_custom=None, noise_device='cpu'):

            # LayerDiffusion
            layerDiffuse = None
            samp_blend_samples = None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None
            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                samp_blend_samples = pipe["blend_samples"] if "blend_samples" in pipe else None
                additional_cond = pipe["loader_settings"]['layer_diffusion_cond'] if "layer_diffusion_cond" in pipe[
                    'loader_settings'] else (None, None, None)
                method = layerDiffuse.get_layer_diffusion_method(pipe['loader_settings']['layer_diffusion_method'],
                                                         samp_blend_samples is not None)

                images = pipe["images"] if "images" in pipe else None
                weight = pipe['loader_settings']['layer_diffusion_weight'] if 'layer_diffusion_weight' in pipe[
                    'loader_settings'] else 1.0
                samp_model, samp_positive, samp_negative = layerDiffuse.apply_layer_diffusion(samp_model, method, weight,
                                                                                      samp_samples, samp_blend_samples,
                                                                                      samp_positive, samp_negative,
                                                                                      images, additional_cond)
                resolution = pipe['loader_settings']['resolution'] if 'resolution' in pipe['loader_settings'] else "自定义 X 自定义"
                empty_latent_width = pipe['loader_settings']['empty_latent_width'] if 'empty_latent_width' in pipe['loader_settings'] else 512
                empty_latent_height = pipe['loader_settings']['empty_latent_height'] if 'empty_latent_height' in pipe['loader_settings'] else 512
                batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
                samp_samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)
            # 推理初始时间
            start_time = int(time.time() * 1000)
            # 开始推理
            if samp_custom is not None:
                _guider, _sampler, sigmas = self.get_sampler_custom(samp_model, samp_positive, samp_negative, samp_custom)
                samp_samples, samp_blend_samples = sampler.custom_advanced_ksampler(_guider, _sampler, sigmas, samp_samples, add_noise, samp_seed, preview_latent=preview_latent)
            elif scheduler == 'align_your_steps':
                model_type = get_sd_version(samp_model)
                if model_type == 'unknown':
                    model_type = 'sdxl'
                sigmas, = alignYourStepsScheduler().get_sigmas(model_type.upper(), steps, denoise)
                _sampler = comfy.samplers.sampler_object(sampler_name)
                samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, _sampler, sigmas, samp_positive, samp_negative, samp_samples, disable_noise=disable_noise, preview_latent=preview_latent, noise_device=noise_device)
            elif scheduler == 'gits':
                sigmas, = gitsScheduler().get_sigmas(coeff=1.2, steps=steps, denoise=denoise)
                _sampler = comfy.samplers.sampler_object(sampler_name)
                samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, _sampler, sigmas, samp_positive, samp_negative, samp_samples, disable_noise=disable_noise, preview_latent=preview_latent, noise_device=noise_device)
            else:
                samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise, noise_device=noise_device)
            # 推理结束时间
            end_time = int(time.time() * 1000)
            latent = samp_samples["samples"]

            # 解码图片
            if image_output == 'None':
                samp_images, new_images, alpha, results = None, None, None, None
                spent_time = 'Diffusion:' + str((end_time - start_time) / 1000) + '″'
            else:
                if tile_size is not None:
                    samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
                else:
                    samp_images = samp_vae.decode(latent).cpu()
                if len(samp_images.shape) == 5:  # Combine batches
                    samp_images = samp_images.reshape(-1, samp_images.shape[-3], samp_images.shape[-2], samp_images.shape[-1])
                # LayerDiffusion Decode
                if layerDiffuse is not None:
                    new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latent, samp_blend_samples, samp_images, samp_model)
                else:
                    new_images = samp_images
                    alpha = None

                # 推理总耗时（包含解码）
                end_decode_time = int(time.time() * 1000)
                spent_time = 'Diffusion:' + str((end_time-start_time)/1000)+'″, VAEDecode:' + str((end_decode_time-end_time)/1000)+'″ '

                results = easySave(new_images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": samp_blend_samples,
                "images": new_images,
                "samp_images": samp_images,
                "alpha": alpha,
                "seed": samp_seed,

                "loader_settings": {
                    **pipe["loader_settings"],
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": denoise,
                    "add_noise": add_noise,
                    "spent_time": spent_time
                }
            }

            del pipe

            if image_output in ("Hide", "Hide&Save", "None"):
                return {"ui":{}, "result":sampler.get_output(new_pipe,)}

            if image_output in ("Sender", "Sender&Save"):
                PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe,)}

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom, noise_device):

            sampleXYplot = easyXYPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id, sampler, easyCache)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                            samp_negative, steps, 0, 10000, cfg,
                                            sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt,
                                            extra_pnginfo, my_unique_id, preview_latent, samp_custom=samp_custom, noise_device=noise_device)

            # Downscale Model Unet
            if samp_model is not None and downscale_options is not None:
                samp_model = downscale_model_unet(samp_model)

            blend_samples = pipe['blend_samples'] if "blend_samples" in pipe else None
            layer_diffusion_method = pipe['loader_settings']['layer_diffusion_method'] if 'layer_diffusion_method' in pipe['loader_settings'] else None

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": pipe["loader_settings"]["lora_name"] if "lora_name" in pipe["loader_settings"] else None,
                "lora_model_strength": pipe["loader_settings"]["lora_model_strength"] if "lora_model_strength" in pipe["loader_settings"] else 1.0,
                "lora_clip_strength": pipe["loader_settings"]["lora_clip_strength"] if "lora_clip_strength" in pipe["loader_settings"] else 1.0,
                "lora_stack":  pipe["loader_settings"]["lora_stack"] if "lora_stack" in pipe["loader_settings"] else None,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "seed": samp_seed,
                "images": pipe['images'],

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive,
                "negative_cond": samp_negative,
                "noise_device":noise_device,

                "ckpt_name": pipe['loader_settings']['ckpt_name'] if "ckpt_name" in pipe["loader_settings"] else None,
                "vae_name": pipe['loader_settings']['vae_name'] if "vae_name" in pipe["loader_settings"] else None,
                "clip_skip": pipe['loader_settings']['clip_skip'] if "clip_skip" in pipe["loader_settings"] else None,
                "positive": pipe['loader_settings']['positive'] if "positive" in pipe["loader_settings"] else None,
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'] if "positive_token_normalization" in pipe["loader_settings"] else None,
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'] if "positive_weight_interpretation" in pipe["loader_settings"] else None,
                "negative": pipe['loader_settings']['negative'] if "negative" in pipe["loader_settings"] else None,
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'] if "negative_token_normalization" in pipe["loader_settings"] else None,
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'] if "negative_weight_interpretation" in pipe["loader_settings"] else None,
            }

            if "models" in pipe["loader_settings"]:
                plot_image_vars["models"] = pipe["loader_settings"]["models"]
            if "vae_use" in pipe["loader_settings"]:
                plot_image_vars["vae_use"] = pipe["loader_settings"]["vae_use"]
            if "a1111_prompt_style" in pipe["loader_settings"]:
                plot_image_vars["a1111_prompt_style"] = pipe["loader_settings"]["a1111_prompt_style"]
            if "cnet_stack" in pipe["loader_settings"]:
                plot_image_vars["cnet"] = pipe["loader_settings"]["cnet_stack"]
            if "positive_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["positive_cond_stack"] = pipe["loader_settings"]["positive_cond_stack"]
            if "negative_cond_stack" in pipe["loader_settings"]:
                plot_image_vars["negative_cond_stack"] = pipe["loader_settings"]["negative_cond_stack"]
            if layer_diffusion_method:
                plot_image_vars["layer_diffusion_method"] = layer_diffusion_method
            if "layer_diffusion_weight" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_weight"] = pipe['loader_settings']['layer_diffusion_weight']
            if "layer_diffusion_cond" in pipe["loader_settings"]:
                plot_image_vars["layer_diffusion_cond"] = pipe['loader_settings']['layer_diffusion_cond']
            if "empty_samples" in pipe["loader_settings"]:
                plot_image_vars["empty_samples"] = pipe["loader_settings"]['empty_samples']

            latent_image = sampleXYplot.get_latent(pipe["samples"])
            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step,
                                                              last_step, force_full_denoise, disable_noise)

            samp_samples = {"samples": latents_plot}

            images, image_list = sampleXYplot.plot_images_and_labels(plot_image_vars)

            # Generate output_images
            output_images = torch.stack([tensor.squeeze() for tensor in image_list])

            if layer_diffusion_method is not None:
                layerDiffuse = LayerDiffuse()
                new_images, samp_images, alpha = layerDiffuse.layer_diffusion_decode(layer_diffusion_method, latents_plot, blend_samples,
                                                                             output_images, samp_model)
            else:
                new_images = output_images
                samp_images = output_images
                alpha = None

            results = easySave(images, save_prefix, image_output, prompt, extra_pnginfo)

            new_pipe = {
                **pipe,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "blend_samples": blend_samples,
                "samp_images": samp_images,
                "images": new_images,
                "seed": samp_seed,
                "alpha": alpha,

                "loader_settings": pipe["loader_settings"],
            }

            del pipe

            if image_output in ("Hide", "Hide&Save", "None"):
                return {"ui": {}, "result": sampler.get_output(new_pipe,)}

            return {"ui": {"images": results}, "result": sampler.get_output(new_pipe)}

        preview_latent = True
        if image_output in ("Hide", "Hide&Save", "None"):
            preview_latent = False

        xyplot_id = next((x for x in prompt if "XYPlot" in str(prompt[x]["class_type"])), None)
        if xyplot_id is None:
            xyPlot = None
        else:
            xyPlot = pipe["loader_settings"]["xyplot"] if "xyplot" in pipe["loader_settings"] else xyPlot

        # Fooocus model patch
        model_options = samp_model.model_options if samp_model.model_options else samp_model.model.model_options
        transformer_options = model_options["transformer_options"] if "transformer_options" in model_options else {}
        if "fooocus" in transformer_options:
            from ..modules.fooocus import applyFooocusInpaint
            del transformer_options["fooocus"]
            with applyFooocusInpaint():
                if xyPlot is not None:
                    return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive,
                                          samp_negative, steps, cfg, sampler_name, scheduler, denoise, image_output,
                                          link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id,
                                          preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom, noise_device)
                else:
                    return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed,
                                                samp_positive, samp_negative, steps, start_step, last_step, cfg,
                                                sampler_name, scheduler, denoise, image_output, link_id, save_prefix,
                                                tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent,
                                                force_full_denoise, disable_noise, samp_custom, noise_device)
        else:
            if xyPlot is not None:
                return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot, force_full_denoise, disable_noise, samp_custom, noise_device)
            else:
                return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, steps, start_step, last_step, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, tile_size, prompt, extra_pnginfo, my_unique_id, preview_latent, force_full_denoise, disable_noise, samp_custom, noise_device)

# 简易采样器
class samplerSimple(samplerFull):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save", "None"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None, None, None, None, image_output, link_id, save_prefix,
                                 None, model, None, None, None, None, None, None,
                                 None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

class samplerSimpleCustom(samplerFull):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save", "None"],{"default": "None"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("pipe", "output", "denoised_output", "image")
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        result = super().run(pipe, None, None, None, None, None, image_output, link_id, save_prefix,
                                 None, model, None, None, None, None, None, None,
                                 None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

        pipe = result["result"][0] if "result" in result else None

        return ({"ui": result['ui'], "result": (pipe, pipe["samples"], pipe["blend_samples"], pipe["images"])})

# 简易采样器 (Tiled)
class samplerSimpleTiled(samplerFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save", "None"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "tiled"
    CATEGORY = "EasyUse/Sampler"

    def tiled(self, pipe, tile_size=512, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

# 简易采样器 (LayerDiffusion)
class samplerSimpleLayerDiffusion(samplerFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"], {"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("pipe", "final_image", "original_image", "alpha")
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "layerDiffusion"
    CATEGORY = "EasyUse/Sampler"

    def layerDiffusion(self, pipe, image_output='preview', link_id=0, save_prefix='ComfyUI', model=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        result = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)
        pipe = result["result"][0] if "result" in result else None
        return ({"ui":result['ui'], "result":(pipe, pipe["images"], pipe["samp_images"], pipe["alpha"])})

# 简易采样器(收缩Unet)
class samplerSimpleDownscaleUnet(samplerFull):

    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "downscale_mode": (["None", "Auto", "Custom"],{"default": "Auto"}),
                 "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                 "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                 "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "downscale_after_skip": ("BOOLEAN", {"default": True}),
                 "downscale_method": (s.upscale_methods,),
                 "upscale_method": (s.upscale_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "downscale_unet"
    CATEGORY = "EasyUse/Sampler"

    def downscale_unet(self, pipe, downscale_mode, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        downscale_options = None
        if downscale_mode == 'Auto':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": None,
                "start_percent": 0,
                "end_percent":0.35,
                "downscale_after_skip": True,
                "downscale_method": "bicubic",
                "upscale_method": "bicubic"
            }
        elif downscale_mode == 'Custom':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": downscale_factor,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "downscale_after_skip": downscale_after_skip,
                "downscale_method": downscale_method,
                "upscale_method": upscale_method
            }

        return super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise, downscale_options)
# 简易采样器 (内补)
class samplerSimpleInpainting(samplerFull):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 "additional": (["None", "InpaintModelCond", "Differential Diffusion", "Fooocus Inpaint", "Fooocus Inpaint + DD", "Brushnet Random", "Brushnet Random + DD", "Brushnet Segmentation", "Brushnet Segmentation + DD"],{"default": "None"})
                 },
                "optional": {
                    "model": ("MODEL",),
                    "mask": ("MASK",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "VAE")
    RETURN_NAMES = ("pipe", "image", "vae")
    OUTPUT_NODE = True
    FUNCTION = "inpainting"
    CATEGORY = "EasyUse/Sampler"

    def dd(self, model, positive, negative, pixels, vae, mask):
        positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask=True)
        cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
        if cls is not None:
            model, = cls().apply(model)
        else:
            raise Exception("Differential Diffusion not found,please update comfyui")
        return positive, negative, latent, model

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*.(safetensors|bin)$'
        elif type == 'segmentation':
            brush_model = BRUSHNET_MODELS['segmentation_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.segmentation.mask.sdxl.*.(safetensors|bin)$'
            else:
                pattern = 'brushnet.segmentation.mask.*.(safetensors|bin)$'


        brushfile = [e for e in folder_paths.get_filename_list('inpaint') if re.search(pattern, e, re.IGNORECASE)]
        brushname = brushfile[0] if brushfile else None
        if not brushname:
            from urllib.parse import urlparse
            get_local_filepath(brush_model, INPAINT_DIR)
            parsed_url = urlparse(brush_model)
            brushname = os.path.basename(parsed_url.path)
        return brushname

    def apply_brushnet(self, brushname, model, vae, image, mask, positive, negative, scale=1.0, start_at=0, end_at=10000):
        if "BrushNetLoader" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("BrushNetLoader not found,please install ComfyUI-BrushNet")
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNetLoader']
        brushnet, = cls().brushnet_loading(brushname, 'float16')
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNet']
        m, positive, negative, latent = cls().model_update(model=model, vae=vae, image=image, mask=mask, brushnet=brushnet, positive=positive, negative=negative, scale=scale, start_at=start_at, end_at=end_at)
        return m, positive, negative, latent

    def inpainting(self, pipe, grow_mask_by, image_output, link_id, save_prefix, additional, model=None, mask=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        _model = model if model is not None else pipe['model']
        latent = pipe['samples'] if 'samples' in pipe else None
        positive = pipe['positive']
        negative = pipe['negative']
        images = pipe["images"] if pipe and "images" in pipe else None
        vae = pipe["vae"] if pipe and "vae" in pipe else None
        if 'noise_mask' in latent and mask is None:
            mask = latent['noise_mask']
        elif mask is not None:
            if images is None:
                raise Exception("No Images found")
            if vae is None:
                raise Exception("No VAE found")

        if additional == 'Differential Diffusion':
            positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
        elif additional == 'InpaintModelCond':
            if mask is not None:
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
            positive, negative, latent = InpaintModelConditioning().encode(positive, negative, images, vae, mask, True)
        elif additional == 'Fooocus Inpaint':
            head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
            patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
            if mask is not None:
                latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
            _model, = ALL_NODE_CLASS_MAPPINGS['easy applyFooocusInpaint']().apply(_model, latent, head, patch)
        elif additional == 'Fooocus Inpaint + DD':
            head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
            patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
            if mask is not None:
                latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
            _model, = ALL_NODE_CLASS_MAPPINGS['easy applyFooocusInpaint']().apply(_model, latent, head, patch)
            positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
        elif additional == 'Brushnet Random':
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
            brush_name = self.get_brushnet_model('random', _model)
            _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive,
                                                                     negative)
        elif additional == 'Brushnet Random + DD':
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
            brush_name = self.get_brushnet_model('random', _model)
            _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive,
                                                                     negative)
            positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
        elif additional == 'Brushnet Segmentation':
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
            brush_name = self.get_brushnet_model('segmentation', _model)
            _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive,
                                                                     negative)
        elif additional == 'Brushnet Segmentation + DD':
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
            brush_name = self.get_brushnet_model('segmentation', _model)
            _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive,
                                                                     negative)
            positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
        else:
            latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)

        results = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, _model, positive, negative, latent, vae, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

        result = results['result']

        return {"ui":results['ui'],"result":(result[0], result[1], result[0]['vae'],)}

# SDTurbo采样器
class samplerSDTurbo:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                     "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                    {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",
                     "my_unique_id": "UNIQUE_ID",
                     "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                     }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "run"

    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None,):
        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        samp_model = pipe["model"] if model is None else model
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = pipe["samples"]
        samp_vae = pipe["vae"]
        samp_clip = pipe["clip"]

        samp_seed = pipe['seed']

        samp_sampler = pipe['loader_settings']['sampler']

        sigmas = pipe['loader_settings']['sigmas']
        cfg = pipe['loader_settings']['cfg']
        steps = pipe['loader_settings']['steps']

        disable_noise = False

        preview_latent = True
        if image_output in ("Hide", "Hide&Save"):
            preview_latent = False

        # 推理初始时间
        start_time = int(time.time() * 1000)
        # 开始推理
        samp_samples = sampler.custom_ksampler(samp_model, samp_seed, steps, cfg, samp_sampler, sigmas, samp_positive, samp_negative, samp_samples,
                        disable_noise, preview_latent)
        # 推理结束时间
        end_time = int(time.time() * 1000)

        latent = samp_samples['samples']

        # 解码图片
        if tile_size is not None:
            samp_images = samp_vae.decode_tiled(latent, tile_x=tile_size // 8, tile_y=tile_size // 8, )
        else:
            samp_images = samp_vae.decode(latent).cpu()

        # 推理总耗时（包含解码）
        end_decode_time = int(time.time() * 1000)
        spent_time = 'Diffusion:' + str((end_time - start_time) / 1000) + '″, VAEDecode:' + str(
            (end_decode_time - end_time) / 1000) + '″ '

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)

        results = easySave(samp_images, save_prefix, image_output, prompt, extra_pnginfo)
        sampler.update_value_by_id("results", my_unique_id, results)

        new_pipe = {
            "model": samp_model,
            "positive": samp_positive,
            "negative": samp_negative,
            "vae": samp_vae,
            "clip": samp_clip,

            "samples": samp_samples,
            "images": samp_images,
            "seed": samp_seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "spent_time": spent_time
            }
        }

        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

        if image_output in ("Sender", "Sender&Save"):
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results},
                "result": sampler.get_output(new_pipe, )}


# Cascade完整采样器
class samplerCascadeFull:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "encode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                     "decode_vae_name": (["None"] + folder_paths.get_filename_list("vae"),),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default":"euler_ancestral"}),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default":"simple"}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],),
                     "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                     "save_prefix": ("STRING", {"default": "ComfyUI"}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                 },

                "optional": {
                    "image_to_latent_c": ("IMAGE",),
                    "latent_c": ("LATENT",),
                    "model_c": ("MODEL",),
                },
                 "hidden":{"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "LATENT")
    RETURN_NAMES = ("pipe", "model_b", "latent_b")
    OUTPUT_NODE = True

    FUNCTION = "run"
    CATEGORY = "EasyUse/Sampler"

    def run(self, pipe, encode_vae_name, decode_vae_name, steps, cfg, sampler_name, scheduler, denoise, image_output, link_id, save_prefix, seed, image_to_latent_c=None, latent_c=None, model_c=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        encode_vae_name = encode_vae_name if encode_vae_name is not None else pipe['loader_settings']['encode_vae_name']
        decode_vae_name = decode_vae_name if decode_vae_name is not None else pipe['loader_settings']['decode_vae_name']

        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent_c is not None:
            if encode_vae_name != 'None':
                encode_vae = easyCache.load_vae(encode_vae_name)
            else:
                encode_vae = pipe['vae'][0]
            if "compression" not in pipe["loader_settings"]:
                raise Exception("compression is not found")

            compression = pipe["loader_settings"]['compression']
            width = image_to_latent_c.shape[-2]
            height = image_to_latent_c.shape[-3]
            out_width = (width // compression) * encode_vae.downscale_ratio
            out_height = (height // compression) * encode_vae.downscale_ratio

            s = comfy.utils.common_upscale(image_to_latent_c.movedim(-1, 1), out_width, out_height, "bicubic",
                                           "center").movedim(1, -1)
            latent_c = encode_vae.encode(s[:, :, :, :3])
            latent_b = torch.zeros([latent_c.shape[0], 4, height // 4, width // 4])

            samples_c = {"samples": latent_c}
            samples_c = RepeatLatentBatch().repeat(samples_c, batch_size)[0]

            samples_b = {"samples": latent_b}
            samples_b = RepeatLatentBatch().repeat(samples_b, batch_size)[0]
            images = image_to_latent_c
        elif latent_c is not None:
            samples_c = latent_c
            samples_b = pipe["samples"][1]
            images = pipe["images"]
        else:
            samples_c = pipe["samples"][0]
            samples_b = pipe["samples"][1]
            images = pipe["images"]

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)
        samp_model = model_c if model_c else pipe["model"][0]
        samp_positive = pipe["positive"]
        samp_negative = pipe["negative"]
        samp_samples = samples_c

        samp_seed = seed if seed is not None else pipe['seed']

        steps = steps if steps is not None else pipe['loader_settings']['steps']
        start_step = pipe['loader_settings']['start_step'] if 'start_step' in pipe['loader_settings'] else 0
        last_step = pipe['loader_settings']['last_step'] if 'last_step' in pipe['loader_settings'] else 10000
        cfg = cfg if cfg is not None else pipe['loader_settings']['cfg']
        sampler_name = sampler_name if sampler_name is not None else pipe['loader_settings']['sampler_name']
        scheduler = scheduler if scheduler is not None else pipe['loader_settings']['scheduler']
        denoise = denoise if denoise is not None else pipe['loader_settings']['denoise']
        noise_device = 'gpu' if "a1111_prompt_style" in pipe['loader_settings'] and pipe['loader_settings']['a1111_prompt_style'] else 'cpu'
        # 推理初始时间
        start_time = int(time.time() * 1000)
        # 开始推理
        samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler,
                                               samp_positive, samp_negative, samp_samples, denoise=denoise,
                                               preview_latent=False, start_step=start_step,
                                               last_step=last_step, force_full_denoise=False,
                                               disable_noise=False, noise_device=noise_device)
        # 推理结束时间
        end_time = int(time.time() * 1000)
        stage_c = samp_samples["samples"]
        results = None

        if image_output not in ['Hide', 'Hide&Save']:
            if decode_vae_name != 'None':
                decode_vae = easyCache.load_vae(decode_vae_name)
            else:
                decode_vae = pipe['vae'][0]
            samp_images = decode_vae.decode(stage_c).cpu()

            results = easySave(samp_images, save_prefix, image_output, prompt, extra_pnginfo)
            sampler.update_value_by_id("results", my_unique_id, results)

        # 推理总耗时（包含解码）
        end_decode_time = int(time.time() * 1000)
        spent_time = 'Diffusion:' + str((end_time - start_time) / 1000) + '″, VAEDecode:' + str(
            (end_decode_time - end_time) / 1000) + '″ '

        # Clean loaded_objects
        easyCache.update_loaded_objects(prompt)
        # zero_out
        c1 = []
        for t in samp_positive:
            d = t[1].copy()
            if "pooled_output" in d:
                d["pooled_output"] = torch.zeros_like(d["pooled_output"])
            n = [torch.zeros_like(t[0]), d]
            c1.append(n)
        # stage_b_conditioning
        c2 = []
        for t in c1:
            d = t[1].copy()
            d['stable_cascade_prior'] = stage_c
            n = [t[0], d]
            c2.append(n)


        new_pipe = {
            "model": pipe['model'][1],
            "positive": c2,
            "negative": c1,
            "vae": pipe['vae'][1],
            "clip": pipe['clip'],

            "samples": samples_b,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "spent_time": spent_time
            }
        }
        sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

        del pipe

        if image_output in ("Hide", "Hide&Save"):
            return {"ui": {},
                    "result": sampler.get_output(new_pipe, )}

        if image_output in ("Sender", "Sender&Save") and results is not None:
            PromptServer.instance.send_sync("img-send", {"link_id": link_id, "images": results})

        return {"ui": {"images": results}, "result": (new_pipe, new_pipe['model'], new_pipe['samples'])}

# 简易采样器Cascade
class samplerCascadeSimple(samplerCascadeFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"], {"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model_c": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model_c=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None,None, None,None,None,None, image_output, link_id, save_prefix,
                               None, None, None, model_c, tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

class unsampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "end_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
             "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
             "normalize": (["disable", "enable"],),

             },
            "optional": {
                "pipe": ("PIPE_LINE",),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("PIPE_LINE", "LATENT",)
    RETURN_NAMES = ("pipe", "latent",)
    FUNCTION = "unsampler"

    CATEGORY = "EasyUse/Sampler"

    def unsampler(self, cfg, sampler_name, steps, end_at_step, scheduler, normalize, pipe=None, optional_model=None, optional_positive=None, optional_negative=None,
                  optional_latent=None):

        model = optional_model if optional_model is not None else pipe["model"]
        positive = optional_positive if optional_positive is not None else pipe["positive"]
        negative = optional_negative if optional_negative is not None else pipe["negative"]
        latent_image = optional_latent if optional_latent is not None else pipe["samples"]

        normalize = normalize == "enable"
        device = comfy.model_management.get_torch_device()
        latent = latent_image
        latent_image = latent["samples"]

        end_at_step = min(end_at_step, steps - 1)
        end_at_step = steps - end_at_step

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = comfy.sampler_helpers.prepare_mask(latent["noise_mask"], noise.shape, device)

        noise = noise.to(device)
        latent_image = latent_image.to(device)

        _positive = comfy.sampler_helpers.convert_cond(positive)
        _negative = comfy.sampler_helpers.convert_cond(negative)
        models, inference_memory = comfy.sampler_helpers.get_additional_models({"positive": _positive, "negative": _negative}, model.model_dtype())


        comfy.model_management.load_models_gpu([model] + models, model.memory_required(noise.shape) + inference_memory)

        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())

        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)

        sigmas = sampler.sigmas.flip(0) + 0.0001

        pbar = comfy.utils.ProgressBar(steps)

        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image,
                                 force_full_denoise=False, denoise_mask=noise_mask, sigmas=sigmas, start_step=0,
                                 last_step=end_at_step, callback=callback)
        if normalize:
            # technically doesn't normalize because unsampling is not guaranteed to end at a std given by the schedule
            samples -= samples.mean()
            samples /= samples.std()
        samples = samples.cpu()

        comfy.sample.cleanup_additional_models(models)

        out = latent.copy()
        out["samples"] = samples

        if pipe is None:
            pipe = {}

        new_pipe = {
            **pipe,
            "samples": out
        }

        return (new_pipe, out,)


NODE_CLASS_MAPPINGS = {
    # kSampler k采样器
    "easy fullkSampler": samplerFull,
    "easy kSampler": samplerSimple,
    "easy kSamplerCustom": samplerSimpleCustom,
    "easy kSamplerTiled": samplerSimpleTiled,
    "easy kSamplerLayerDiffusion": samplerSimpleLayerDiffusion,
    "easy kSamplerInpainting": samplerSimpleInpainting,
    "easy kSamplerDownscaleUnet": samplerSimpleDownscaleUnet,
    "easy kSamplerSDTurbo": samplerSDTurbo,
    "easy fullCascadeKSampler": samplerCascadeFull,
    "easy cascadeKSampler": samplerCascadeSimple,
    "easy unSampler": unsampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "easy kSampler": "EasyKSampler",
    "easy kSamplerCustom": "EasyKSampler (Custom)",
    "easy fullkSampler": "EasyKSampler (Full)",
    "easy kSamplerTiled": "EasyKSampler (Tiled Decode)",
    "easy kSamplerLayerDiffusion": "EasyKSampler (LayerDiffuse)",
    "easy kSamplerInpainting": "EasyKSampler (Inpainting)",
    "easy kSamplerDownscaleUnet": "EasyKsampler (Downscale Unet)",
    "easy kSamplerSDTurbo": "EasyKSampler (SDTurbo)",
    "easy cascadeKSampler": "EasyCascadeKsampler",
    "easy fullCascadeKSampler": "EasyCascadeKsampler (Full)",
    "easy unSampler": "EasyUnSampler",
}
import comfy.samplers
import comfy.utils
import torch
import numpy as np
from tqdm.auto import trange, tqdm
import math


@torch.no_grad()
def sample_lcm_upscale(model, x, sigmas, extra_args=None, callback=None, disable=None, total_upscale=2.0, upscale_method="bislerp", upscale_steps=None):
    extra_args = {} if extra_args is None else extra_args

    if upscale_steps is None:
        upscale_steps = max(len(sigmas) // 2 + 1, 2)
    else:
        upscale_steps += 1
        upscale_steps = min(upscale_steps, len(sigmas) + 1)

    upscales = np.linspace(1.0, total_upscale, upscale_steps)[1:]

    orig_shape = x.size()
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        x = denoised
        if i < len(upscales):
            x = comfy.utils.common_upscale(x, round(orig_shape[-1] * upscales[i]), round(orig_shape[-2] * upscales[i]), upscale_method, "disabled")

        if sigmas[i + 1] > 0:
            x += sigmas[i + 1] * torch.randn_like(x)
    return x


class SamplerLCMUpscale:
    upscale_methods = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"scale_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.01}),
                     "scale_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                     "upscale_method": (s.upscale_methods,),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, scale_ratio, scale_steps, upscale_method):
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscale, extra_options={"total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return (sampler, )

from comfy.k_diffusion.sampling import to_d
import comfy.model_patcher

@torch.no_grad()
def sample_euler_pp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args

    temp = [0]
    def post_cfg_function(args):
        temp[0] = args["uncond_denoised"]
        return args["denoised"]

    model_options = extra_args.get("model_options", {}).copy()
    extra_args["model_options"] = comfy.model_patcher.set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=True)

    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma_hat = sigmas[i]
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = to_d(x - denoised + temp[0], sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


class SamplerEulerCFGpp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"version": (["regular", "alternative"],),}
               }
    RETURN_TYPES = ("SAMPLER",)
    # CATEGORY = "sampling/custom_sampling/samplers"
    CATEGORY = "_for_testing"

    FUNCTION = "get_sampler"

    def get_sampler(self, version):
        if version == "alternative":
            sampler = comfy.samplers.KSAMPLER(sample_euler_pp)
        else:
            sampler = comfy.samplers.ksampler("euler_cfg_pp")
        return (sampler, )

NODE_CLASS_MAPPINGS = {
    "SamplerLCMUpscale": SamplerLCMUpscale,
    "SamplerEulerCFGpp": SamplerEulerCFGpp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerEulerCFGpp": "SamplerEulerCFG++",
}

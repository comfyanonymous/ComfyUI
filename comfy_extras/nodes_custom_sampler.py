import comfy.samplers
import comfy.sample
from comfy.k_diffusion import sampling as k_diffusion_sampling
import latent_preview
import torch
import comfy.utils
import node_helpers


class BasicScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return (sigmas, )


class KarrasScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

class ExponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min):
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        return (sigmas, )

class PolyexponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

class LaplaceScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step":0.1, "round": False}),
                     "beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step":0.1, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, mu, beta):
        sigmas = k_diffusion_sampling.get_sigmas_laplace(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, mu=mu, beta=beta)
        return (sigmas, )


class SDTurboScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
        sigmas = model.get_model_object("model_sampling").sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas, )

class BetaSamplingScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "alpha": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 50.0, "step":0.01, "round": False}),
                     "beta": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 50.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, alpha, beta):
        sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta)
        return (sigmas, )

class VPScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}), #TODO: fix default values
                     "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step":0.0001, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, beta_d, beta_min, eps_s):
        sigmas = k_diffusion_sampling.get_sigmas_vp(n=steps, beta_d=beta_d, beta_min=beta_min, eps_s=eps_s)
        return (sigmas, )

class SplitSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, step):
        sigmas1 = sigmas[:step + 1]
        sigmas2 = sigmas[step:]
        return (sigmas1, sigmas2)

class SplitSigmasDenoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, denoise):
        steps = max(sigmas.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        sigmas1 = sigmas[:-(total_steps)]
        sigmas2 = sigmas[-(total_steps + 1):]
        return (sigmas1, sigmas2)

class FlipSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas):
        if len(sigmas) == 0:
            return (sigmas,)

        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        return (sigmas,)

class KSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name):
        sampler = comfy.samplers.sampler_object(sampler_name)
        return (sampler, )

class SamplerDPMPP_3M_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_3m_sde"
        else:
            sampler_name = "dpmpp_3m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerDPMPP_2M_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"solver_type": (['midpoint', 'heun'], ),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, solver_type, eta, s_noise, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_2m_sde"
        else:
            sampler_name = "dpmpp_2m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type})
        return (sampler, )


class SamplerDPMPP_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_sde"
        else:
            sampler_name = "dpmpp_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r})
        return (sampler, )

class SamplerDPMPP_2S_Ancestral:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler("dpmpp_2s_ancestral", {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerEulerAncestral:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler("euler_ancestral", {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerEulerAncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step":0.01, "round": False}),
            }}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler(
            "euler_ancestral_cfg_pp",
            {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerLMS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"order": ("INT", {"default": 4, "min": 1, "max": 100}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, order):
        sampler = comfy.samplers.ksampler("lms", {"order": order})
        return (sampler, )

class SamplerDPMAdaptative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"order": ("INT", {"default": 3, "min": 2, "max": 3}),
                     "rtol": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "atol": ("FLOAT", {"default": 0.0078, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "h_init": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "pcoeff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "icoeff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "dcoeff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "accept_safety": ("FLOAT", {"default": 0.81, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise):
        sampler = comfy.samplers.ksampler("dpm_adaptive", {"order": order, "rtol": rtol, "atol": atol, "h_init": h_init, "pcoeff": pcoeff,
                                                              "icoeff": icoeff, "dcoeff": dcoeff, "accept_safety": accept_safety, "eta": eta,
                                                              "s_noise":s_noise })
        return (sampler, )

class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

class SamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

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
        return (out, out_denoised)

class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class BasicGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "conditioning": ("CONDITIONING", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning):
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return (guider,)

class CFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)

class Guider_DualCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, cfg2):
        self.cfg1 = cfg1
        self.cfg2 = cfg2

    def set_conds(self, positive, middle, negative):
        middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        middle_cond = self.conds.get("middle", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, self.conds.get("positive", None)], x, timestep, model_options)
        return comfy.samplers.cfg_function(self.inner_model, out[1], out[0], self.cfg2, x, timestep, model_options=model_options, cond=middle_cond, uncond=negative_cond) + (out[2] - out[1]) * self.cfg1

class DualCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "cond1": ("CONDITIONING", ),
                    "cond2": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg_conds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg_cond2_negative": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative):
        guider = Guider_DualCFG(model)
        guider.set_conds(cond1, cond2, negative)
        guider.set_cfg(cfg_conds, cfg_cond2_negative)
        return (guider,)

class DisableNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                     }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self):
        return (Noise_EmptyNoise(),)


class RandomNoise(DisableNoise):
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     }
                }

    def get_noise(self, noise_seed):
        return (Noise_RandomNoise(noise_seed),)


class SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

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
        return (out, out_denoised)

class AddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "noise": ("NOISE", ),
                     "sigmas": ("SIGMAS", ),
                     "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "add_noise"

    CATEGORY = "_for_testing/custom_sampling/noise"

    def add_noise(self, model, noise, sigmas, latent_image):
        if len(sigmas) == 0:
            return latent_image

        latent = latent_image
        latent_image = latent["samples"]

        noisy = noise.generate_noise(latent)

        model_sampling = model.get_model_object("model_sampling")
        process_latent_out = model.get_model_object("process_latent_out")
        process_latent_in = model.get_model_object("process_latent_in")

        if len(sigmas) > 1:
            scale = torch.abs(sigmas[0] - sigmas[-1])
        else:
            scale = sigmas[0]

        if torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = process_latent_in(latent_image)
        noisy = model_sampling.noise_scaling(scale, noisy, latent_image)
        noisy = process_latent_out(noisy)
        noisy = torch.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)

        out = latent.copy()
        out["samples"] = noisy
        return (out,)


NODE_CLASS_MAPPINGS = {
    "SamplerCustom": SamplerCustom,
    "BasicScheduler": BasicScheduler,
    "KarrasScheduler": KarrasScheduler,
    "ExponentialScheduler": ExponentialScheduler,
    "PolyexponentialScheduler": PolyexponentialScheduler,
    "LaplaceScheduler": LaplaceScheduler,
    "VPScheduler": VPScheduler,
    "BetaSamplingScheduler": BetaSamplingScheduler,
    "SDTurboScheduler": SDTurboScheduler,
    "KSamplerSelect": KSamplerSelect,
    "SamplerEulerAncestral": SamplerEulerAncestral,
    "SamplerEulerAncestralCFGPP": SamplerEulerAncestralCFGPP,
    "SamplerLMS": SamplerLMS,
    "SamplerDPMPP_3M_SDE": SamplerDPMPP_3M_SDE,
    "SamplerDPMPP_2M_SDE": SamplerDPMPP_2M_SDE,
    "SamplerDPMPP_SDE": SamplerDPMPP_SDE,
    "SamplerDPMPP_2S_Ancestral": SamplerDPMPP_2S_Ancestral,
    "SamplerDPMAdaptative": SamplerDPMAdaptative,
    "SplitSigmas": SplitSigmas,
    "SplitSigmasDenoise": SplitSigmasDenoise,
    "FlipSigmas": FlipSigmas,

    "CFGGuider": CFGGuider,
    "DualCFGGuider": DualCFGGuider,
    "BasicGuider": BasicGuider,
    "RandomNoise": RandomNoise,
    "DisableNoise": DisableNoise,
    "AddNoise": AddNoise,
    "SamplerCustomAdvanced": SamplerCustomAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerEulerAncestralCFGPP": "SamplerEulerAncestralCFG++",
}

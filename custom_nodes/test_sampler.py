import latent_preview
from custom_nodes.debug_model import DebugModel
from nodes import common_ksampler


class TestSampler:
    SCHEDULERS = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    SAMPLERS = ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddim", "uni_pc",
                "uni_pc_bh2"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (TestSampler.SAMPLERS,),
                     "scheduler": (TestSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "mixture": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                     }
                }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    FUNCTION = "sample"

    CATEGORY = "inflamously"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0,
               mixture=1.0):
        a_val = common_ksampler(model, seed, round(steps / 2), cfg, sampler_name, scheduler, positive, negative,
                                latent_image, denoise=.8)
        b_val = common_ksampler(model, seed + 1, round(steps / 2), cfg, sampler_name, scheduler, positive, negative,
                                a_val[0], denoise=.9)
        x_val = common_ksampler(model, seed + 2, round(steps), cfg, sampler_name, scheduler, positive, negative, b_val[0], denoise=denoise)
        return (x_val[0], a_val[0], b_val[0])

# c_val = [{"samples": None}]
# c_val[0]["samples"] = (a_val[0]["samples"] * 0.5 * (1.0 - mixture)) + (b_val[0]["samples"] * 0.5 * (0.0 + mixture))
# c_val[0]["samples"] = (a_val[0]["samples"] * (1.0 - mixture)) - (b_val[0]["samples"] * (0.0 + mixture))

NODE_CLASS_MAPPINGS = {
    "TestSampler": TestSampler
}

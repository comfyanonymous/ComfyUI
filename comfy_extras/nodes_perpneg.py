import torch
import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
import node_helpers

def perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale):
    pos = noise_pred_pos - noise_pred_nocond
    neg = noise_pred_neg - noise_pred_nocond

    perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
    perp_neg = perp * neg_scale
    cfg_result = noise_pred_nocond + cond_scale*(pos - perp_neg)
    return cfg_result

#TODO: This node should be removed, it has been replaced with PerpNegGuider
class PerpNeg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "empty_conditioning": ("CONDITIONING", ),
                             "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, empty_conditioning, neg_scale):
        m = model.clone()
        nocond = comfy.sampler_helpers.convert_cond(empty_conditioning)

        def cfg_function(args):
            model = args["model"]
            noise_pred_pos = args["cond_denoised"]
            noise_pred_neg = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            nocond_processed = comfy.samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

            (noise_pred_nocond,) = comfy.samplers.calc_cond_batch(model, [nocond_processed], x, sigma, model_options)

            cfg_result = x - perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale)
            return cfg_result

        m.set_model_sampler_cfg_function(cfg_function)

        return (m, )


class Guider_PerpNeg(comfy.samplers.CFGGuider):
    def set_conds(self, positive, negative, empty_negative_prompt):
        empty_negative_prompt = node_helpers.conditioning_set_values(empty_negative_prompt, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "empty_negative_prompt": empty_negative_prompt, "negative": negative})

    def set_cfg(self, cfg, neg_scale):
        self.cfg = cfg
        self.neg_scale = neg_scale

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)
        empty_cond = self.conds.get("empty_negative_prompt", None)

        out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, positive_cond, empty_cond], x, timestep, model_options)
        return perp_neg(x, out[1], out[0], out[2], self.neg_scale, self.cfg)

class PerpNegGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "empty_conditioning": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "_for_testing"

    def get_guider(self, model, positive, negative, empty_conditioning, cfg, neg_scale):
        guider = Guider_PerpNeg(model)
        guider.set_conds(positive, negative, empty_conditioning)
        guider.set_cfg(cfg, neg_scale)
        return (guider,)

NODE_CLASS_MAPPINGS = {
    "PerpNeg": PerpNeg,
    "PerpNegGuider": PerpNegGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerpNeg": "Perp-Neg (DEPRECATED by PerpNegGuider)",
}

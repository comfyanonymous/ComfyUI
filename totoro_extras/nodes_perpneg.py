import torch
import totoro.model_management
import totoro.sampler_helpers
import totoro.samplers
import totoro.utils
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
        nocond = totoro.sampler_helpers.convert_cond(empty_conditioning)

        def cfg_function(args):
            model = args["model"]
            noise_pred_pos = args["cond_denoised"]
            noise_pred_neg = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            nocond_processed = totoro.samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

            (noise_pred_nocond,) = totoro.samplers.calc_cond_batch(model, [nocond_processed], x, sigma, model_options)

            cfg_result = x - perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_nocond, neg_scale, cond_scale)
            return cfg_result

        m.set_model_sampler_cfg_function(cfg_function)

        return (m, )


class Guider_PerpNeg(totoro.samplers.CFGGuider):
    def set_conds(self, positive, negative, empty_negative_prompt):
        empty_negative_prompt = node_helpers.conditioning_set_values(empty_negative_prompt, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "empty_negative_prompt": empty_negative_prompt, "negative": negative})

    def set_cfg(self, cfg, neg_scale):
        self.cfg = cfg
        self.neg_scale = neg_scale

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        # in CFGGuider.predict_noise, we call sampling_function(), which uses cfg_function() to compute pos & neg
        # but we'd rather do a single batch of sampling pos, neg, and empty, so we call calc_cond_batch([pos,neg,empty]) directly
        
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)
        empty_cond = self.conds.get("empty_negative_prompt", None)

        (noise_pred_pos, noise_pred_neg, noise_pred_empty) = \
            totoro.samplers.calc_cond_batch(self.inner_model, [positive_cond, negative_cond, empty_cond], x, timestep, model_options)
        cfg_result = perp_neg(x, noise_pred_pos, noise_pred_neg, noise_pred_empty, self.neg_scale, self.cfg)

        # normally this would be done in cfg_function, but we skipped 
        # that for efficiency: we can compute the noise predictions in
        # a single call to calc_cond_batch() (rather than two)
        # so we replicate the hook here
        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {
                "denoised": cfg_result,
                "cond": positive_cond,
                "uncond": negative_cond,
                "model": self.inner_model,
                "uncond_denoised": noise_pred_neg,
                "cond_denoised": noise_pred_pos,
                "sigma": timestep,
                "model_options": model_options,
                "input": x,
                # not in the original call in samplers.py:cfg_function, but made available for future hooks
                "empty_cond": empty_cond,
                "empty_cond_denoised": noise_pred_empty,}
            cfg_result = fn(args)

        return cfg_result

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

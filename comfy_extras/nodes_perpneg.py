import torch
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils


class PerpNeg:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                             "empty_conditioning": ("CONDITIONING", ),
                             "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, empty_conditioning, neg_scale):
        m = model.clone()
        nocond = comfy.sample.convert_cond(empty_conditioning)

        def cfg_function(args):
            model = args["model"]
            noise_pred_pos = args["cond_denoised"]
            noise_pred_neg = args["uncond_denoised"]
            cond_scale = args["cond_scale"]
            x = args["input"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            nocond_processed = comfy.samplers.encode_model_conds(model.extra_conds, nocond, x, x.device, "negative")

            (noise_pred_nocond, _) = comfy.samplers.calc_cond_uncond_batch(model, nocond_processed, None, x, sigma, model_options)

            pos = noise_pred_pos - noise_pred_nocond
            neg = noise_pred_neg - noise_pred_nocond
            perp = neg - ((torch.mul(neg, pos).sum())/(torch.norm(pos)**2)) * pos
            perp_neg = perp * neg_scale
            cfg_result = noise_pred_nocond + cond_scale*(pos - perp_neg)
            cfg_result = x - cfg_result
            return cfg_result

        m.set_model_sampler_cfg_function(cfg_function)

        return (m, )


NODE_CLASS_MAPPINGS = {
    "PerpNeg": PerpNeg,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerpNeg": "Perp-Neg",
}

import torch

# https://github.com/WeichenFan/CFG-Zero-star
def optimized_scale(positive, negative):
    positive_flat = positive.reshape(positive.shape[0], -1)
    negative_flat = negative.reshape(negative.shape[0], -1)

    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)

    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star.reshape([positive.shape[0]] + [1] * (positive.ndim - 1))

class CFGZeroStar:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                            }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "advanced/guidance"

    def patch(self, model):
        m = model.clone()
        def cfg_zero_star(args):
            guidance_scale = args['cond_scale']
            x = args['input']
            cond_p = args['cond_denoised']
            uncond_p = args['uncond_denoised']
            out = args["denoised"]
            alpha = optimized_scale(x - cond_p, x - uncond_p)

            return out + uncond_p * (alpha - 1.0)  + guidance_scale * uncond_p * (1.0 - alpha)
        m.set_model_sampler_post_cfg_function(cfg_zero_star)
        return (m, )

class CFGNorm:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                            }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "advanced/guidance"
    EXPERIMENTAL = True

    def patch(self, model, strength):
        m = model.clone()
        def cfg_norm(args):
            cond_p = args['cond_denoised']
            pred_text_ = args["denoised"]

            norm_full_cond = torch.norm(cond_p, dim=1, keepdim=True)
            norm_pred_text = torch.norm(pred_text_, dim=1, keepdim=True)
            scale = (norm_full_cond / (norm_pred_text + 1e-8)).clamp(min=0.0, max=1.0)
            return pred_text_ * scale * strength

        m.set_model_sampler_post_cfg_function(cfg_norm)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "CFGZeroStar": CFGZeroStar,
    "CFGNorm": CFGNorm,
}

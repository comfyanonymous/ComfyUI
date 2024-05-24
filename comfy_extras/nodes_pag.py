#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

import comfy.model_patcher
import comfy.samplers

class PerturbedAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale):
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            return v

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            # Replace Self-attention with PAG
            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention, "attn1", unet_block, unet_block_id)
            (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - pag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)

NODE_CLASS_MAPPINGS = {
    "PerturbedAttentionGuidance": PerturbedAttentionGuidance,
}

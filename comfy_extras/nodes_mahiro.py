import torch
import torch.nn.functional as F

class Mahiro:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                            }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "_for_testing"
    DESCRIPTION = "Modify the guidance to scale more on the 'direction' of the positive prompt rather than the difference between the negative prompt."
    def patch(self, model):
        m = model.clone()
        def mahiro_normd(args):
            scale: float = args['cond_scale']
            cond_p: torch.Tensor = args['cond_denoised']
            uncond_p: torch.Tensor = args['uncond_denoised']
            #naive leap
            leap = cond_p * scale
            #sim with uncond leap
            u_leap = uncond_p * scale
            cfg = args["denoised"]
            merge = (leap + cfg) / 2
            normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
            normm = torch.sqrt(merge.abs()) * merge.sign()
            sim = F.cosine_similarity(normu, normm).mean()
            simsc = 2 * (sim+1)
            wm = (simsc*cfg + (4-simsc)*leap) / 4
            return wm
        m.set_model_sampler_post_cfg_function(mahiro_normd)
        return (m, )

NODE_CLASS_MAPPINGS = {
    "Mahiro": Mahiro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Mahiro": "Mahiro is so cute that she deserves a better guidance function!! (。・ω・。)",
}

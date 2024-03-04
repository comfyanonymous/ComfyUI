# code adapted from https://github.com/exx8/differential-diffusion

import torch

class DifferentialDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)


NODE_CLASS_MAPPINGS = {
    "DifferentialDiffusion": DifferentialDiffusion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DifferentialDiffusion": "Differential Diffusion",
}

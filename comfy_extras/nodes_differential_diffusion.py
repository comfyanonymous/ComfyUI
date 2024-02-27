# code adapted from https://github.com/exx8/differential-diffusion

import torch
import inspect

class DifferentialDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        DifferentialDiffusion.INIT = s.INIT = True
        return ""

    def __init__(self) -> None:
        DifferentialDiffusion.INIT = False
        self.sigmas: torch.Tensor = None
        self.mask_i = None
        self.valid_sigmas = False
        self.varying_sigmas_samplers = ["dpmpp_2s", "dpmpp_sde", "dpm_2", "heun", "restart"]

    def apply(self, model):
        model = model.clone()
        model.model_options["denoise_mask_function"] = self.forward
        return (model,)
    
    def init_sigmas(self, sigma: torch.Tensor, denoise_mask: torch.Tensor):
        self.__init__()
        self.sigmas, sampler = find_outer_instance("sigmas", callback=get_sigmas_and_sampler) or (None, "")
        self.valid_sigmas = not ("sample_" not in sampler or any(s in sampler for s in self.varying_sigmas_samplers)) or "generic" in sampler
        if self.sigmas is None:
            self.sigmas = sigma[:1].repeat(2)
            self.sigmas[-1:].zero_()
        ts = self.sigmas[:-1]
        self.sigmas_min = ts_min = ts.min()
        self.sigmas_max = ts_max = ts.max()
        if ts_min == ts_max: ts_min.zero_()
        if self.valid_sigmas:
            thresholds = (ts - ts_min) / (ts_max - ts_min)
            thresholds = thresholds.reshape(-1, 1, 1, 1, 1)
            mask = denoise_mask.unsqueeze(0)
            mask = (mask >= thresholds) & (mask > 0)
            mask = mask.to(denoise_mask.dtype)
            self.mask_i = iter(mask)
    
    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor):
        if self.sigmas is None or DifferentialDiffusion.INIT:
            self.init_sigmas(sigma, denoise_mask)
        if self.valid_sigmas:
            try:
                return next(self.mask_i)
            except StopIteration:
                self.valid_sigmas = False
        threshold = (sigma[0] - self.sigmas_min) / (self.sigmas_max - self.sigmas_min)
        mask = (denoise_mask >= threshold) & (denoise_mask > 0)
        return mask.to(denoise_mask.dtype)

def get_sigmas_and_sampler(frame, target):
    found = frame.f_locals[target]
    if isinstance(found, torch.Tensor) and found[-1] < 0.1:
        return found, frame.f_code.co_name
    return False

def find_outer_instance(target: str, target_type=None, callback=None):
    frame = inspect.currentframe()
    i = 0
    while frame and i < 100:
        if target in frame.f_locals:
            if callback is not None:
                res = callback(frame, target)
                if res:
                    return res
            else:
                found = frame.f_locals[target]
                if isinstance(found, target_type):
                    return found
        frame = frame.f_back
        i += 1
    return None

    
NODE_CLASS_MAPPINGS = {
    "DifferentialDiffusion": DifferentialDiffusion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DifferentialDiffusion": "Differential Diffusion",
}

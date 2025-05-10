import torch

def project(v0, v1):
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel, v0_orthogonal

class AGP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "norm_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "momentum": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling"

    def patch(self, model, eta, norm_threshold, momentum):
        running_avg = 0
        prev_shape = None
        
        def pre_cfg_function(args):
            nonlocal running_avg, prev_shape
            
            cond = args["conds_out"][0]
            uncond = args["conds_out"][1]
            
            guidance = cond - uncond
            
            if prev_shape is not None and guidance.shape != prev_shape:
                running_avg = 0
            prev_shape = guidance.shape
            
            if momentum > 0:
                if not torch.is_tensor(running_avg):
                    running_avg = guidance
                else:
                    running_avg = momentum * running_avg + guidance
                guidance = running_avg
            
            if norm_threshold > 0:
                guidance_norm = guidance.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale = torch.minimum(
                    torch.ones_like(guidance_norm), 
                    norm_threshold / guidance_norm
                )
                guidance = guidance * scale
            
            guidance_parallel, guidance_orthogonal = project(guidance, cond)
            modified_guidance = guidance_orthogonal + eta * guidance_parallel
            
            modified_cond = uncond + modified_guidance
            
            return [modified_cond, uncond]

        m = model.clone()
        m.set_model_sampler_pre_cfg_function(pre_cfg_function)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "AGP": AGP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AGP": "Adaptive Gradient Projection",
}
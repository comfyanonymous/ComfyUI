import torch
import comfy.model_management

RMSNorm = torch.nn.RMSNorm

def rms_norm(x, weight=None, eps=1e-6, fused=True):
    if not fused:
        orig_dtype = x.dtype
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + eps, -0.5)
        if weight is not None:
            weight = comfy.model_management.cast_to(weight, dtype=torch.float32, device=x.device)
            normed = normed * weight
        return normed.to(orig_dtype)

    if weight is None:
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps)
    else:
        return torch.nn.functional.rms_norm(x, weight.shape, weight=comfy.model_management.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)

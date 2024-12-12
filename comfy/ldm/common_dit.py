import torch
import comfy.ops

def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"
    pad_h = (patch_size[0] - img.shape[-2] % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - img.shape[-1] % patch_size[1]) % patch_size[1]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)

try:
    rms_norm_torch = torch.nn.functional.rms_norm
except:
    rms_norm_torch = None

def rms_norm(x, weight=None, eps=1e-6):
    if rms_norm_torch is not None and not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        if weight is None:
            return rms_norm_torch(x, (x.shape[-1],), eps=eps)
        else:
            return rms_norm_torch(x, weight.shape, weight=comfy.ops.cast_to(weight, dtype=x.dtype, device=x.device), eps=eps)
    else:
        r = x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
        if weight is None:
            return r
        else:
            return r * comfy.ops.cast_to(weight, dtype=x.dtype, device=x.device)

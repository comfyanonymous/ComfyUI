import torch
import comfy.rmsnorm


def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)


rms_norm = comfy.rmsnorm.rms_norm

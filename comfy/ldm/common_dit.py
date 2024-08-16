import torch

def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and torch.jit.is_tracing() or torch.jit.is_scripting():
        padding_mode = "reflect"
    pad_h = (patch_size[0] - img.shape[-2] % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - img.shape[-1] % patch_size[1]) % patch_size[1]
    return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode=padding_mode)

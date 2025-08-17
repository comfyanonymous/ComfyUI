import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.utils.data import DataLoader
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device

CKPT_CONFIGS = {
    "XVFInet_X4K1000FPS_exp1_latest.pt": {
        "module_scale_factor": 4,
        "S_trn": 3,
        "S_tst": 5
    },
    "XVFInet_Vimeo_exp1_latest.pt": {
        "module_scale_factor": 2,
        "S_trn": 1,
        "S_tst": 1
    }
}

class XVFI_Inference(nn.Module):
    def __init__(self, model_path, model_config) -> None:
        super(XVFI_Inference, self).__init__()
        from .xvfi_arch import XVFInet, weights_init
        model_config = model_config
        args = argparse.Namespace(
            gpu=get_torch_device(),
            nf=64,
            **model_config,
            img_ch=3,
        )
        self.model = XVFInet(args).apply(weights_init).to(get_torch_device())
        self.model.load_state_dict(torch.load(model_path, map_location=get_torch_device())["state_dict_Model"])

    def forward(self, I0, I1, timestep):
        #"Real" inference is called "test_custom" in the original repo
        #https://github.com/JihyongOh/XVFI/blob/main/utils.py#L434
        #https://github.com/JihyongOh/XVFI/blob/main/main.py#L336

        x = torch.stack([I0, I1], dim=0)
        x = einops.rearrange(x, "t b c h w -> b c t h w")
        return self.model(x, timestep, is_training=False)

MODEL_TYPE = pathlib.Path(__file__).parent.name

class XVFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPT_CONFIGS.keys()), ),
                "frames": ("IMAGE", ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "multipler": ("INT", {"default": 2, "min": 2, "max": 1000}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr, 
        frames: torch.Tensor, 
        batch_size: typing.SupportsInt = 1,
        multipler: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None
    ):
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        ckpt_config = CKPT_CONFIGS[ckpt_name]
        global model
        model = XVFI_Inference(model_path, ckpt_config)

        frames = preprocess_frames(frames)
        #https://github.com/JihyongOh/XVFI/blob/main/main.py#L314
        divide = 2 ** (ckpt_config["S_tst"]) * ckpt_config["module_scale_factor"] * 4
        B, C, H, W = frames.size()
        H_padding = (divide - H % divide) % divide
        W_padding = (divide - W % divide) % divide
        if H_padding != 0 or W_padding != 0:
            frames = F.pad(frames, (0, W_padding, 0, H_padding), "constant")
        
        frame_dict = {
            str(i): frames[i].unsqueeze(0) for i in range(frames.shape[0])
        }

        if optional_interpolation_states is None:
            interpolation_states = [True] * (frames.shape[0] - 1)
        else:
            interpolation_states = optional_interpolation_states

        enabled_former_idxs = [i for i, state in enumerate(interpolation_states) if state]
        former_idxs_loader = DataLoader(enabled_former_idxs, batch_size=batch_size)
        
        for former_idxs_batch in former_idxs_loader:
            for middle_i in range(1, multipler):
                _middle_frames = model(
                    frames[former_idxs_batch], 
                    frames[former_idxs_batch + 1], 
                    timestep=torch.tensor([middle_i/multipler]).repeat(len(former_idxs_batch)).unsqueeze(1).to(get_torch_device())
                )
                for i, former_idx in enumerate(former_idxs_batch):
                    frame_dict[f'{former_idx}.{middle_i}'] = _middle_frames[i].unsqueeze(0)
        
        out_frames = torch.cat([frame_dict[key] for key in sorted(frame_dict.keys())], dim=0)[:, :, :H, :W]
        return (postprocess_frames(out_frames), )
        

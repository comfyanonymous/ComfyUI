import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
import typing
import torch
import torch.nn as nn
from comfy.model_management import soft_empty_cache, get_torch_device

MODEL_TYPE = pathlib.Path(__file__).parent.name
MODEL_FILE_NAMES = {
    "ssl": "eisai_ssl.pt",
    "dtm": "eisai_dtm.pt",
    "raft": "eisai_anime_interp_full.ckpt"
}

class EISAI(nn.Module):
    def __init__(self, model_file_names) -> None:
        from .eisai_arch import SoftsplatLite, DTM, RAFT
        super(EISAI, self).__init__()
        self.raft = RAFT(load_file_from_github_release(MODEL_TYPE, model_file_names["raft"]))
        self.raft.to(get_torch_device()).eval()

        self.ssl = SoftsplatLite()
        self.ssl.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["ssl"])))
        self.ssl.to(get_torch_device()).eval()

        self.dtm = DTM()
        self.dtm.load_state_dict(torch.load(load_file_from_github_release(MODEL_TYPE, model_file_names["dtm"])))
        self.dtm.to(get_torch_device()).eval()
    
    def forward(self, img0, img1, t):
        with torch.no_grad():
            flow0, _ = self.raft(img0, img1)
            flow1, _ = self.raft(img1, img0)
            x = {
                "images": torch.stack([img0, img1], dim=1),
                "flows": torch.stack([flow0, flow1], dim=1),
            }
            out_ssl, _ = self.ssl(x, t=t, return_more=True)
            out_dtm, _ = self.dtm(x, out_ssl, _, return_more=False)
        return out_dtm[:, :3]

class EISAI_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["eisai"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        interpolation_model = EISAI(MODEL_FILE_NAMES)
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)
        
        def return_middle_frame(frame_0, frame_1, timestep, model):
            return model(frame_0, frame_1, t=timestep)
        
        scale = 1
        
        args = [interpolation_model, scale]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float32)
        )
        return (out,)

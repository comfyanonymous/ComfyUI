import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, generic_frame_loop, InterpolationStateList
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from comfy.model_management import get_torch_device


GLOBAL_MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPTS_PATH_CONFIG = {
    "GMFSS_fortuna_union": {
        "ifnet": ("rife", "rife46.pth"),
        "flownet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_flownet.pkl"),
        "metricnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_metric.pkl"),
        "feat_ext": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_feat.pkl"),
        "fusionnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_union_fusionnet.pkl")
    },
    "GMFSS_fortuna": {
        "flownet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_flownet.pkl"),
        "metricnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_metric.pkl"),
        "feat_ext": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_feat.pkl"),
        "fusionnet": (GLOBAL_MODEL_TYPE, "GMFSS_fortuna_fusionnet.pkl")
    }
}

class CommonModelInference(nn.Module):
    def __init__(self, model_type):
        super(CommonModelInference, self).__init__()
        from .GMFSS_Fortuna_arch import Model as GMFSS
        from .GMFSS_Fortuna_union_arch import Model as GMFSS_Union
        self.model = GMFSS_Union() if "union" in model_type else GMFSS()
        self.model.eval()
        self.model.device()
        _model_path_config = CKPTS_PATH_CONFIG[model_type]
        self.model.load_model({
            key: load_file_from_github_release(*_model_path_config[key])
            for key in _model_path_config
        })

    def forward(self, I0, I1, timestep, scale=1.0):
        n, c, h, w = I0.shape
        tmp = max(64, int(64 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)
        I0 = F.pad(I0, padding)
        I1 = F.pad(I1, padding)
        (
            flow01,
            flow10,
            metric0,
            metric1,
            feat11,
            feat12,
            feat13,
            feat21,
            feat22,
            feat23,
        ) = self.model.reuse(I0, I1, scale)

        output = self.model.inference(
            I0,
            I1,
            flow01,
            flow10,
            metric0,
            metric1,
            feat11,
            feat12,
            feat13,
            feat21,
            feat22,
            feat23,
            timestep,
        )
        return output[:, :, :h, :w]

class GMFSS_Fortuna_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (list(CKPTS_PATH_CONFIG.keys()), ),
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
        """
        Perform video frame interpolation using a given checkpoint model.
    
        Args:
            ckpt_name (str): The name of the checkpoint model to use.
            frames (torch.Tensor): A tensor containing input video frames.
            clear_cache_after_n_frames (int, optional): The number of frames to process before clearing CUDA cache
                to prevent memory overflow. Defaults to 10. Lower numbers are safer but mean more processing time.
                How high you should set it depends on how many input frames there are, input resolution (after upscaling),
                how many times you want to multiply them, and how long you're willing to wait for the process to complete.
            multiplier (int, optional): The multiplier for each input frame. 60 input frames * 2 = 120 output frames. Defaults to 2.
    
        Returns:
            tuple: A tuple containing the output interpolated frames.
    
        Note:
            This method interpolates frames in a video sequence using a specified checkpoint model. 
            It processes each frame sequentially, generating interpolated frames between them.
    
            To prevent memory overflow, it clears the CUDA cache after processing a specified number of frames.
        """
        
        interpolation_model = CommonModelInference(model_type=ckpt_name)
        interpolation_model.eval().to(get_torch_device())
        frames = preprocess_frames(frames)

        def return_middle_frame(frame_0, frame_1, timestep, model, scale):
            return model(frame_0, frame_1, timestep, scale)
        
        scale = 1
        
        args = [interpolation_model, scale]
        out = postprocess_frames(
            generic_frame_loop(type(self).__name__, frames, clear_cache_after_n_frames, multiplier, return_middle_frame, *args, 
                               interpolation_states=optional_interpolation_states, dtype=torch.float32)
        )
        return (out,)

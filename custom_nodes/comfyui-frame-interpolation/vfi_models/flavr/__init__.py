import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames, assert_batch_size
import pathlib
import warnings
from .flavr_arch import UNet_3D_3D, InputPadder
import gc

device = get_torch_device()
NBR_FRAME = 4

def build_flavr(model_path):
    sd = torch.load(model_path)['state_dict']
    sd = {k.partition("module.")[-1]:v for k,v in sd.items()}

    #Ref: Class UNet_3D_3D
    model = UNet_3D_3D("unet_18", n_inputs=NBR_FRAME, n_outputs=sd["outconv.1.weight"].shape[0] // 3, joinType="concat" , upmode="transpose")
    model.load_state_dict(sd)
    model.to(device).eval()
    del sd
    return model

MODEL_TYPE = pathlib.Path(__file__).parent.name
CKPT_NAMES = ["FLAVR_2x.pth", "FLAVR_4x.pth", "FLAVR_8x.pth"]

class FLAVR_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (CKPT_NAMES, ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 2}), #TODO: Implement recursively invoking interpolator for multi-frame interpolation
                "duplicate_first_last_frames": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }
    
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"        

    #Reference: https://github.com/danier97/ST-MFNet/blob/main/interpolate_yuv.py#L93
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        duplicate_first_last_frames: bool = False,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        if multiplier != 2:
            warnings.warn("Currently, FLAVR only supports 2x interpolation. The process will continue but please set multiplier=2 afterward")

        assert_batch_size(frames, batch_size=4, vfi_name="ST-MFNet")
        interpolation_states = optional_interpolation_states
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        model = build_flavr(model_path)
        frames = preprocess_frames(frames)
        padder = InputPadder(frames.shape, 16)
        frames = padder.pad(frames)

        number_of_frames_processed_since_last_cleared_cuda_cache = 0
        output_frames = []
        for frame_itr in range(len(frames) - 3):
            #Does skipping frame i+1 make sanse in this case?
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr) and interpolation_states.is_frame_skipped(frame_itr + 1):
                continue
            
            #Ensure that input frames are in fp32 - the same dtype as model
            frame0, frame1, frame2, frame3 = (
                frames[frame_itr:frame_itr+1].float(),
                frames[frame_itr+1:frame_itr+2].float(), 
                frames[frame_itr+2:frame_itr+3].float(), 
                frames[frame_itr+3:frame_itr+4].float()
            )
            new_frame = model([frame0.to(device), frame1.to(device), frame2.to(device), frame3.to(device)])[0].detach().cpu()
            number_of_frames_processed_since_last_cleared_cuda_cache += 2
            
            if frame_itr == 0:
                output_frames.append(frame0)
                if duplicate_first_last_frames:
                    output_frames.append(frame0) # repeat the first frame
                output_frames.append(frame1)
            output_frames.append(new_frame)
            output_frames.append(frame2)
            if frame_itr == len(frames) - 4:
                output_frames.append(frame3)
                if duplicate_first_last_frames:
                    output_frames.append(frame3) # repeat the last frame

            # Try to avoid a memory overflow by clearing cuda cache regularly
            if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
                print("Comfy-VFI: Clearing cache...", end = ' ')
                soft_empty_cache()
                number_of_frames_processed_since_last_cleared_cuda_cache = 0
                print("Done cache clearing")
            gc.collect()
        
        dtype = torch.float32
        output_frames = [frame.cpu().to(dtype=dtype) for frame in output_frames] #Ensure all frames are in cpu
        out = torch.cat(output_frames, dim=0)
        out = padder.unpad(out)
        # clear cache for courtesy
        print("Comfy-VFI: Final clearing cache...", end=' ')
        soft_empty_cache()
        print("Done cache clearing")
        return (postprocess_frames(out), )

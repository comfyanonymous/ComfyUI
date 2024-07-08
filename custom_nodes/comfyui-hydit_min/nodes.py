import os
import torch
import folder_paths
from .constant import T5_PATH
from .dit import load_checkpoint, load_vae
from .clip import CLIP


class DiffusersCheckpointLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
        
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                             "model_name": (folder_paths.get_filename_list("checkpoints"),),  }}

    RETURN_TYPES = ("MODEL", )

    FUNCTION = "load_checkpoint"

    CATEGORY = "HunYuan"

    def load_checkpoint(self, model_name):
        MODEL_PATH = folder_paths.get_full_path("checkpoints", model_name)
        out = load_checkpoint(MODEL_PATH)
        return out
    

class DiffusersVAELoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
        
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                             "model_name": (folder_paths.get_filename_list("vae"),),  }}

    RETURN_TYPES = ("VAE", )

    FUNCTION = "load_vae"

    CATEGORY = "HunYuan"

    def load_vae(self, model_name):
        MODEL_PATH = folder_paths.get_full_path("vae", model_name)
        out = load_vae(MODEL_PATH)
        return out
    


class DiffusersCLIPLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
        
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                             "text_encoder_path": (folder_paths.get_filename_list("clip"),),  
                             "t5_text_encoder_path": (os.listdir(T5_PATH),), }}

    RETURN_TYPES = ("CLIP", )

    FUNCTION = "load_clip"

    CATEGORY = "HunYuan"

    def load_clip(self, text_encoder_path, t5_text_encoder_path):
        CLIP_PATH = folder_paths.get_full_path("clip", text_encoder_path)
        t5_file = os.path.join(T5_PATH, t5_text_encoder_path)
        root = None
        out = CLIP(root, CLIP_PATH, t5_file)
        return (out,)
    



NODE_CLASS_MAPPINGS = {
    "DiffusersCheckpointLoader": DiffusersCheckpointLoader,
    "DiffusersVAELoader": DiffusersVAELoader,
    "DiffusersCLIPLoader": DiffusersCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersCheckpointLoader": "HunYuan Checkpoint Loader",
    "DiffusersVAELoader": "HunYuan VAE Loader", 
    "DiffusersCLIPLoader": "HunYuan CLIP Loader",  
}

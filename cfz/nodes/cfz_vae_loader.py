import torch
import folder_paths
from comfy import model_management
from nodes import VAELoader

class CFZVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (folder_paths.get_filename_list("vae"), ),
                "precision": (["fp32", "fp16", "bf16"], {"default": "fp32"}),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"
    TITLE = "CFZ VAE Loader"

    def load_vae(self, vae_name, precision):
        # Map precision to dtype
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16
        }
        target_dtype = dtype_map[precision]
        
        # Temporarily patch model_management functions
        original_should_use_bf16 = model_management.should_use_bf16
        original_should_use_fp16 = model_management.should_use_fp16
        
        def custom_should_use_bf16(*args, **kwargs):
            return precision == "bf16"
        
        def custom_should_use_fp16(*args, **kwargs):
            return precision == "fp16"
        
        # Apply patches
        model_management.should_use_bf16 = custom_should_use_bf16
        model_management.should_use_fp16 = custom_should_use_fp16
        
        try:
            # Load the VAE with patched precision functions
            vae_loader = VAELoader()
            vae = vae_loader.load_vae(vae_name)[0]
            print(f"CFZ VAE: Loaded with forced precision {precision}")
            return (vae,)
        finally:
            # Restore original functions
            model_management.should_use_bf16 = original_should_use_bf16
            model_management.should_use_fp16 = original_should_use_fp16

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CFZVAELoader": CFZVAELoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZVAELoader": "CFZ VAE Loader"
}
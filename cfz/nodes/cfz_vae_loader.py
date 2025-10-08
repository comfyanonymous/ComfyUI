import torch
import folder_paths
from comfy import model_management, model_base
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
        
        # Patch vae_dtype for loading
        original_vae_dtype = model_management.vae_dtype
        model_management.vae_dtype = lambda *args, **kwargs: target_dtype
        
        try:
            # Load VAE
            vae_loader = VAELoader()
            vae = vae_loader.load_vae(vae_name)[0]
        finally:
            model_management.vae_dtype = original_vae_dtype
        
        # Override VAE methods to maintain dtype
        if hasattr(vae, 'patcher'):
            # Override model_dtype
            vae.patcher.model_dtype = lambda: target_dtype
            
            # Wrap the decode method to ensure proper dtype handling
            if hasattr(vae, 'decode'):
                original_decode = vae.decode
                
                def forced_dtype_decode(samples_in):
                    # Ensure model is in correct dtype before decode
                    if hasattr(vae, 'first_stage_model'):
                        vae.first_stage_model = vae.first_stage_model.to(target_dtype)
                    
                    # Convert input to match model dtype
                    if isinstance(samples_in, torch.Tensor):
                        samples_in = samples_in.to(target_dtype)
                    
                    return original_decode(samples_in)
                
                vae.decode = forced_dtype_decode
            
            # Wrap encode similarly
            if hasattr(vae, 'encode'):
                original_encode = vae.encode
                
                def forced_dtype_encode(pixels):
                    # Ensure model is in correct dtype
                    if hasattr(vae, 'first_stage_model'):
                        vae.first_stage_model = vae.first_stage_model.to(target_dtype)
                    
                    # Convert input to match model dtype
                    if isinstance(pixels, torch.Tensor):
                        pixels = pixels.to(target_dtype)
                    
                    return original_encode(pixels)
                
                vae.encode = forced_dtype_encode
        
        print(f"CFZ VAE: Loaded with precision {precision} (dtype: {target_dtype})")
        return (vae,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CFZVAELoader": CFZVAELoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFZVAELoader": "CFZ VAE Loader"
}

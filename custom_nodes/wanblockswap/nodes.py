import comfy.model_management
import gc
import torch
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21
from tqdm import tqdm

#Based on https://github.com/kijai/ComfyUI-WanVideoWrapper
class WanVideoBlockSwap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {"default": 20, "min": 0, "max": 40, "step": 1, "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B model has 30 blocks"}),
                "offload_img_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload img_emb to offload_device"}),
                "offload_txt_emb": ("BOOLEAN", {"default": False, "tooltip": "Offload txt_emb to offload_device"}),
                "use_non_blocking": ("BOOLEAN", {"default": False, "tooltip": "Use non-blocking memory transfer for offloading, reserves more RAM but is faster"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "ComfyUI-wanBlockswap"
    FUNCTION = "set_callback"

    def set_callback(self, model: ModelPatcher, blocks_to_swap, offload_txt_emb, offload_img_emb, use_non_blocking):
        
        def swap_blocks(model: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
            base_model = model.model
            main_device=torch.device('cuda')
            if isinstance(base_model, WAN21):
                unet = base_model.diffusion_model
            for b, block in tqdm(enumerate(unet.blocks), total=len(unet.blocks), desc="Initializing block swap"):
                 
                if b > blocks_to_swap:
                    block.to(main_device)
                else:
                    block.to(model.offload_device)
                        
                if offload_txt_emb:
                    unet.text_embedding.to(model.offload_device, non_blocking=use_non_blocking)
                if offload_img_emb:
                    unet.img_emb.to(model.offload_device, non_blocking=use_non_blocking)

            comfy.model_management.soft_empty_cache()
            gc.collect()
        
        model = model.clone()
        model.add_callback(CallbacksMP.ON_LOAD,swap_blocks)

        return (model, )

NODE_CLASS_MAPPINGS = {
    "wanBlockSwap": WanVideoBlockSwap
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wanBlockSwap": "WanVideoBlockSwap"
}
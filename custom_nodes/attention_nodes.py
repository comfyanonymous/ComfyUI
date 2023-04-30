import os

import comfy
import folder_paths


class SaveAttention:
    @classmethod
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "attention": ("ATTENTION",),
                "filename": ("STRING", {"default": "attention.safetensor"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    RETURN_TYPES = ()
    FUNCTION = "save_attention"
    CATEGORY = "attention"
    OUTPUT_NODE = True

    # attention[a][b][c][d]
    # a: number of steps/sigma in this diffusion process
    # b: number of SpatialTransformer or AttentionBlocks used in the middle blocks of the latent diffusion model
    # c: number of transformer layers in the SpatialTransformer or AttentionBlocks
    # d: attn1, attn2
    def save_attention(self, attention, filename):
        comfy.utils.save_attention(attention, filename)
        return {"ui": {"message": "Saved attention to " + filename}}

class LoadAttention:
    @classmethod
    def __init__(self, event_dispatcher):
        self.event_dispatcher = event_dispatcher

    @classmethod
    def INPUT_TYPES(cls):
        output_dir = folder_paths.get_output_directory()
        return {
            "required": {
                "filename": (sorted(os.listdir(output_dir)), )},
        }

    RETURN_TYPES = ("ATTENTION",)
    FUNCTION = "load_attention"
    CATEGORY = "attention"

    def load_attention(self, filename):
        return (comfy.utils.load_attention(filename),)


NODE_CLASS_MAPPINGS = {
    "SaveAttention": SaveAttention,
    "LoadAttention": LoadAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAttention": "Save Attention",
    "LoadAttention": "Load Attention",
}

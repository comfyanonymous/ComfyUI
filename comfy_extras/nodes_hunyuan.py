import nodes
import torch
import comfy.model_management


class CLIPTextEncodeHunyuanDiT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "bert": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "mt5xl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, bert, mt5xl):
        tokens = clip.tokenize(bert)
        tokens["mt5xl"] = clip.tokenize(mt5xl)["mt5xl"]

        return (clip.encode_from_tokens_scheduled(tokens), )

class EmptyHunyuanLatentVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 848, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "length": ("INT", {"default": 25, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/video"

    def generate(self, width, height, length, batch_size=1):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        return ({"samples":latent}, )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeHunyuanDiT": CLIPTextEncodeHunyuanDiT,
    "EmptyHunyuanLatentVideo": EmptyHunyuanLatentVideo,
}

import folder_paths
import comfy.sd
import comfy.model_management
import nodes
import torch

class TripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("clip"), ), "clip_name2": (folder_paths.get_filename_list("clip"), ), "clip_name3": (folder_paths.get_filename_list("clip"), )
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, clip_name3):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)

class EmptySD3LatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/sd3"

    def generate(self, width, height, batch_size=1):
        latent = torch.ones([batch_size, 16, height // 8, width // 8], device=self.device) * 0.0609
        return ({"samples":latent}, )

class CLIPTextEncodeSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "empty_padding": (["none", "empty_prompt"], )
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, clip_l, clip_g, t5xxl, empty_padding):
        no_padding = empty_padding == "none"

        tokens = clip.tokenize(clip_g)
        if len(clip_g) == 0 and no_padding:
            tokens["g"] = []

        if len(clip_l) == 0 and no_padding:
            tokens["l"] = []
        else:
            tokens["l"] = clip.tokenize(clip_l)["l"]

        if len(t5xxl) == 0 and no_padding:
            tokens["t5xxl"] =  []
        else:
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )


class ControlNetApplySD3(nodes.ControlNetApplyAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}
    CATEGORY = "conditioning/controlnet"

NODE_CLASS_MAPPINGS = {
    "TripleCLIPLoader": TripleCLIPLoader,
    "EmptySD3LatentImage": EmptySD3LatentImage,
    "CLIPTextEncodeSD3": CLIPTextEncodeSD3,
    "ControlNetApplySD3": ControlNetApplySD3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Sampling
    "ControlNetApplySD3": "ControlNetApply SD3 and HunyuanDiT",
}

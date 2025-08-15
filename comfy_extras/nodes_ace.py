import torch
import comfy.model_management
import node_helpers

class TextEncodeAceStepAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "tags": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "lyrics": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "lyrics_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, tags, lyrics, lyrics_strength):
        tokens = clip.tokenize(tags, lyrics=lyrics)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        conditioning = node_helpers.conditioning_set_values(conditioning, {"lyrics_strength": lyrics_strength})
        return (conditioning, )


class EmptyAceStepLatentAudio:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/audio"

    def generate(self, seconds, batch_size):
        length = int(seconds * 44100 / 512 / 8)
        latent = torch.zeros([batch_size, 8, 16, length], device=self.device)
        return ({"samples": latent, "type": "audio"}, )


NODE_CLASS_MAPPINGS = {
    "TextEncodeAceStepAudio": TextEncodeAceStepAudio,
    "EmptyAceStepLatentAudio": EmptyAceStepLatentAudio,
}

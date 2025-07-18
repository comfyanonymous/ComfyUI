from nodes import MAX_RESOLUTION

class CLIPTextEncodePixArtAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            # "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True}), "clip": ("CLIP", ),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "advanced/conditioning"
    DESCRIPTION = "Encodes text and sets the resolution conditioning for PixArt Alpha. Does not apply to PixArt Sigma."

    def encode(self, clip, width, height, text):
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"width": width, "height": height}),)

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodePixArtAlpha": CLIPTextEncodePixArtAlpha,
}

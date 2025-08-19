

class TextEncodeQwenImageEdit:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {"image": ("IMAGE", ),}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, prompt, image=None):
        if image is None:
            images = []
        else:
            images = [image]
        tokens = clip.tokenize(prompt, images=images)
        return (clip.encode_from_tokens_scheduled(tokens), )


NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEdit": TextEncodeQwenImageEdit,
}

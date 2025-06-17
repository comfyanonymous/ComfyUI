

class CLIPTextEncodeControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP", ), "conditioning": ("CONDITIONING", ), "text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "_for_testing/conditioning"

    def encode(self, clip, conditioning, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        c = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]['cross_attn_controlnet'] = cond
            n[1]['pooled_output_controlnet'] = pooled
            c.append(n)
        return (c, )

class T5TokenizerOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "min_padding": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "min_length": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    CATEGORY = "_for_testing/conditioning"
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "set_options"

    def set_options(self, clip, min_padding, min_length):
        clip = clip.clone()
        for t5_type in ["t5xxl", "pile_t5xl", "t5base", "mt5xl", "umt5xxl"]:
            clip.set_tokenizer_option("{}_min_padding".format(t5_type), min_padding)
            clip.set_tokenizer_option("{}_min_length".format(t5_type), min_length)

        return (clip, )

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeControlnet": CLIPTextEncodeControlnet,
    "T5TokenizerOptions": T5TokenizerOptions,
}

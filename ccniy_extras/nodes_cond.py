

class CLIPTextEncodeControlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP", ), "conditioning": ("CONDITIONING", ), "text": ("STRING", {"multiline": True})}}
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

NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeControlnet": CLIPTextEncodeControlnet
}

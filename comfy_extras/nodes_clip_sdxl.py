import math
from nodes import MAX_RESOLUTION
from comfy.dynamic_prompt import process_dynamic_prompt, is_dynamic_prompt_changed #, validate_dynamic_prompt


class CLIPTextEncodeSDXLRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True}),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        text = process_dynamic_prompt(text)
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[
            cond,
            {
                "pooled_output": pooled, "aesthetic_score": ascore,
                "width": width, "height": height
            }
        ]],)

    @classmethod
    def IS_CHANGED(s, clip, ascore, width, height, text):
        return is_dynamic_prompt_changed(text)

    # @classmethod
    # def VALIDATE_INPUTS(cls, text):
    #     return validate_dynamic_prompt(text)


class CLIPTextEncodeSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP", ),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        text_g = process_dynamic_prompt(text_g)
        text_l = process_dynamic_prompt(text_l)
        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[
            cond,
            {
                "pooled_output": pooled,
                "width": width, "height": height,
                "crop_w": crop_w, "crop_h": crop_h,
                "target_width": target_width, "target_height": target_height
            }
        ]], )

    @classmethod
    def IS_CHANGED(cls, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
        text_g_changed = is_dynamic_prompt_changed(text_g)
        text_l_changed = is_dynamic_prompt_changed(text_l)
        # If either is NaN, return NaN, otherwise return the sum
        if math.isnan(text_g_changed) or math.isnan(text_l_changed):
            return float("nan")
        else:
            # Log to console
            return text_g_changed + text_l_changed

    # @classmethod
    # def VALIDATE_INPUTS(cls, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l):
    #     # These return `True`, or an error message if the input is invalid
    #     text_g_valid = validate_dynamic_prompt(text_g)
    #     text_l_valid = validate_dynamic_prompt(text_l)
    #     # If either is not `True`, return the error message, otherwise return `True`
    #     if text_g_valid is not True:
    #         return text_g_valid
    #     elif text_l_valid is not True:
    #         return text_l_valid


NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeSDXLRefiner": CLIPTextEncodeSDXLRefiner,
    "CLIPTextEncodeSDXL": CLIPTextEncodeSDXL,
}

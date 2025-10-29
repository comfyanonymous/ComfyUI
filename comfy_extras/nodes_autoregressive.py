from comfy.autoregressive_sampling import auto_sample
from comfy.comfy_types import IO

class AutoRegressiveGeneration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for generation."}),
                "input_ids": ("TOKENS", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for controling the generation."}),
                "max_new_length": ("INT", {"default": 1024, "min": 1, "max": 10_000, "tooltip": "The max length for generation."}),
                "min_new_length": ("INT", {"default": 1, "min": 1, "max": 10_000, "tooltip": "The min length for generation."}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 30_000, "tooltip": "Takes the top k of the most probable tokens."}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Percentage of tokens to leave after generation (top most probable tokens)."}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50, "step": 0.01, "tooltip": "Temperature controls randomess by decreasing or increasing the probability of lesser likely tokens. Higher Temperature -> More Randomness"}),
                "do_sample": ("BOOLEAN", {"default": False, "tooltip": "Add randomness in decoding the tokens."}),
            }
        }

    RETURN_TYPES = ("TOKENS",)
    FUNCTION = "generate"

    CATEGORY = "sampling"

    # for cuda graphs
    _cached_autoregressive_sampler = None

    def generate(self, model, input_ids, seed, max_new_length, min_new_length, top_k, top_p, temperature, do_sample):
        return (auto_sample(self, model, input_ids, max_new_length, min_new_length, top_k, top_p, temperature, do_sample, seed = seed),)

class DecodeTokens:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP, {"tooltip": "The model used for generation."}),
                "tokens": ("TOKENS", ),}
        }

    FUNCTION = "decode"
    CATEGORY = "conditioning"
    RETURN_TYPES = ("TEXT", "AUDIO")

    def decode(self, clip, tokens):
        clip.load_model()
        if hasattr(clip.cond_stage_model, "decode_tokens"): # for special tokenizers
            return clip.cond_stage_model.decode_tokens(tokens)
        else:
            return (clip.tokenizer.decode(tokens, skip_special_tokens=True), None)

NODE_CLASS_MAPPINGS = {
    "DecodeTokens": DecodeTokens,
    "AutoRegressiveGeneration": AutoRegressiveGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRegressiveGeneration": "Autoregressive Generation",
    "DecodeTokens": "Decode Tokens",
}

from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
import comfy.text_encoders.llama


class Gemma2BTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=2304, embedding_key='gemma2_2b', tokenizer_class=SPieceTokenizer, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_args={"add_bos": True, "add_eos": False})

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class LuminaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma2_2b", tokenizer=Gemma2BTokenizer)


class Gemma2_2BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options={}):
        llama_scaled_fp8 = model_options.get("llama_scaled_fp8", None)
        if llama_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = llama_scaled_fp8

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Gemma2_2B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class LuminaModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="gemma2_2b", clip_model=Gemma2_2BModel, model_options=model_options)


def te(dtype_llama=None, llama_scaled_fp8=None):
    class LuminaTEModel_(LuminaModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "llama_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["llama_scaled_fp8"] = llama_scaled_fp8
                if dtype_llama is not None:
                    dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return LuminaTEModel_

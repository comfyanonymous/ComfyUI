from .llama import Gemma2_2B, Gemma3_4B
from .spiece_tokenizer import SPieceTokenizer
from .. import sd1_clip


class Gemma2BTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=2304, embedding_directory=None, embedding_key='gemma2_2b', tokenizer_class=SPieceTokenizer, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_args={"add_bos": True, "add_eos": False}, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class Gemma3_4BTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=2560, embedding_directory=None, embedding_key='gemma3_4b', tokenizer_class=SPieceTokenizer, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_args={"add_bos": True, "add_eos": False}, disable_weights=True, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class LuminaTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma2_2b", tokenizer=Gemma2BTokenizer)


class NTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="gemma3_4b", tokenizer=Gemma3_4BTokenizer)


class Gemma2_2BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = {}
        textmodel_json_config = textmodel_json_config or {}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=Gemma2_2B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class Gemma3_4BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None, attention_mask=True, model_options={}, textmodel_json_config=None):
        if textmodel_json_config is None:
            textmodel_json_config = {}
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 2, "pad": 0}, layer_norm_hidden_state=False, model_class=Gemma3_4B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)

class LuminaModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, name="gemma2_2b", clip_model=Gemma2_2BModel):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, name=name, clip_model=clip_model, model_options=model_options)


def te(dtype_llama=None, llama_quantization_metadata=None, model_type="gemma2_2b"):
    model = None
    if model_type == "gemma2_2b":
        model = Gemma2_2BModel
    elif model_type == "gemma3_4b":
        model = Gemma3_4BModel

    class LuminaTEModel_(LuminaModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, name=model_type, model_options=model_options, clip_model=model)

    return LuminaTEModel_

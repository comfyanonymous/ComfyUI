from transformers import T5TokenizerFast

from .t5 import T5
from .. import sd1_clip
from ..component_model import files
from ..component_model.files import get_path_as_dict


class T5XXLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = {}
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "t5_old_config_xxl.json", package=__package__)
        t5xxl_quantization_metadata = model_options.get("t5xxl_quantization_metadata", None)
        if t5xxl_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = t5xxl_quantization_metadata

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, zero_out_masked=attention_mask, model_options=model_options)


class CosmosT5XXL(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, name="t5xxl", clip_model=T5XXLModel, model_options=model_options)



class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer_path = files.get_package_as_path("comfy.text_encoders.t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=1024, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=512, tokenizer_data=tokenizer_data)


class CosmosT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5xxl", tokenizer=T5XXLTokenizer)



def te(dtype_t5=None, t5_quantization_metadata=None):
    class CosmosTEModel_(CosmosT5XXL):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if t5_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["t5xxl_quantization_metadata"] = t5_quantization_metadata
            if dtype is None:
                dtype = dtype_t5
            super().__init__(device=device, dtype=dtype, model_options=model_options)

    return CosmosTEModel_

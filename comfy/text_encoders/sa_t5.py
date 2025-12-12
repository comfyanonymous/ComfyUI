from ..transformers_compat import T5TokenizerFast

from .t5 import T5
from .. import sd1_clip
from ..component_model import files


class T5BaseModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = dict()
        textmodel_json_config = files.get_path_as_dict(textmodel_json_config, "t5_config_base.json", package=__package__)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5, model_options=model_options, enable_attention_masks=True, zero_out_masked=True)


class T5BaseTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, *args, **kwargs):
        tokenizer_path = files.get_package_as_path("comfy.text_encoders.t5_tokenizer")
        tokenizer_data = kwargs.pop("tokenizer_data", {})
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=768, embedding_key='t5base', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=128, tokenizer_data=tokenizer_data)


class SAT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = dict()
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5base", tokenizer=T5BaseTokenizer)


class SAT5Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, model_options=model_options, name="t5base", clip_model=T5BaseModel, **kwargs)

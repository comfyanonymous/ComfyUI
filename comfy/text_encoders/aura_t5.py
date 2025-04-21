from importlib import resources

from .. import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
from ..text_encoders import t5
from ..component_model.files import get_path_as_dict

class PT5XlModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = dict()
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "t5_pile_config_xl.json", package=__package__)
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 2, "pad": 1}, model_class=t5.T5, enable_attention_masks=True, zero_out_masked=True)


class PT5XlTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, **kwargs):
        tokenizer_path = resources.files("comfy.text_encoders.t5_pile_tokenizer") / "tokenizer.model"
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=2048, embedding_key='pile_t5xl', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, pad_token=1, tokenizer_data=tokenizer_data)


class AuraT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = dict()
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="pile_t5xl", tokenizer=PT5XlTokenizer)


class AuraT5Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None, **kwargs):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, model_options=model_options, name="pile_t5xl", clip_model=PT5XlModel, **kwargs)

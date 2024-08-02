from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
import comfy.text_encoders.t5
import os

class PT5XlModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_pile_config_xl.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 2, "pad": 1}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, zero_out_masked=True)

class PT5XlTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_pile_tokenizer"), "tokenizer.model")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=2048, embedding_key='pile_t5xl', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, pad_token=1)

class AuraT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="pile_t5xl", tokenizer=PT5XlTokenizer)

class AuraT5Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, **kwargs):
        super().__init__(device=device, dtype=dtype, name="pile_t5xl", clip_model=PT5XlModel, **kwargs)

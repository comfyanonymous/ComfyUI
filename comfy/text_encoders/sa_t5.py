from comfy import sd1_clip
from transformers import T5TokenizerFast
import comfy.text_encoders.t5
import os

class T5BaseModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_base.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, model_options=model_options, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, zero_out_masked=True)

class T5BaseTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=768, embedding_key='t5base', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=128, tokenizer_data=tokenizer_data)

class SAT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5base", tokenizer=T5BaseTokenizer)

class SAT5Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, name="t5base", clip_model=T5BaseModel, **kwargs)

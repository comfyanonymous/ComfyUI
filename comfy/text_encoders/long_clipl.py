from comfy import sd1_clip
import os

from comfy.component_model.files import get_path_as_dict


class LongClipTokenizer_(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(max_length=248, embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

class LongClipModel_(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, textmodel_json_config=None):
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "long_clipl.json", package=__package__)
        super().__init__(device=device, textmodel_json_config=textmodel_json_config, return_projected_pooled=False, dtype=dtype, model_options=model_options)

class LongClipTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, tokenizer=LongClipTokenizer_)

class LongClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, clip_model=LongClipModel_, **kwargs)

from comfy import sd1_clip
import os

class LongClipTokenizer_(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(max_length=248, embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

class LongClipModel_(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "long_clipl.json")
        super().__init__(device=device, textmodel_json_config=textmodel_json_config, return_projected_pooled=False, dtype=dtype, model_options=model_options)

class LongClipTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, tokenizer=LongClipTokenizer_)

class LongClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, clip_model=LongClipModel_, **kwargs)

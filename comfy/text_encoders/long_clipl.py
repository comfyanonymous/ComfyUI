from comfy import sd1_clip
import os

class LongClipTokenizer_(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(max_length=248, embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

class LongClipModel_(sd1_clip.SDClipModel):
    def __init__(self, *args, **kwargs):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "long_clipl.json")
        super().__init__(*args, textmodel_json_config=textmodel_json_config, **kwargs)

class LongClipTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, tokenizer=LongClipTokenizer_)

class LongClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__(device=device, dtype=dtype, model_options=model_options, clip_model=LongClipModel_, **kwargs)

def model_options_long_clip(sd, tokenizer_data, model_options):
    w = sd.get("clip_l.text_model.embeddings.position_embedding.weight", None)
    if w is None:
        w = sd.get("text_model.embeddings.position_embedding.weight", None)
    if w is not None and w.shape[0] == 248:
        tokenizer_data = tokenizer_data.copy()
        model_options = model_options.copy()
        tokenizer_data["clip_l_tokenizer_class"] = LongClipTokenizer_
        model_options["clip_l_class"] = LongClipModel_
    return tokenizer_data, model_options

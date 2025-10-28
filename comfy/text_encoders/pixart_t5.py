import os

from comfy import sd1_clip
import comfy.text_encoders.t5
import comfy.text_encoders.sd3_clip
from comfy.sd1_clip import gen_empty_tokens

from transformers import T5TokenizerFast

class T5XXLModel(comfy.text_encoders.sd3_clip.T5XXLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def gen_empty_tokens(self, special_tokens, *args, **kwargs):
        # PixArt expects the negative to be all pad tokens
        special_tokens = special_tokens.copy()
        special_tokens.pop("end")
        return gen_empty_tokens(special_tokens, *args, **kwargs)

class PixArtT5XXL(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="t5xxl", clip_model=T5XXLModel, model_options=model_options)

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_data=tokenizer_data) # no padding

class PixArtTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5xxl", tokenizer=T5XXLTokenizer)

def pixart_te(dtype_t5=None, t5xxl_scaled_fp8=None):
    class PixArtTEModel_(PixArtT5XXL):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            if dtype is None:
                dtype = dtype_t5
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return PixArtTEModel_

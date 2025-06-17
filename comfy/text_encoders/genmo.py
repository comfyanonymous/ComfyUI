from transformers import T5TokenizerFast

from .. import sd1_clip
from . import sd3_clip
from ..component_model import files


class T5XXLModel(sd3_clip.T5XXLModel):
    def __init__(self, **kwargs):
        kwargs["attention_mask"] = True
        super().__init__(**kwargs)


class MochiT5XXL(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options=None):
        if model_options is None:
            model_options = {}
        super().__init__(device=device, dtype=dtype, name="t5xxl", clip_model=T5XXLModel, model_options=model_options)


class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer_path = files.get_package_as_path("comfy.text_encoders.t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, tokenizer_data=tokenizer_data)


class MochiT5Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, clip_name="t5xxl", tokenizer=T5XXLTokenizer)
        if tokenizer_data is None:
            tokenizer_data = {}


def mochi_te(dtype_t5=None, t5xxl_scaled_fp8=None):
    class MochiTEModel_(MochiT5XXL):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            if dtype is None:
                dtype = dtype_t5
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MochiTEModel_

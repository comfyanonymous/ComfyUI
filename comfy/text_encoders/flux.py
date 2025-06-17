import copy

import torch
from transformers import T5TokenizerFast

from .sd3_clip import T5XXLModel
from .. import sd1_clip, model_management
from ..component_model import files


class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = dict()
        tokenizer_path = files.get_package_as_path("comfy.text_encoders.t5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, tokenizer_data=tokenizer_data)


class FluxTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = dict()
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids, **kwargs),
            "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)
        }
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

    def clone(self):
        return copy.copy(self)


class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None, model_options=None):
        super().__init__()
        if model_options is None:
            model_options = {}
        dtype_t5 = model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = {dtype, dtype_t5}

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_t5 = token_weight_pairs["t5xxl"]

        t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pairs_t5)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return t5_out, l_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)


def flux_clip(dtype_t5=None, t5xxl_scaled_fp8=None):
    class FluxClipModel_(FluxClipModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if t5xxl_scaled_fp8 is not None and "t5xxl_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["t5xxl_scaled_fp8"] = t5xxl_scaled_fp8
            super().__init__(dtype_t5=dtype_t5, device=device, dtype=dtype, model_options=model_options)

    return FluxClipModel_

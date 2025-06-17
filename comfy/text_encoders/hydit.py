import copy

import torch
from transformers import BertTokenizer

from .bert import BertModel
from .spiece_tokenizer import SPieceTokenizer
from .t5 import T5
from .. import sd1_clip
from ..component_model.files import get_path_as_dict, get_package_as_path


class HyditBertModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = dict()
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "hydit_clip.json", package=__package__)
        model_options = {**model_options, "model_name": "hydit_clip"}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 101, "end": 102, "pad": 0}, model_class=BertModel, enable_attention_masks=True, return_attention_masks=True, model_options=model_options)

class HyditBertTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, **kwargs):
        tokenizer_path = get_package_as_path(f"{__package__}.hydit_clip_tokenizer")
        tokenizer_data = kwargs.pop("tokenizer_data", {})
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=1024, embedding_key='chinese_roberta', tokenizer_class=BertTokenizer, pad_to_max_length=False, max_length=512, min_length=77, tokenizer_data=tokenizer_data)


class MT5XLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = dict()
        textmodel_json_config = get_path_as_dict(textmodel_json_config, "mt5_config_xl.json", package=__package__)
        model_options = {**model_options, "model_name": "mt5xl"}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=T5, enable_attention_masks=True, return_attention_masks=True, model_options=model_options)

class MT5XLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_data=None, **kwargs):
        if tokenizer_data is None:
            tokenizer_data = dict()
        if not "spiece_model" in tokenizer_data:
            raise FileNotFoundError("expected a checkpoint that contains the mt5 tokenizer's sentencepiece model")
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=2048, embedding_key='mt5xl', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}


class HyditTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None or "mt5xl.spiece_model" not in tokenizer_data:
            raise FileNotFoundError("expected mt5xl tokenizer data in the checkpoint")
        mt5_tokenizer_data = tokenizer_data.get("mt5xl.spiece_model", None)
        self.hydit_clip = HyditBertTokenizer(embedding_directory=embedding_directory)
        self.mt5xl = MT5XLTokenizer(tokenizer_data={**tokenizer_data, "spiece_model": mt5_tokenizer_data}, embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = {}
        out["hydit_clip"] = self.hydit_clip.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["mt5xl"] = self.mt5xl.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.hydit_clip.untokenize(token_weight_pair)

    def state_dict(self):
        return {"mt5xl.spiece_model": self.mt5xl.state_dict()["spiece_model"]}

    def clone(self):
        return copy.copy(self)


class HyditModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options=None):
        super().__init__()
        if model_options is None:
            model_options = dict()
        self.hydit_clip = HyditBertModel(dtype=dtype, model_options=model_options)
        self.mt5xl = MT5XLModel(dtype=dtype, model_options=model_options)

        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def encode_token_weights(self, token_weight_pairs):
        hydit_out = self.hydit_clip.encode_token_weights(token_weight_pairs["hydit_clip"])
        mt5_out = self.mt5xl.encode_token_weights(token_weight_pairs["mt5xl"])
        return hydit_out[0], hydit_out[1], {"attention_mask": hydit_out[2]["attention_mask"], "conditioning_mt5xl": mt5_out[0], "attention_mask_mt5xl": mt5_out[2]["attention_mask"]}

    def load_sd(self, sd):
        if "bert.encoder.layer.0.attention.self.query.weight" in sd:
            return self.hydit_clip.load_sd(sd)
        else:
            return self.mt5xl.load_sd(sd)

    def set_clip_options(self, options):
        self.hydit_clip.set_clip_options(options)
        self.mt5xl.set_clip_options(options)

    def reset_clip_options(self):
        self.hydit_clip.reset_clip_options()
        self.mt5xl.reset_clip_options()

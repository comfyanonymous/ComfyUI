from comfy import sd1_clip
from transformers import BertTokenizer
from .spiece_tokenizer import SPieceTokenizer
from .bert import BertModel
import comfy.text_encoders.t5
import os
import torch

class HyditBertModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hydit_clip.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"start": 101, "end": 102, "pad": 0}, model_class=BertModel, enable_attention_masks=True, return_attention_masks=True)

class HyditBertTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "hydit_clip_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=1024, embedding_key='chinese_roberta', tokenizer_class=BertTokenizer, pad_to_max_length=False, max_length=512, min_length=77)


class MT5XLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mt5_config_xl.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, return_attention_masks=True)

class MT5XLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        #tokenizer_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "mt5_tokenizer"), "spiece.model")
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=2048, embedding_key='mt5xl', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=256)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}

class HyditTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        mt5_tokenizer_data = tokenizer_data.get("mt5xl.spiece_model", None)
        self.hydit_clip = HyditBertTokenizer(embedding_directory=embedding_directory)
        self.mt5xl = MT5XLTokenizer(tokenizer_data={"spiece_model": mt5_tokenizer_data}, embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["hydit_clip"] = self.hydit_clip.tokenize_with_weights(text, return_word_ids)
        out["mt5xl"] = self.mt5xl.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.hydit_clip.untokenize(token_weight_pair)

    def state_dict(self):
        return {"mt5xl.spiece_model": self.mt5xl.state_dict()["spiece_model"]}

class HyditModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        self.hydit_clip = HyditBertModel(dtype=dtype)
        self.mt5xl = MT5XLModel(dtype=dtype)

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

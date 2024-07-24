from comfy import sd1_clip
from comfy import sdxl_clip
from transformers import T5TokenizerFast
import comfy.text_encoders.t5
import torch
import os
import comfy.model_management
import logging

class T5XXLModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_config_xxl.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5)

class T5XXLTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=4096, embedding_key='t5xxl', tokenizer_class=T5TokenizerFast, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=77)


class SD3Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory)
        self.clip_g = sdxl_clip.SDXLClipGTokenizer(embedding_directory=embedding_directory)
        self.t5xxl = T5XXLTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        out["t5xxl"] = self.t5xxl.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

class SD3ClipModel(torch.nn.Module):
    def __init__(self, clip_l=True, clip_g=True, t5=True, dtype_t5=None, device="cpu", dtype=None):
        super().__init__()
        self.dtypes = set()
        if clip_l:
            self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False, return_projected_pooled=False)
            self.dtypes.add(dtype)
        else:
            self.clip_l = None

        if clip_g:
            self.clip_g = sdxl_clip.SDXLClipG(device=device, dtype=dtype)
            self.dtypes.add(dtype)
        else:
            self.clip_g = None

        if t5:
            if dtype_t5 is None:
                dtype_t5 = dtype
            elif comfy.model_management.dtype_size(dtype_t5) > comfy.model_management.dtype_size(dtype):
                dtype_t5 = dtype

            if not comfy.model_management.supports_cast(device, dtype_t5):
                dtype_t5 = dtype

            self.t5xxl = T5XXLModel(device=device, dtype=dtype_t5)
            self.dtypes.add(dtype_t5)
        else:
            self.t5xxl = None

        logging.debug("Created SD3 text encoder with: clip_l {}, clip_g {}, t5xxl {}:{}".format(clip_l, clip_g, t5, dtype_t5))

    def set_clip_options(self, options):
        if self.clip_l is not None:
            self.clip_l.set_clip_options(options)
        if self.clip_g is not None:
            self.clip_g.set_clip_options(options)
        if self.t5xxl is not None:
            self.t5xxl.set_clip_options(options)

    def reset_clip_options(self):
        if self.clip_l is not None:
            self.clip_l.reset_clip_options()
        if self.clip_g is not None:
            self.clip_g.reset_clip_options()
        if self.t5xxl is not None:
            self.t5xxl.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pars_t5 = token_weight_pairs["t5xxl"]
        lg_out = None
        pooled = None
        out = None

        if len(token_weight_pairs_g) > 0 or len(token_weight_pairs_l) > 0:
            if self.clip_l is not None:
                lg_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
            else:
                l_pooled = torch.zeros((1, 768), device=comfy.model_management.intermediate_device())

            if self.clip_g is not None:
                g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
                if lg_out is not None:
                    lg_out = torch.cat([lg_out, g_out], dim=-1)
                else:
                    lg_out = torch.nn.functional.pad(g_out, (768, 0))
            else:
                g_out = None
                g_pooled = torch.zeros((1, 1280), device=comfy.model_management.intermediate_device())

            if lg_out is not None:
                lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
                out = lg_out
            pooled = torch.cat((l_pooled, g_pooled), dim=-1)

        if self.t5xxl is not None:
            t5_out, t5_pooled = self.t5xxl.encode_token_weights(token_weight_pars_t5)
            if lg_out is not None:
                out = torch.cat([lg_out, t5_out], dim=-2)
            else:
                out = t5_out

        if out is None:
            out = torch.zeros((1, 77, 4096), device=comfy.model_management.intermediate_device())

        if pooled is None:
            pooled = torch.zeros((1, 768 + 1280), device=comfy.model_management.intermediate_device())

        return out, pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        elif "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.t5xxl.load_sd(sd)

def sd3_clip(clip_l=True, clip_g=True, t5=True, dtype_t5=None):
    class SD3ClipModel_(SD3ClipModel):
        def __init__(self, device="cpu", dtype=None):
            super().__init__(clip_l=clip_l, clip_g=clip_g, t5=t5, dtype_t5=dtype_t5, device=device, dtype=dtype)
    return SD3ClipModel_

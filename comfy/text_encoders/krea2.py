"""Krea 2 (K2) text encoder: Qwen3-VL-4B language model, 12-layer tap.

K2 conditions on a stack of hidden states from 12 layers of Qwen3-VL-4B
(reference taps ``hidden_states[2,5,8,...,35]``), kept as a ``(B, 12, seq, 2560)`` tensor and
consumed by the DiT's internal ``txtfusion`` adapter. Comfy carries conditioning as a 3D tensor,
so the 12-layer stack is flattened to ``(B, seq, 12*2560)`` here and unpacked inside the model.
"""

import os
import numbers

import torch
from transformers import Qwen2Tokenizer

import comfy.text_encoders.llama
from comfy import sd1_clip

# tap k == hidden_states[k] (no offset).
KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
QWEN3VL_4B_CONFIG = {"rope_theta": 5000000.0, "final_norm": False, "lm_head": False}

# Identical system template to Qwen-Image; Krea2 strips the system+user-opening prefix.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class Qwen3VL4BTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory,
                         embedding_size=2560, embedding_key='qwen3vl_4b', tokenizer_class=Qwen2Tokenizer,
                         has_start_token=False, has_end_token=False, pad_to_max_length=False,
                         max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Krea2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="qwen3vl_4b", tokenizer=Qwen3VL4BTokenizer)
        self.llama_template = KREA2_TEMPLATE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, **kwargs):
        if text.startswith('<|im_start|>'):
            llama_text = text
        elif llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)
        return super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)


class Qwen3VL4BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=KREA2_TAP_LAYERS, layer_idx=None,
                         textmodel_json_config=dict(QWEN3VL_4B_CONFIG),
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=comfy.text_encoders.llama.Qwen3_4B,
                         enable_attention_masks=attention_mask, return_attention_masks=attention_mask,
                         model_options=model_options)


class Krea2TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_4b", clip_model=Qwen3VL4BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs, template_end=-1):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)  # out: (B, 12, seq, 2560)
        tok_pairs = token_weight_pairs["qwen3vl_4b"][0]

        # Strip the system + user-opening prefix
        count_im_start = 0
        if template_end == -1:
            for i, v in enumerate(tok_pairs):
                elem = v[0]
                if not torch.is_tensor(elem) and isinstance(elem, numbers.Integral):
                    if elem == 151644 and count_im_start < 2:
                        template_end = i
                        count_im_start += 1
            if out.shape[2] > (template_end + 3):
                if tok_pairs[template_end + 1][0] == 872:      # "user"
                    if tok_pairs[template_end + 2][0] == 198:   # "\n"
                        template_end += 3

        out = out[:, :, template_end:]

        b, n, seq, h = out.shape
        # Flatten the 12-layer axis into the feature dim: (B, seq, 12*2560). Unpacked in the model.
        out = out.permute(0, 2, 1, 3).reshape(b, seq, n * h)

        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, template_end:]
            if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
                extra.pop("attention_mask")

        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class Krea2TEModel_(Krea2TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Krea2TEModel_

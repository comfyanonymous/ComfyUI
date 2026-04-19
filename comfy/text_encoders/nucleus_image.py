from transformers import Qwen2Tokenizer
import comfy.text_encoders.llama
from comfy import sd1_clip
import os
import torch


class NucleusImageQwen3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(
            tokenizer_path,
            pad_with_end=False,
            embedding_directory=embedding_directory,
            embedding_size=4096,
            embedding_key='qwen3_8b',
            tokenizer_class=Qwen2Tokenizer,
            has_start_token=False,
            has_end_token=False,
            pad_to_max_length=False,
            max_length=99999999,
            min_length=1,
            pad_token=151643,
            tokenizer_data=tokenizer_data,
        )


class NucleusImageTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen3_8b = NucleusImageQwen3Tokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = "<|im_start|>system\nYou are an image generation assistant. Follow the user's prompt literally. Pay careful attention to spatial layout: objects described as on the left must appear on the left, on the right on the right. Match exact object counts and assign colors to the correct objects.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        llama_text = self.llama_template.format(text)
        tokens = self.qwen3_8b.tokenize_with_weights(llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs)
        return {"qwen3_8b": tokens}

    def untokenize(self, token_weight_pair):
        return self.qwen3_8b.untokenize(token_weight_pair)

    def state_dict(self):
        return {}

    def decode(self, token_ids, **kwargs):
        return self.qwen3_8b.decode(token_ids, **kwargs)


class NucleusImageQwen3VLText(comfy.text_encoders.llama.Qwen3_8B):
    def __init__(self, config_dict, dtype, device, operations):
        config_dict = dict(config_dict)
        config_dict.setdefault("max_position_embeddings", 262144)
        config_dict.setdefault("rope_theta", 5000000.0)
        super().__init__(config_dict, dtype, device, operations)


class NucleusImageQwen3_8BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-8, dtype=None, attention_mask=True, model_options={}):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,
            textmodel_json_config={},
            dtype=dtype,
            special_tokens={"pad": 151643},
            layer_norm_hidden_state=False,
            model_class=NucleusImageQwen3VLText,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


class NucleusImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device,
            dtype=dtype,
            name="qwen3_8b",
            clip_model=NucleusImageQwen3_8BModel,
            model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class NucleusImageTEModel_(NucleusImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return NucleusImageTEModel_

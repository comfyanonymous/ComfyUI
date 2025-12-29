import re

from ..transformers_compat import ByT5Tokenizer

from .llama import Qwen25_7BVLI
from .qwen_image import QwenImageTokenizer, QwenImageTEModel
from .t5 import T5
from .. import sd1_clip
from ..component_model import files


class ByT5SmallTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        tokenizer_path = files.get_package_as_path(f"{__package__}.byt5_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=None, pad_with_end=False, embedding_size=1472, embedding_key='byt5_small', tokenizer_class=ByT5Tokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_data=tokenizer_data)


class HunyuanImageTokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        # self.llama_template_images = "{}"
        self.byt5 = ByT5SmallTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text: str, return_word_ids=False, **kwargs):
        out = super().tokenize_with_weights(text, return_word_ids, **kwargs)

        # ByT5 processing for HunyuanImage
        text_prompt_texts = []
        pattern_quote_double = r'\"(.*?)\"'
        pattern_quote_chinese_single = r'‘(.*?)’'
        pattern_quote_chinese_double = r'“(.*?)”'

        matches_quote_double = re.findall(pattern_quote_double, text)
        matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, text)
        matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, text)

        text_prompt_texts.extend(matches_quote_double)
        text_prompt_texts.extend(matches_quote_chinese_single)
        text_prompt_texts.extend(matches_quote_chinese_double)

        if len(text_prompt_texts) > 0:
            out['byt5'] = self.byt5.tokenize_with_weights(''.join(map(lambda a: 'Text "{}". '.format(a), text_prompt_texts)), return_word_ids, **kwargs)
        return out


class Qwen25_7BVLIModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-3, dtype=None, attention_mask=True, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = {}
        llama_quantization_metadata = model_options.get("llama_quantization_metadata", None)
        if llama_quantization_metadata is not None:
            model_options = model_options.copy()
            model_options["quantization_metadata"] = llama_quantization_metadata
        if textmodel_json_config is None:
            textmodel_json_config = {}
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class ByT5SmallModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options=None, textmodel_json_config=None):
        if model_options is None:
            model_options = {}
        textmodel_json_config = files.get_path_as_dict(textmodel_json_config, "byt5_config_small_glyph.json", package=__package__)

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, model_options=model_options, special_tokens={"end": 1, "pad": 0}, model_class=T5, enable_attention_masks=True, zero_out_masked=True)


class HunyuanImageTEModel(QwenImageTEModel):
    def __init__(self, byt5=True, device="cpu", dtype=None, model_options=None):
        if model_options is None:
            model_options = {}
        super(QwenImageTEModel, self).__init__(device=device, dtype=dtype, name="qwen25_7b", clip_model=Qwen25_7BVLIModel, model_options=model_options)
        if byt5:
            self.byt5_small = ByT5SmallModel(device=device, dtype=dtype, model_options=model_options)
        else:
            self.byt5_small = None

    def encode_token_weights(self, token_weight_pairs):
        tok_pairs = token_weight_pairs["qwen25_7b"][0]
        template_end = -1
        if tok_pairs[0][0] == 27:
            if len(tok_pairs) > 36:  # refiner prompt uses a fixed 36 template_end
                template_end = 36

        cond, p, extra = super().encode_token_weights(token_weight_pairs, template_end=template_end)
        if self.byt5_small is not None and "byt5" in token_weight_pairs:
            out = self.byt5_small.encode_token_weights(token_weight_pairs["byt5"])
            extra["conditioning_byt5small"] = out[0]
        return cond, p, extra

    def set_clip_options(self, options):
        super().set_clip_options(options)
        if self.byt5_small is not None:
            self.byt5_small.set_clip_options(options)

    def reset_clip_options(self):
        super().reset_clip_options()
        if self.byt5_small is not None:
            self.byt5_small.reset_clip_options()

    def load_sd(self, sd):
        if "encoder.block.0.layer.0.SelfAttention.o.weight" in sd:
            return self.byt5_small.load_sd(sd)
        else:
            return super().load_sd(sd)


def te(byt5=True, dtype_llama=None, llama_quantization_metadata=None):
    class QwenImageTEModel_(HunyuanImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options=None):
            if model_options is None:
                model_options = {}
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["llama_quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(byt5=byt5, device=device, dtype=dtype, model_options=model_options)

    return QwenImageTEModel_

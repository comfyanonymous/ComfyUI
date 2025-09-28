from comfy import sd1_clip
import comfy.text_encoders.llama
from .qwen_image import QwenImageTokenizer, QwenImageTEModel
from transformers import ByT5Tokenizer
import os
import re

class ByT5SmallTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "byt5_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=1472, embedding_key='byt5_small', tokenizer_class=ByT5Tokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, tokenizer_data=tokenizer_data)

class HunyuanImageTokenizer(QwenImageTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
        # self.llama_template_images = "{}"
        self.byt5 = ByT5SmallTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
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
    def __init__(self, device="cpu", layer="hidden", layer_idx=-3, dtype=None, attention_mask=True, model_options={}):
        llama_scaled_fp8 = model_options.get("qwen_scaled_fp8", None)
        if llama_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = llama_scaled_fp8
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen25_7BVLI, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class ByT5SmallModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "byt5_config_small_glyph.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, model_options=model_options, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, zero_out_masked=True)


class HunyuanImageTEModel(QwenImageTEModel):
    def __init__(self, byt5=True, device="cpu", dtype=None, model_options={}):
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

def te(byt5=True, dtype_llama=None, llama_scaled_fp8=None):
    class QwenImageTEModel_(HunyuanImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["qwen_scaled_fp8"] = llama_scaled_fp8
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(byt5=byt5, device=device, dtype=dtype, model_options=model_options)
    return QwenImageTEModel_

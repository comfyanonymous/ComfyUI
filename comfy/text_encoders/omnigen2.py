from transformers import Qwen2Tokenizer
from comfy import sd1_clip
import comfy.text_encoders.llama
import os


class Qwen25_3BTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(tokenizer_path, pad_with_end=False, embedding_size=2048, embedding_key='qwen25_3b', tokenizer_class=Qwen2Tokenizer, has_start_token=False, has_end_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=151643, tokenizer_data=tokenizer_data)


class Omnigen2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen25_3b", tokenizer=Qwen25_3BTokenizer)
        self.llama_template = '<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n'

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None,**kwargs):
        if llama_template is None:
            llama_text = self.llama_template.format(text)
        else:
            llama_text = llama_template.format(text)
        return super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids, **kwargs)

class Qwen25_3BModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen25_3B, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class Omnigen2Model(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen25_3b", clip_model=Qwen25_3BModel, model_options=model_options)


def te(dtype_llama=None, llama_scaled_fp8=None):
    class Omnigen2TEModel_(Omnigen2Model):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["scaled_fp8"] = llama_scaled_fp8
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return Omnigen2TEModel_

from comfy import sd1_clip
import comfy.model_management
import comfy.text_encoders.llama
from transformers import LlamaTokenizerFast
import torch
import os


def llama_detect(state_dict, prefix=""):
    out = {}
    t5_key = "{}model.norm.weight".format(prefix)
    if t5_key in state_dict:
        out["dtype_llama"] = state_dict[t5_key].dtype

    scaled_fp8_key = "{}scaled_fp8".format(prefix)
    if scaled_fp8_key in state_dict:
        out["llama_scaled_fp8"] = state_dict[scaled_fp8_key].dtype

    return out


class LLAMA3Tokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}, min_length=256):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "llama_tokenizer")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, pad_with_end=False, embedding_size=4096, embedding_key='llama', tokenizer_class=LlamaTokenizerFast, has_start_token=True, has_end_token=False, pad_to_max_length=False, max_length=99999999, pad_token=128258, end_token=128009, min_length=min_length)

class LLAMAModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="hidden", layer_idx=-3, dtype=None, attention_mask=True, model_options={}):
        llama_scaled_fp8 = model_options.get("llama_scaled_fp8", None)
        if llama_scaled_fp8 is not None:
            model_options = model_options.copy()
            model_options["scaled_fp8"] = llama_scaled_fp8

        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={}, dtype=dtype, special_tokens={"start": 128000, "pad": 128258}, layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Llama2, enable_attention_masks=attention_mask, return_attention_masks=attention_mask, model_options=model_options)


class HunyuanVideoTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        clip_l_tokenizer_class = tokenizer_data.get("clip_l_tokenizer_class", sd1_clip.SDTokenizer)
        self.clip_l = clip_l_tokenizer_class(embedding_directory=embedding_directory)
        self.llama_template = """<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: 1. The main content and theme of the video.2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.4. background environment, light, style and atmosphere.5. camera angles, movements, and transitions used in the video:<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"""  # 95 tokens
        self.llama = LLAMA3Tokenizer(embedding_directory=embedding_directory, min_length=1)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)

        llama_text = "{}{}".format(self.llama_template, text)
        out["llama"] = self.llama.tokenize_with_weights(llama_text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_l.untokenize(token_weight_pair)

    def state_dict(self):
        return {}


class HunyuanVideoClipModel(torch.nn.Module):
    def __init__(self, dtype_llama=None, device="cpu", dtype=None, model_options={}):
        super().__init__()
        dtype_llama = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, device)
        clip_l_class = model_options.get("clip_l_class", sd1_clip.SDClipModel)
        self.clip_l = clip_l_class(device=device, dtype=dtype, return_projected_pooled=False, model_options=model_options)
        self.llama = LLAMAModel(device=device, dtype=dtype_llama, model_options=model_options)
        self.dtypes = set([dtype, dtype_llama])

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.llama.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_l.reset_clip_options()
        self.llama.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_l = token_weight_pairs["l"]
        token_weight_pairs_llama = token_weight_pairs["llama"]

        llama_out, llama_pooled, llama_extra_out = self.llama.encode_token_weights(token_weight_pairs_llama)

        template_end = 0
        for i, v in enumerate(token_weight_pairs_llama[0]):
            if v[0] == 128007:  # <|end_header_id|>
                template_end = i

        if llama_out.shape[1] > (template_end + 2):
            if token_weight_pairs_llama[0][template_end + 1][0] == 271:
                template_end += 2
        llama_out = llama_out[:, template_end:]
        llama_extra_out["attention_mask"] = llama_extra_out["attention_mask"][:, template_end:]
        if llama_extra_out["attention_mask"].sum() == torch.numel(llama_extra_out["attention_mask"]):
            llama_extra_out.pop("attention_mask")  # attention mask is useless if no masked elements

        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return llama_out, l_pooled, llama_extra_out

    def load_sd(self, sd):
        if "text_model.encoder.layers.1.mlp.fc1.weight" in sd:
            return self.clip_l.load_sd(sd)
        else:
            return self.llama.load_sd(sd)


def hunyuan_video_clip(dtype_llama=None, llama_scaled_fp8=None):
    class HunyuanVideoClipModel_(HunyuanVideoClipModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_scaled_fp8 is not None and "llama_scaled_fp8" not in model_options:
                model_options = model_options.copy()
                model_options["llama_scaled_fp8"] = llama_scaled_fp8
            super().__init__(dtype_llama=dtype_llama, device=device, dtype=dtype, model_options=model_options)
    return HunyuanVideoClipModel_

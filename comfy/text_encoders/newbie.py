import torch

import comfy.model_management
import comfy.text_encoders.jina_clip_2
import comfy.text_encoders.lumina2

class NewBieTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.gemma = comfy.text_encoders.lumina2.Gemma3_4BTokenizer(embedding_directory=embedding_directory, tokenizer_data={"spiece_model": tokenizer_data["gemma_spiece_model"]})
        self.jina = comfy.text_encoders.jina_clip_2.JinaClip2Tokenizer(embedding_directory=embedding_directory, tokenizer_data={"spiece_model": tokenizer_data["jina_spiece_model"]})

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["gemma"] = self.gemma.tokenize_with_weights(text, return_word_ids, **kwargs)
        out["jina"] = self.jina.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        raise NotImplementedError

    def state_dict(self):
        return {}

class NewBieClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__()
        self.gemma = comfy.text_encoders.lumina2.Gemma3_4BModel(device=device, dtype=dtype, model_options=model_options)
        self.jina = comfy.text_encoders.jina_clip_2.JinaClip2TextModel(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = {dtype}

    def set_clip_options(self, options):
        self.gemma.set_clip_options(options)
        self.jina.set_clip_options(options)

    def reset_clip_options(self):
        self.gemma.reset_clip_options()
        self.jina.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_gemma = token_weight_pairs["gemma"]
        token_weight_pairs_jina = token_weight_pairs["jina"]

        gemma_out, gemma_pooled, gemma_extra = self.gemma.encode_token_weights(token_weight_pairs_gemma)
        jina_out, jina_pooled, jina_extra = self.jina.encode_token_weights(token_weight_pairs_jina)

        return gemma_out, jina_pooled, gemma_extra

    def load_sd(self, sd):
        if "model.layers.0.self_attn.q_norm.weight" in sd:
            return self.gemma.load_sd(sd)
        else:
            return self.jina.load_sd(sd)

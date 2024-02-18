from comfy import sd1_clip
import torch
import os

class SDXLClipG(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=77, freeze=True, layer="penultimate", layer_idx=None, dtype=None):
        if layer == "penultimate":
            layer="hidden"
            layer_idx=-2

        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config_bigg.json")
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype,
                         special_tokens={"start": 49406, "end": 49407, "pad": 0}, layer_norm_hidden_state=False)

    def load_sd(self, sd):
        return super().load_sd(sd)

class SDXLClipGTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g')


class SDXLTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory)
        self.clip_g = SDXLClipGTokenizer(embedding_directory=embedding_directory)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

class SDXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None):
        super().__init__()
        self.clip_l = sd1_clip.SDClipModel(layer="hidden", layer_idx=-2, device=device, dtype=dtype, layer_norm_hidden_state=False)
        self.clip_g = SDXLClipG(device=device, dtype=dtype)

    def clip_layer(self, layer_idx):
        self.clip_l.clip_layer(layer_idx)
        self.clip_g.clip_layer(layer_idx)

    def reset_clip_layer(self):
        self.clip_g.reset_clip_layer()
        self.clip_l.reset_clip_layer()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)

class SDXLRefinerClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None):
        super().__init__(device=device, dtype=dtype, clip_name="g", clip_model=SDXLClipG)


class StableCascadeClipGTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, tokenizer_path=None, embedding_directory=None):
        super().__init__(tokenizer_path, pad_with_end=True, embedding_directory=embedding_directory, embedding_size=1280, embedding_key='clip_g')

class StableCascadeTokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None):
        super().__init__(embedding_directory=embedding_directory, clip_name="g", tokenizer=StableCascadeClipGTokenizer)

class StableCascadeClipG(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", max_length=77, freeze=True, layer="hidden", layer_idx=-1, dtype=None):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "clip_config_bigg.json")
        super().__init__(device=device, freeze=freeze, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype,
                         special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=False, enable_attention_masks=True)

    def load_sd(self, sd):
        return super().load_sd(sd)

class StableCascadeClipModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None):
        super().__init__(device=device, dtype=dtype, clip_name="g", clip_model=StableCascadeClipG)

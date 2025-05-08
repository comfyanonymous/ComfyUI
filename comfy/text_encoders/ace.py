from comfy import sd1_clip
from .spiece_tokenizer import SPieceTokenizer
import comfy.text_encoders.t5
import os
import re
import torch
import logging

from tokenizers import Tokenizer
from .ace_text_cleaners import multilingual_cleaners, japanese_to_romaji

SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285,
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293,
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}

structure_pattern = re.compile(r"\[.*?\]")

DEFAULT_VOCAB_FILE = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ace_lyrics_tokenizer"), "vocab.json")


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=DEFAULT_VOCAB_FILE):
        self.tokenizer = None
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt, lang):
        txt = multilingual_cleaners(txt, lang)
        return txt

    def encode(self, txt, lang='en'):
        # lang = lang.split("-")[0]  # remove the region
        # self.check_input_length(txt, lang)
        txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def get_lang(self, line):
        if line.startswith("[") and line[3:4] == ']':
            lang = line[1:3].lower()
            if lang in SUPPORT_LANGUAGES:
                return lang, line[4:]
        return "en", line

    def __call__(self, string):
        lines = string.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang, line = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            try:
                line_out = japanese_to_romaji(line)
                if line_out != line:
                    lang = "ja"
                line = line_out
            except:
                pass

            try:
                if structure_pattern.match(line):
                    token_idx = self.encode(line, "en")
                else:
                    token_idx = self.encode(line, lang)
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                logging.warning("tokenize error {} for line {} major_language {}".format(e, line, lang))
        return {"input_ids": lyric_token_idx}

    @staticmethod
    def from_pretrained(path, **kwargs):
        return VoiceBpeTokenizer(path, **kwargs)

    def get_vocab(self):
        return {}


class UMT5BaseModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "umt5_config_base.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=comfy.text_encoders.t5.T5, enable_attention_masks=True, zero_out_masked=False, model_options=model_options)

class UMT5BaseTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = tokenizer_data.get("spiece_model", None)
        super().__init__(tokenizer, pad_with_end=False, embedding_size=768, embedding_key='umt5base', tokenizer_class=SPieceTokenizer, has_start_token=False, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=0, tokenizer_data=tokenizer_data)

    def state_dict(self):
        return {"spiece_model": self.tokenizer.serialize_model()}

class LyricsTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), "ace_lyrics_tokenizer"), "vocab.json")
        super().__init__(tokenizer, pad_with_end=False, embedding_size=1024, embedding_key='lyrics', tokenizer_class=VoiceBpeTokenizer, has_start_token=True, pad_to_max_length=False, max_length=99999999, min_length=1, pad_token=2, has_end_token=False, tokenizer_data=tokenizer_data)

class AceT5Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.voicebpe = LyricsTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)
        self.umt5base = UMT5BaseTokenizer(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data)

    def tokenize_with_weights(self, text:str, return_word_ids=False, **kwargs):
        out = {}
        out["lyrics"] = self.voicebpe.tokenize_with_weights(kwargs.get("lyrics", ""), return_word_ids, **kwargs)
        out["umt5base"] = self.umt5base.tokenize_with_weights(text, return_word_ids, **kwargs)
        return out

    def untokenize(self, token_weight_pair):
        return self.umt5base.untokenize(token_weight_pair)

    def state_dict(self):
        return self.umt5base.state_dict()

class AceT5Model(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}, **kwargs):
        super().__init__()
        self.umt5base = UMT5BaseModel(device=device, dtype=dtype, model_options=model_options)
        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

    def set_clip_options(self, options):
        self.umt5base.set_clip_options(options)

    def reset_clip_options(self):
        self.umt5base.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_umt5base = token_weight_pairs["umt5base"]
        token_weight_pairs_lyrics = token_weight_pairs["lyrics"]

        t5_out, t5_pooled = self.umt5base.encode_token_weights(token_weight_pairs_umt5base)

        lyrics_embeds = torch.tensor(list(map(lambda a: a[0], token_weight_pairs_lyrics[0]))).unsqueeze(0)
        return t5_out, None, {"conditioning_lyrics": lyrics_embeds}

    def load_sd(self, sd):
        return self.umt5base.load_sd(sd)

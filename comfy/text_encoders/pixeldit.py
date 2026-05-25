import torch

from comfy import sd1_clip
from .lumina2 import Gemma2BTokenizer, LuminaModel
import comfy.text_encoders.llama


class PixelDiTGemma2_2BModel(sd1_clip.SDClipModel):
    """Gemma-2-2b-it text encoder for PixelDiT.

    Uses the FINAL hidden state (layer='last')
    """
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, attention_mask=True, model_options={}):
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx,
            textmodel_json_config={}, dtype=dtype,
            special_tokens={"start": 2, "pad": 0},
            layer_norm_hidden_state=False,
            model_class=comfy.text_encoders.llama.Gemma2_2B,
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            model_options=model_options,
        )


_PIXELDIT_CHI_PROMPT = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
    "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly without "
    "overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, "
    "sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
    "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
    "passing by towering glass skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:\n"
    "User Prompt: "
)

_PIXELDIT_MAX_LENGTH = 300
_PIXELDIT_CHI_PROMPT_DETECT_PREFIX = 'Given a user prompt, generate an "Enhanced prompt"'


def _build_padded_tokens(combined_text: str, spiece_tokenizer, pad_id: int, chi_token_count: int):
    # Right-pad to chi_token_count + 300 - 2 (matches upstream's max_length_all).
    max_length_all = chi_token_count + _PIXELDIT_MAX_LENGTH - 2
    ids = spiece_tokenizer(combined_text)["input_ids"]
    if len(ids) > max_length_all:
        ids = ids[:max_length_all]
    elif len(ids) < max_length_all:
        ids = ids + [pad_id] * (max_length_all - len(ids))
    return ids


class PixelDiTGemma2Tokenizer(sd1_clip.SD1Tokenizer):
    """Gemma-2-2b-it tokenizer that prepends PixelDiT's chi_prompt.

    Empty text -> BOS + pad to 300. Text already starting with the chi_prompt
    preamble is tokenized verbatim (override mirrors QwenImageTokenizer's
    `<|im_start|>` detection). Else chi_prompt is prepended.
    """
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        if tokenizer_data is None:
            tokenizer_data = {}
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         name="gemma2_2b", tokenizer=Gemma2BTokenizer)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        spiece_tokenizer = self.gemma2_2b.tokenizer
        pad_id = self.gemma2_2b.pad_token

        if not (isinstance(text, str) and text.strip()):
            ids = spiece_tokenizer("")["input_ids"]
            ids = ids + [pad_id] * (_PIXELDIT_MAX_LENGTH - len(ids))
            return {"gemma2_2b": [[(t, 1.0) for t in ids]]}

        chi_token_count = len(spiece_tokenizer(_PIXELDIT_CHI_PROMPT)["input_ids"])
        combined = text if text.startswith(_PIXELDIT_CHI_PROMPT_DETECT_PREFIX) else _PIXELDIT_CHI_PROMPT + text
        ids = _build_padded_tokens(combined, spiece_tokenizer, pad_id, chi_token_count)
        return {"gemma2_2b": [[(t, 1.0) for t in ids]]}

    def untokenize(self, token_weight_pair):
        return self.gemma2_2b.untokenize(token_weight_pair)

    def state_dict(self):
        return self.gemma2_2b.state_dict()


class PixelDiTGemma2TE(LuminaModel):
    """Text encoder wrapper for PixelDiT.

    Overrides `encode_token_weights` to perform PixelDiT's `select_index` step:
    encode the full padded sequence (up to ~chi_prompt_tokens + 298), then
    return `[BOS_emb] + last_299_embs` as the 300-position conditioning that
    matches the diffusion model's learned `y_pos_embedding` positions.
    """
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="gemma2_2b",
                         clip_model=PixelDiTGemma2_2BModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        result = super().encode_token_weights(token_weight_pairs)
        cond, pooled = result[0], result[1]
        extra = result[2] if len(result) > 2 else None
        L = cond.shape[1]
        if L > _PIXELDIT_MAX_LENGTH:
            head = cond[:, :1]
            tail = cond[:, -(_PIXELDIT_MAX_LENGTH - 1):]
            cond = torch.cat([head, tail], dim=1)
            if extra is not None and "attention_mask" in extra:
                am = extra["attention_mask"]
                if am.dim() == 1:
                    am = am.unsqueeze(0)
                if am.shape[-1] == L:
                    head_m = am[..., :1]
                    tail_m = am[..., -(_PIXELDIT_MAX_LENGTH - 1):]
                    extra = {**extra, "attention_mask": torch.cat([head_m, tail_m], dim=-1)}
        if extra is not None:
            return cond, pooled, extra
        return cond, pooled

"""JoyImageEdit text encoder: Qwen3-VL multimodal stack feeding the JoyImageEdit DiT.

Plugs the generic Qwen3-VL stack from `comfy.text_encoders.qwen3_vl` into the
`SDClipModel` / `SD1ClipModel` contract, adding only the JoyImage-specific
templates, drop_idx, tokenizer wrapper, and `te()` factory.
"""

import os

from transformers import Qwen2Tokenizer

from comfy import sd1_clip
from comfy.text_encoders.qwen3_vl import Qwen3VLBase

# Prompt templates for the text-only and image-conditioned modes. The
# image-conditioned template wraps the user text with a single
# `<|vision_start|><|image_pad|><|vision_end|>` block; this encoder supports one
# user turn per call.
JOYIMAGE_TEMPLATE_TEXT = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

JOYIMAGE_TEMPLATE_IMAGE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
)

# Tokens 0..33 of either formatted template (system prompt + leading
# `<|im_start|>` of the user block) are stripped from the encoded output by
# JoyImageTEModel.encode_token_weights so that the kept tail begins at the
# `user` token (prefix[:34] decodes to the system block ending at the leading
# `<|im_start|>` of the user turn).
JOYIMAGE_DROP_IDX = 34

# Special-token ids from the JoyImage Qwen3-VL tokenizer (vocab is shared
# with Qwen2.5 / Qwen3 — vocab_size 151936).
IMAGE_PAD_TOKEN = 151655
PAD_TOKEN = 151643


class Qwen3VL8B_JoyImage(Qwen3VLBase):
    """Bind `Qwen3VLBase` to the JoyImage-specific config dict shape.

    The JoyImage checkpoint follows the standard Qwen3-VL 8B text dims
    (4096 / 36L / 32H / 8 kv / silu / qkv_bias=False, q/k_norm=gemma3) plus
    interleaved 3D MRoPE with rope_dims=[24, 20, 20] and rope_theta=5e6 —
    all defaults of `Qwen3VLConfig`. Vision tower uses the defaults of
    `Qwen3VLVisionConfig` (1152/4304/4096/16H, 27 blocks, patch_size=16,
    deepstack_visual_indexes=[8, 16, 24]).
    """

    def __init__(self, config_dict, dtype, device, operations):
        super().__init__(config_dict, dtype, device, operations)


class _JoyImageBaseTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        # Reuse the existing qwen25_tokenizer artefacts shipped with ComfyUI;
        # the JoyImage tokenizer is the same vocab/merges as Qwen2.5/Qwen3
        # (vocab_size 151936). The image-pad / vision-start / vision-end
        # special tokens are present in that vocab.
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "qwen25_tokenizer")
        super().__init__(
            tokenizer_path, pad_with_end=False, embedding_directory=embedding_directory,
            embedding_size=4096, embedding_key="qwen3vl_8b", tokenizer_class=Qwen2Tokenizer,
            has_start_token=False, has_end_token=False, pad_to_max_length=False,
            max_length=99999999, min_length=1, pad_token=PAD_TOKEN, tokenizer_data=tokenizer_data,
        )


class JoyImageTokenizer(sd1_clip.SD1Tokenizer):
    """JoyImageEdit tokenizer.

    ``tokenize_with_weights(text, images=[...])`` selects the image-conditioned
    template when one or more image tensors are passed, otherwise the text-only
    template. Each ``<|image_pad|>`` token in the formatted prompt is replaced
    with an embedding marker so `SDClipModel.process_tokens` routes the image
    through `Qwen3VL8B_JoyImage.preprocess_embed`; ``drop_idx=34`` leading
    template tokens are stripped downstream by
    `JoyImageTEModel.encode_token_weights`.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
            name="qwen3vl_8b", tokenizer=_JoyImageBaseTokenizer,
        )
        self.llama_template = JOYIMAGE_TEMPLATE_TEXT
        self.llama_template_images = JOYIMAGE_TEMPLATE_IMAGE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None,
                              images=[], **kwargs):
        if text.startswith("<|im_start|>"):
            llama_text = text
        elif llama_template is not None:
            llama_text = llama_template.format(text)
        elif len(images) > 0:
            llama_text = self.llama_template_images.format(text)
        else:
            llama_text = self.llama_template.format(text)

        tokens = super().tokenize_with_weights(
            llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs,
        )

        key_name = next(iter(tokens))
        embed_count = 0
        qwen_tokens = tokens[key_name]
        for r in qwen_tokens:
            for i in range(len(r)):
                if r[i][0] == IMAGE_PAD_TOKEN:
                    if len(images) > embed_count:
                        r[i] = ({"type": "image", "data": images[embed_count],
                                 "original_type": "image"},) + r[i][1:]
                        embed_count += 1
        if embed_count != len(images):
            raise ValueError(
                f"JoyImageTokenizer: prompt had {embed_count} <|image_pad|> placeholders "
                f"but {len(images)} image(s) were supplied. Either pre-format the prompt "
                f"with `<|vision_start|><|image_pad|><|vision_end|>` per image or pass an "
                f"image-free prompt."
            )
        return tokens


class _JoyImageClipModel(sd1_clip.SDClipModel):
    """Qwen3-VL multimodal encoder wrapper.

    ``layer="hidden", layer_idx=-1`` + ``layer_norm_hidden_state=False`` is the
    pre-norm hook: `SDClipModel.forward` calls the transformer with
    ``intermediate_output=-1`` (resolved to ``num_layers - 1``) and
    ``final_layer_norm_intermediate=False``, so the captured intermediate is
    the **post-layer-N, pre-final-norm** output of the last decoder layer —
    NOT the post-norm ``last_hidden_state``. **Do NOT 'simplify' to
    layer="last" / final_layer_norm_intermediate=True**: that returns the
    post-norm output, which differs by ~10x in scale (std approx 21 vs 2)
    and produces broken DiT outputs.
    """

    def __init__(self, device="cpu", layer="hidden", layer_idx=-1, dtype=None,
                 attention_mask=True, model_options={}):
        super().__init__(
            device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config={},
            dtype=dtype, special_tokens={"pad": PAD_TOKEN}, layer_norm_hidden_state=False,
            model_class=Qwen3VL8B_JoyImage, enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask, model_options=model_options,
        )


class JoyImageTEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(
            device=device, dtype=dtype, name="qwen3vl_8b",
            clip_model=_JoyImageClipModel, model_options=model_options,
        )

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        # Strip the JOYIMAGE_DROP_IDX-token system-prompt prefix from both the
        # embedding sequence and the attention mask.
        if out.shape[1] <= JOYIMAGE_DROP_IDX:
            raise ValueError(
                f"JoyImageTEModel: encoded sequence length {out.shape[1]} is shorter "
                f"than drop_idx={JOYIMAGE_DROP_IDX}; the prompt did not include the "
                f"template prefix."
            )
        out = out[:, JOYIMAGE_DROP_IDX:]
        if "attention_mask" in extra:
            extra["attention_mask"] = extra["attention_mask"][:, JOYIMAGE_DROP_IDX:]
        return out, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class JoyImageTEModel_(JoyImageTEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            if dtype_llama is not None:
                dtype = dtype_llama
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return JoyImageTEModel_

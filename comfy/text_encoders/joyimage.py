"""JoyImageEdit text encoder: a stock Qwen3-VL-8B multimodal stack feeding the
JoyImageEdit DiT, built on `comfy.text_encoders.qwen3vl` with the
JoyImage-specific prompt templates, system-prompt strip, image preprocessing,
and conditioning-path multimodal handling.
"""

import torch

from comfy import sd1_clip
import comfy.text_encoders.qwen_vl
from comfy.text_encoders.qwen3vl import Qwen3VL, Qwen3VLTokenizer

# Prompt templates for the text-only and image-conditioned modes. The image-conditioned template
# wraps the user text with one `<|vision_start|><|image_pad|><|vision_end|>` block per reference
# image (no separator between blocks); `{vision}` is filled with the N concatenated blocks and
# `{prompt}` with the user text.
JOYIMAGE_TEMPLATE_TEXT = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

JOYIMAGE_TEMPLATE_IMAGE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{vision}{prompt}<|im_end|>\n<|im_start|>assistant\n"
)

# A single vision block; N copies are concatenated to condition on N reference images.
JOYIMAGE_VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# Number of leading template tokens (system prompt + the user block's opening
# `<|im_start|>`) stripped from the encoded output by
# JoyImageTEModel.encode_token_weights, so the kept sequence begins at the
# `user` token.
JOYIMAGE_DROP_IDX = 34

# Special-token ids (vocab shared with Qwen2.5 / Qwen3, vocab_size 151936).
IMAGE_PAD_TOKEN = 151655
PAD_TOKEN = 151643


class Qwen3VL8B_JoyImage(Qwen3VL):
    """JoyImage Qwen3-VL-8B encoder.

    Stock `qwen3vl_8b` config (text dims 4096 / 36L / 32H / 8 kv; interleaved
    3D MRoPE rope_dims=[24,20,20], rope_theta=5e6; vision 1152/4304, depth 27,
    patch_size 16, deepstack_visual_indexes=[8,16,24]).
    """

    model_type = "qwen3vl_8b"

    def preprocess_embed(self, embed, device):
        # Run the vision tower with JoyImage's bicubic+clamp preprocessing and
        # return ``(merged, {"grid", "deepstack"})``.
        if embed["type"] == "image":
            image, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
                embed["data"], min_pixels=65536, max_pixels=16777216, patch_size=16,
                image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
                interpolation="bicubic", clamp=True,
            )
            merged, deepstack = self.visual(image.to(device, dtype=torch.float32), grid)
            return merged, {"grid": grid, "deepstack": deepstack}
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=()):
        # The conditioning path must build the 3D MRoPE position ids for the
        # image-token block and inject the deepstack visual features.
        # `build_image_inputs` returns the kwargs the decoder expects:
        # (position_ids, visual_pos_masks, deepstack).
        if embeds is not None:
            position_ids, visual_pos_masks, deepstack = self.build_image_inputs(embeds, embeds_info)
        else:
            position_ids, visual_pos_masks, deepstack = None, None, None
        return self.model(
            x,
            attention_mask=attention_mask,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype,
            position_ids=position_ids,
            deepstack_embeds=deepstack,
            visual_pos_masks=visual_pos_masks,
        )


class JoyImageTokenizer(Qwen3VLTokenizer):
    """JoyImageEdit tokenizer.

    ``tokenize_with_weights(text, images=[...])`` selects the image-conditioned
    template when one or more image tensors are passed, emitting one
    ``<|vision_start|><|image_pad|><|vision_end|>`` block per image (N blocks
    for N reference images), otherwise the text-only template. Each
    ``<|image_pad|>`` token in the formatted prompt is replaced with an
    embedding marker so `SDClipModel.process_tokens` routes each image through
    `Qwen3VL8B_JoyImage.preprocess_embed`; ``drop_idx=34`` leading template
    tokens are stripped downstream by `JoyImageTEModel.encode_token_weights`.
    No ``<think>`` block is appended.
    """

    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(
            embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
            model_type="qwen3vl_8b",
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
            # One vision block per reference image.
            vision = JOYIMAGE_VISION_BLOCK * len(images)
            llama_text = self.llama_template_images.format(vision=vision, prompt=text)
        else:
            llama_text = self.llama_template.format(text)

        # Tokenize the already-rendered template via the grandparent
        # (SD1Tokenizer); calling `super()` would re-apply the Qwen3VL template.
        tokens = sd1_clip.SD1Tokenizer.tokenize_with_weights(
            self, llama_text, return_word_ids=return_word_ids, disable_weights=True, **kwargs,
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

    Conditions on the **pre-final-norm** output of the last decoder layer
    (``layer="hidden", layer_idx=-1, layer_norm_hidden_state=False``). The
    post-norm ``last_hidden_state`` differs by ~10x in scale and produces broken
    DiT outputs, so these flags must not be changed.
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

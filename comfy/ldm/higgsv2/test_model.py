"""Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio."""

import torch
import torch.nn as nn
from transformers.models.auto import CONFIG_MAPPING
import math
import glob
import functools

import os
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from safetensors.torch import load_file
from typing import Optional, Tuple, Union, List, Dict, Any

from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateNonBeamOutput
from transformers.utils import logging, ModelOutput
from transformers.configuration_utils import PretrainedConfig
from torch import nn

def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to
from transformers.modeling_utils import PreTrainedModel

class HiggsAudioConfig(PretrainedConfig):

    model_type = "higgs_audio"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        audio_encoder_config=None,
        audio_tokenizer_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_ffn_hidden_size=4096,
        audio_ffn_intermediate_size=14336,
        audio_dual_ffn_layers=None,
        audio_decoder_proj_num_layers=0,
        encode_whisper_embed=True,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        skip_audio_tower=False,
        use_audio_out_embed_projector=False,
        use_audio_out_self_attention=False,
        use_rq_transformer=False,
        rq_transformer_hidden_size=None,
        rq_transformer_intermediate_size=None,
        rq_transformer_num_attention_heads=None,
        rq_transformer_num_key_value_heads=None,
        rq_transformer_num_hidden_layers=3,
        audio_num_codebooks=12,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        pad_token_id=128001,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        **kwargs,
    ):

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        assert audio_adapter_type in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ], f"Invalid audio adapter type: {audio_adapter_type}"
        if audio_adapter_type.startswith("dual_ffn"):
            assert audio_dual_ffn_layers is not None, (
                "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
            )
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.encode_whisper_embed = encode_whisper_embed
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        self.use_rq_transformer = use_rq_transformer

        if self.use_rq_transformer:
            assert not self.use_delay_pattern, "Delay pattern is not supported if you turned on RQ-Transformer!"
        self.rq_transformer_hidden_size = rq_transformer_hidden_size
        self.rq_transformer_intermediate_size = rq_transformer_intermediate_size
        self.rq_transformer_num_attention_heads = rq_transformer_num_attention_heads
        self.rq_transformer_num_key_value_heads = rq_transformer_num_key_value_heads
        self.rq_transformer_num_hidden_layers = rq_transformer_num_hidden_layers

        if use_rq_transformer:
            # For RQ-Transformer, we set the hidden_size to the same as the text model's hidden size if it is not specified.
            if self.rq_transformer_hidden_size is None:
                self.rq_transformer_hidden_size = text_config.hidden_size
            assert self.rq_transformer_hidden_size % 128 == 0
            if self.rq_transformer_intermediate_size is None:
                self.rq_transformer_intermediate_size = text_config.intermediate_size
            if self.rq_transformer_num_attention_heads is None:
                self.rq_transformer_num_attention_heads = self.rq_transformer_hidden_size // 128
            if self.rq_transformer_num_key_value_heads is None:
                self.rq_transformer_num_key_value_heads = self.rq_transformer_hidden_size // 128 // 4
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_attention_heads == 0
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_key_value_heads == 0

        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id


class HiggsAudioPreTrainedModel(PreTrainedModel):
    config_class = HiggsAudioConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

                
def merge_input_ids_with_audio_features(
    audio_in_embed, audio_in_ids_start, audio_out_embed, audio_out_ids_start,
    audio_in_token_idx, audio_out_token_idx, inputs_embeds, input_ids,
    attention_mask, label_ids, pad_token_id, ignore_index=-100,
    round_to=8, left_padding=True,
):
    def compute_audio_codes_length(ids_start, embed):
        return torch.concat([
            ids_start[1:] - ids_start[:-1],
            torch.tensor([embed.shape[0] - ids_start[-1]], device=ids_start.device, dtype=torch.long),
        ], dim=0).long()

    def fill_audio_embeddings(final_embedding, final_input_ids, final_labels, final_audio_mask,
                              embed, token_idx, ids_start, codes_length, token_ends, batch_id,
                              skip_labels, ignore_index):
        seq_indices = torch.arange(max_token_num, device=target_device).unsqueeze(0).expand(ids_start.shape[0], max_token_num)
        token_starts = token_ends - codes_length + 1
        batch_indices, col_indices = torch.where((seq_indices >= token_starts.unsqueeze(1)) & (seq_indices <= token_ends.unsqueeze(1)))
        batch_indices = batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = embed
        final_input_ids[batch_indices, col_indices] = token_idx
        if not skip_labels: final_labels[batch_indices, col_indices] = ignore_index
        final_audio_mask[batch_indices, col_indices] = True

    skip_labels = label_ids is None
    if audio_in_embed is not None and audio_in_embed.shape[0] == 0: audio_in_embed = None
    if audio_out_embed is not None and audio_out_embed.shape[0] == 0: audio_out_embed = None

    batch_size, sequence_length, embed_dim = inputs_embeds.shape
    target_device = inputs_embeds.device
    if left_padding is None: left_padding = torch.any(attention_mask[:, 0] == 0)

    audio_in_token_mask, audio_out_token_mask = input_ids == audio_in_token_idx, input_ids == audio_out_token_idx
    text_token_mask = (input_ids != audio_in_token_idx) & (input_ids != audio_out_token_idx)
    token_placeholder_num = torch.ones_like(input_ids)

    if audio_in_embed is not None:
        audio_in_codes_length = compute_audio_codes_length(audio_in_ids_start, audio_in_embed)
        token_placeholder_num[audio_in_token_mask] = audio_in_codes_length
    if audio_out_embed is not None:
        audio_out_codes_length = compute_audio_codes_length(audio_out_ids_start, audio_out_embed)
        token_placeholder_num[audio_out_token_mask] = audio_out_codes_length

    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = _ceil_to_nearest(token_placeholder_num.sum(-1).max(), round_to)
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
    if left_padding: new_token_positions += nb_audio_pad[:, None]

    final_embedding = torch.zeros((batch_size, max_token_num, embed_dim), dtype=inputs_embeds.dtype, device=target_device)
    final_attention_mask = torch.zeros((batch_size, max_token_num), dtype=attention_mask.dtype, device=target_device)
    final_input_ids = torch.full((batch_size, max_token_num), pad_token_id, dtype=input_ids.dtype, device=target_device)
    final_labels = None if skip_labels else torch.full((batch_size, max_token_num), ignore_index, dtype=label_ids.dtype, device=target_device)
    final_audio_in_mask = torch.zeros((batch_size, max_token_num), dtype=torch.bool, device=target_device)
    final_audio_in_discrete_codes_mask = torch.zeros((batch_size, max_token_num), dtype=torch.bool, device=target_device)
    final_audio_out_mask = torch.zeros((batch_size, max_token_num), dtype=torch.bool, device=target_device)

    batch_id = torch.arange(batch_size, device=target_device).unsqueeze(1).expand(batch_size, sequence_length)
    audio_in_batch_id, audio_out_batch_id = batch_id[audio_in_token_mask], batch_id[audio_out_token_mask]
    audio_in_token_ends, audio_out_token_ends = new_token_positions[audio_in_token_mask], new_token_positions[audio_out_token_mask]

    if audio_in_embed is not None:
        fill_audio_embeddings(final_embedding, final_input_ids, final_labels, final_audio_in_mask,
                              audio_in_embed, audio_in_token_idx, audio_in_ids_start,
                              audio_in_codes_length, audio_in_token_ends, audio_in_batch_id,
                              skip_labels, ignore_index)
        final_audio_in_discrete_codes_mask = final_audio_in_mask.clone()

    if audio_out_embed is not None:
        fill_audio_embeddings(final_embedding, final_input_ids, final_labels, final_audio_out_mask,
                              audio_out_embed, audio_out_token_idx, audio_out_ids_start,
                              audio_out_codes_length, audio_out_token_ends, audio_out_batch_id,
                              skip_labels, ignore_index)

    batch_indices, text_indices = torch.where(text_token_mask)
    text_to_overwrite = new_token_positions[batch_indices, text_indices]
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, text_indices]
    if not skip_labels: final_labels[batch_indices, text_to_overwrite] = label_ids[batch_indices, text_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, text_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, text_indices]

    final_attention_mask |= final_audio_in_mask | final_audio_out_mask

    if left_padding:
        first_non_zero_loc = (final_attention_mask.sum(0).nonzero()[0] // round_to) * round_to
        if first_non_zero_loc > 0:
            final_attention_mask = final_attention_mask[:, first_non_zero_loc:]
            final_embedding = final_embedding[:, first_non_zero_loc:]
            if not skip_labels: final_labels = final_labels[:, first_non_zero_loc:]
            final_input_ids = final_input_ids[:, first_non_zero_loc:]
            final_audio_in_mask = final_audio_in_mask[:, first_non_zero_loc:]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, first_non_zero_loc:]
            final_audio_out_mask = final_audio_out_mask[:, first_non_zero_loc:]
    else:
        last_non_zero_loc = ((final_attention_mask.sum(0).nonzero()[-1] + 1 + round_to - 1) // round_to) * round_to
        if last_non_zero_loc < max_token_num:
            final_attention_mask = final_attention_mask[:, :last_non_zero_loc]
            final_embedding = final_embedding[:, :last_non_zero_loc]
            if not skip_labels: final_labels = final_labels[:, :last_non_zero_loc]
            final_input_ids = final_input_ids[:, :last_non_zero_loc]
            final_audio_in_mask = final_audio_in_mask[:, :last_non_zero_loc]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, :last_non_zero_loc]
            final_audio_out_mask = final_audio_out_mask[:, :last_non_zero_loc]

    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(final_attention_mask == 0, 1)

    return (final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids,
            final_audio_in_mask, final_audio_in_discrete_codes_mask, final_audio_out_mask)


from torch.nn import RMSNorm
from cuda_graph_runner import CUDAGraphRunner

class HiggsAudioDecoderProjector(HiggsAudioPreTrainedModel):
    """Projection layers that map hidden states from the LLM component to audio / text logits.

    We support two type of audio head:
    - Basic Audio Head:
        Directly map the hidden states to audio logits for all the codebooks.
    """

    def __init__(self, config: HiggsAudioConfig, layer_idx: Optional[int] = None):
        super().__init__(config)
        self.text_lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.audio_lm_head = nn.Linear(
            config.text_config.hidden_size, config.audio_num_codebooks * (config.audio_codebook_size + 2), bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        hidden_states,
        audio_out_mask,
        label_audio_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_audio_hidden_states=False,
        cache_position=None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_size)`):
                Hidden states from the LLM component
            audio_out_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask for identifying the audio out tokens.
            label_audio_ids (`torch.Tensor` of shape `(num_codebooks, num_audio_out_tokens)`):
                Label tokens for the audio-out part. This is used for calculating the logits if RQ-Transformer is used.
            attention_mask (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Mask to avoid performing attention on padding token indices
            position_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
                Position ids for the input tokens

        Returns:
            logits (`torch.Tensor` of shape `(batch_size, seq_len, vocab_size)`):
                Logits for text tokens
            audio_logits (`torch.Tensor` of shape `(num_audio_out_tokens, audio_num_codebooks * audio_codebook_size)`):
                Logits for audio tokens. We ensure `num_text_tokens + num_audio_tokens == batch_size * seq_len`
        """
        logits = self.text_lm_head(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        if self.config.audio_decoder_proj_num_layers > 0:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.transformer_layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states = layer_outputs[0]
            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        next_cache = next_decoder_cache if use_cache else None

        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])

        if output_audio_hidden_states:
            audio_hidden_states = hidden_states[audio_out_mask]
        else:
            audio_hidden_states = None

        return logits, audio_logits

logger = logging.get_logger(__name__)


class GenerationMode(Enum):
    """Enum for different generation modes in HiggsAudio model."""

    TEXT = 0  # Text generation mode
    AUDIO_INIT = 1  # Audio generation mode initialization
    AUDIO_IN_PROGRESS = 2  # Audio generation mode in progress


def _whisper_encoder_zero_shape_forward(whisper_encoder, *args, **kwargs):
    """The whisper encoder does not support zero-shape tensor by default due to the following implementations

        key_states = self._shape(self.k_proj(current_states), -1, bsz)

    If `bsz` is 0, the "-1" dimension will be ambiguous and triggers error in the shape inference pass.

    See also: https://github.com/huggingface/transformers/blob/30335093276212ce74938bdfd85bfd5df31a668a/src/transformers/models/whisper/modeling_whisper.py#L306-L307

    This function monkey-patches all `_shape` functions in the whisper encoder's self-attention layers to ensure function supports zero-shape tensor.

    #FIXME!!!! This is a temporary workaround and should be removed once the upstream issue is resolved.

    """

    global _higgs_flash_attention_forward

    def _patched_shape(tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int):
        if seq_len == -1:
            return tensor.view(bsz, tensor.shape[1], num_heads, head_dim).transpose(1, 2).contiguous()
        else:
            return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def _patched_scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
    ) -> torch.Tensor:
        # IMPORTANT! Implementation here is wrong and is only for the purpose of obtaining the correct attn_weight shape
        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1)
        return attn_weight @ value

    # Apply monkey-patch
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        old_shape_functions = []
        for layer in whisper_encoder.layers:
            old_shape_functions.append(getattr(layer.self_attn, "_shape"))
            layer.self_attn._shape = functools.partial(
                _patched_shape, num_heads=layer.self_attn.num_heads, head_dim=layer.self_attn.head_dim
            )

    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = _patched_scaled_dot_product_attention

    out = whisper_encoder(*args, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = original_scaled_dot_product_attention

    # Restore the original shape functions
    if whisper_encoder.config._attn_implementation != "flash_attention_2":
        for layer, old_shape_function in zip(whisper_encoder.layers, old_shape_functions):
            layer.self_attn._shape = old_shape_function

    return out


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask


class HiggsAudioFeatureProjector(nn.Module):
    """Projector that maps audio features extracted by Whisper to hidden state of the text model."""

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.linear = nn.Linear(config.audio_encoder_config.d_model, config.text_config.hidden_size, bias=True)

    def forward(self, audio_features):
        hidden_states = self.linear(audio_features)
        return hidden_states


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(xq, xk, freqs_cis):
    cos = freqs_cis[0].unsqueeze(1)
    sin = freqs_cis[1].unsqueeze(1)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed, sin, cos

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

class LLama3RoPE(nn.Module):
    def __init__(self, config, device = None, dtype = None):
        super().__init__()

        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):

        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
import torch.nn.functional as F
class Attention(nn.Module):

    def __init__(self, config, layer_idx: int, device, dtype, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.inner_size =  config.num_attention_heads * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        optimized_attention=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk, sin, cos = apply_rope(xq, xk, freqs_cis=position_embeddings)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            xk, xv = past_key_value.update(xk, xv, self.layer_idx, cache_kwargs)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        out = self.o_proj(output)


        return out, None, past_key_value
class MLP(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops: Any = nn):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        if config.mlp_activation == "silu":
            self.activation = torch.nn.functional.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: torch.nn.functional.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))

SDP_BATCH_LIMIT = 2**15
def attention_pytorch(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, is_causal = False):
    if skip_reshape:
        b, _, _, dim_head = q.shape
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = map(
            lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
            (q, k, v),
        )

    if mask is not None:
        # add a batch dimension if there isn't already one
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # add a heads dimension if there isn't already one
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
            
    SDP_BATCH_LIMIT = b
    if SDP_BATCH_LIMIT >= b:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=is_causal)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
    else:
        out = torch.empty((b, q.shape[2], heads * dim_head), dtype=q.dtype, layout=q.layout, device=q.device)
        for i in range(0, b, SDP_BATCH_LIMIT):
            m = mask
            if mask is not None:
                if mask.shape[0] > 1:
                    m = mask[i : i + SDP_BATCH_LIMIT]

            out[i : i + SDP_BATCH_LIMIT] = torch.nn.functional.scaled_dot_product_attention(
                q[i : i + SDP_BATCH_LIMIT],
                k[i : i + SDP_BATCH_LIMIT],
                v[i : i + SDP_BATCH_LIMIT],
                attn_mask=m,
                dropout_p=0.0, is_causal=False
            ).transpose(1, 2).reshape(-1, q.shape[2], heads * dim_head)
    return out

from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding, LLAMA_ATTENTION_CLASSES
class HiggsAudioDualFFNDecoderLayer(nn.Module):

    def __init__(
        self, config, layer_idx: int, fast_forward: bool = False, use_audio_attention: bool = False, device = None, dtype = None,
    ):
        super().__init__()
        text_config = config.text_config
        self.hidden_size = text_config.hidden_size
        self.layer_idx = layer_idx
        text_config.qkv_bias = text_config.mlp_bias
        text_config.mlp_activation = text_config.hidden_act
        self.self_attn = Attention(config=text_config, layer_idx=layer_idx, device = device, dtype = dtype)
        #self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=text_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(text_config)

        if not fast_forward:
            self.audio_mlp = LlamaMLP(text_config)#, device = device, dtype = dtype)
            self.audio_input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)#, device = device, dtype = dtype)
            self.audio_post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)#, device = device, dtype = dtype)

        self.use_audio_attention = use_audio_attention
        self.fast_forward = fast_forward
        if self.fast_forward:
            assert not self.use_audio_attention, (
                "We cannot use audio_attention if the layer is marked as fast-forward."
            )
        self.input_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)#, device = device, dtype = dtype)
        self.post_attention_layernorm = LlamaRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)#, device = device, dtype = dtype)

        self.text_config = text_config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fast_forward_attention_mask: Optional[torch.Tensor] = None,
        audio_out_mask: Optional[torch.BoolTensor] = None,
        is_decoding_audio_token: Optional[bool] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_using_cuda_graph: Optional[bool] = False,
        position_embeddings = None,
        **kwargs,
    ):

        residual = hidden_states
        target_length = hidden_states.shape[1]
        use_static_cache = isinstance(past_key_value, StaticCache)
        decode_stage = hidden_states.shape[1] == 1
        if is_using_cuda_graph:
            assert decode_stage and use_static_cache, (
                "The CUDA graph mode should only be used in the decoding stage with static cache."
            )

        # If we are decoding an audio token and the layer is marked as fast-forward,
        # we can skip it.
        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        # considering that hidden_state = audio + text, I chose to make small_input dynamic
        small_input = target_length <= 2048
        optimized_attention = attention_pytorch#optimized_attention_for_device(hidden_states.device, small_input = small_input)

        if self.fast_forward and has_audio_out:
            original_hidden_states = hidden_states.clone()
            min_dtype = torch.finfo(hidden_states.dtype).min
            if attention_mask is None:
                attention_mask = ~audio_out_mask

                if optimized_attention.__name__ != "attention_flash":
                    sequence_length = audio_out_mask.shape[1]
                    attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                        attention_mask=attention_mask,
                        sequence_length=sequence_length,
                        target_length=sequence_length,
                        dtype=hidden_states.dtype,
                        min_dtype=min_dtype,
                        device=hidden_states.device,
                        cache_position=cache_position,
                        batch_size=hidden_states.shape[0],
                    )
                    if use_cache:
                        attention_mask = attention_mask[:, :, -target_length:, :]
            elif len(attention_mask.shape) == 2:
                # Attention mask has shape (batch_size, sequence_length)
                # We should be using flash attention 2
                attention_mask = attention_mask * ~audio_out_mask
            elif len(attention_mask.shape) == 4:
                # When using static cache, the attention mask was already preprocessed in the previous layer
                if use_static_cache:
                    attention_mask = fast_forward_attention_mask
                else:
                    if use_cache:
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask[:, -target_length:].reshape(audio_out_mask.shape[0], 1, target_length, 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                        )
                    else:
                        attention_mask = attention_mask.masked_fill(
                            audio_out_mask.reshape(audio_out_mask.shape[0], 1, audio_out_mask.shape[1], 1)
                            | audio_out_mask.reshape(audio_out_mask.shape[0], 1, 1, audio_out_mask.shape[1]),
                            min_dtype,
                        )
            else:
                raise NotImplementedError(f"Unsupported attention_mask format, attention_mask={attention_mask}")

            if (
                optimized_attention.__name__ == "attention_pytorch"
                and attention_mask is not None
                and attention_mask.device.type == "cuda"
            ):
                attention_mask =  attention_mask.mul(~torch.all(attention_mask == min_dtype, dim=-1, keepdim=True))

        if has_audio_out and not self.fast_forward:
            # Apply separate layernorm layers for audio tokens and text tokens
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
            else:
                hidden_states = torch.where(
                    audio_out_mask_sq.unsqueeze(-1),
                    self.audio_input_layernorm(hidden_states),
                    self.input_layernorm(hidden_states),
                )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Text Attention
        #freqs_cis = precompute_freqs_cis(self.text_config.head_dim, hidden_states.shape[1], self.text_config.rope_theta, device = hidden_states.device)
        hidden_states, _, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_embeddings = position_embeddings,
            past_key_value=past_key_value,
            optimized_attention = optimized_attention,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if has_audio_out and not self.fast_forward:
            if use_cache:
                real_audio_out_mask = audio_out_mask_sq[:, -target_length:]
            else:
                real_audio_out_mask = audio_out_mask_sq

            # Make whole graph in decode stage
            if decode_stage and is_using_cuda_graph:
                assert is_decoding_audio_token is not None, (
                    "is_decoding_audio_token should be present in the decoding stage."
                )
                if is_decoding_audio_token:
                    hidden_states = self.audio_post_attention_layernorm(hidden_states)
                    hidden_states = self.audio_mlp(hidden_states)
                else:
                    hidden_states = self.post_attention_layernorm(hidden_states)
                    hidden_states = self.mlp(hidden_states)
                residual = residual + hidden_states
            else:
                text_hidden_states = self.post_attention_layernorm(hidden_states[~real_audio_out_mask])
                audio_hidden_states = self.audio_post_attention_layernorm(hidden_states[real_audio_out_mask])

                text_hidden_states = self.mlp(text_hidden_states)
                residual[~real_audio_out_mask] += text_hidden_states

                audio_hidden_states = self.audio_mlp(audio_hidden_states)
                residual[real_audio_out_mask] += audio_hidden_states

            hidden_states = residual
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

        if self.fast_forward and has_audio_out:
            if use_cache:
                hidden_states = torch.where(
                    audio_out_mask_sq[:, -target_length:].unsqueeze(-1), original_hidden_states, hidden_states
                )
            else:
                hidden_states = torch.where(audio_out_mask_sq.unsqueeze(-1), original_hidden_states, hidden_states)

        outputs = (hidden_states,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

@dataclass
class HiggsAudioModelOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    llm_loss: Optional[torch.FloatTensor] = None
    audio_loss: Optional[torch.FloatTensor] = None
    codebook_losses: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    expanded_input_ids: Optional[torch.LongTensor] = None
    expanded_labels: Optional[torch.LongTensor] = None
    audio_in_mask: Optional[torch.BoolTensor] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None
    attention_mask: Optional[torch.BoolTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    audio_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class HiggsAudioGenerationOutput(ModelOutput):
    """
    Outputs of HiggsAudio generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        audio_sequences (`tuple(torch.LongTensor)` *optional*):
            The generated discrete audio codes. These codes can be used to fill-in related locations of <|AUDIO_OUT|> at input sequences.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True`):
            Unprocessed prediction scores of the language modeling head or the audio head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token).
            If the generated token is a text token, the tensor will have shape `(batch_size, config.vocab_size)`.
            If the generated token is an audio token, the tensor will have shape `(config.audio_num_codebooks, self.audio_codebook_size)`
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True`):
            Returns the model cache, used to speed up decoding. Different models have a different cache format, check
            the model's documentation. Usually, a [`~cache_utils.Cache`] instance.
    """

    sequences: torch.LongTensor = None
    audio_sequences: Optional[List[torch.LongTensor]] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class HiggsAudioModel(HiggsAudioPreTrainedModel, GenerationMixin):
    """Higgs-Audio is an end-to-end multimodal model with the capability to understand and generate text / audio.

    Consider the following example for mixed text/audio understanding / generation:

    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_bos|>[AUDIO]<|audio_eos|><text_token4>
    - input_tokens: <text_token1><|audio_bos|>[AUDIO]<|audio_eos|><text_token2><|audio_out_bos|>[AUDIO_OUT]<|audio_eos|><text_token4>

    We will fill [AUDIO] with the audio features extracted by Whisper and fill [AUDIO_OUT] with the audio tokens.

    Consider the following example for mixed text/audio generation:

    text: <|audio_out_bos|>    MASK           MASK           MASK          MASK               MASK         <|audio_eos|> [text_token1]
    audio:     MASK    <|audio_stream_bos|> [audio_token1] [audio_token2] [audio_token3] <|audio_stream_eos|>   MASK           MASK
    token_type: 0               1              1              1             1                  1                 0              0

    """

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, config: HiggsAudioConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.audio_in_token_idx = config.audio_in_token_idx
        self.audio_out_token_idx = config.audio_out_token_idx
        self.audio_out_bos_token_id = config.audio_out_bos_token_id if "audio_out_bos_token_id" in config else None
        self.audio_eos_token_id = config.audio_eos_token_id if "audio_eos_token_id" in config else None
        self.vocab_size = config.text_config.vocab_size
        self.audio_num_codebooks = config.audio_num_codebooks
        self.use_delay_pattern = config.use_delay_pattern
        self.use_audio_out_embed_projector = config.use_audio_out_embed_projector
        self.use_audio_out_self_attention = config.use_audio_out_self_attention

        self.embed_tokens = nn.Embedding(self.vocab_size, config.text_config.hidden_size, self.padding_idx)

        if config.audio_adapter_type == "dual_ffn":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config, layer_idx, use_audio_attention=self.use_audio_out_self_attention
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(LlamaDecoderLayer(config.text_config, layer_idx))
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "dual_ffn_fast_forward":
            layer_idx = 0
            layers = []
            for j in range(config.text_config.num_hidden_layers):
                if j in config.audio_dual_ffn_layers:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            config,
                            layer_idx,
                            fast_forward=False,
                            use_audio_attention=self.use_audio_out_self_attention,
                        )
                    )
                    layer_idx += 2 if self.use_audio_out_self_attention else 1
                else:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(config, layer_idx, fast_forward=True, use_audio_attention=False)
                    )
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        elif config.audio_adapter_type == "stack":
            self.layers = nn.ModuleList(
                [
                    LlamaDecoderLayer(config.text_config, layer_idx)
                    for layer_idx in range(config.text_config.num_hidden_layers)
                ]
            )
            layer_idx = config.text_config.num_hidden_layers
        else:
            raise NotImplementedError(f"Audio adapter type {config.audio_adapter_type} not implemented.")

        self.num_activation_checkpointing_layers = len(self.layers)

        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = RMSNorm(config.text_config.hidden_size, eps=config.text_config.rms_norm_eps)
        self.rotary_emb = LLama3RoPE(config = config.text_config)#LlamaRotaryEmbedding(config=config.text_config)

        if not config.skip_audio_tower:
            self.audio_tower = HiggsAudioEncoder(config.audio_encoder_config)
            self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        else:
            self.audio_tower = None
            self.audio_encoder_proj = None
        self.audio_decoder_proj = HiggsAudioDecoderProjector(config, layer_idx=layer_idx)
        self.audio_codebook_size = (
            config.audio_codebook_size + 2
        )  # We add 1 for the audio_stream_bos token and 1 for the audio_stream_eos token

        if config.use_audio_out_embed_projector:
            self.audio_out_embed_projector = nn.Linear(
                config.text_config.hidden_size, config.text_config.hidden_size, bias=False
            )

        self.audio_codebook_embeddings = nn.Embedding(
            config.audio_num_codebooks * self.audio_codebook_size, config.text_config.hidden_size
        )

        self.audio_codebook_weights = (
            torch.ones(config.audio_num_codebooks) / config.audio_num_codebooks
        )  # default to equal weights
        self.post_init()

    def set_num_activation_checkpointing_layers(self, num_layers):
        self.num_activation_checkpointing_layers = num_layers

    def set_delay_pattern(self):
        self.config.use_delay_pattern = True
        self.use_delay_pattern = True

    def set_audio_special_tokens(self, tokenizer: AutoTokenizer):
        self.audio_out_bos_token_id = tokenizer.convert_tokens_to_ids("<|audio_out_bos|>")
        self.audio_eos_token_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

    def _embed_audio_ids(self, audio_ids):
        """Embed the audio ids

        Args:
            audio_ids: torch.LongTensor of shape (num_codebooks, audio_in_total_length)

        Returns:
            audio_embed: torch.LongTensor of shape (audio_in_total_length, hidden_size)
        """
        codebook_shift = (
            torch.arange(self.config.audio_num_codebooks, device=audio_ids.device) * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        if self.config.audio_embed_avg:
            audio_embed = torch.mean(audio_embed, dim=0)
        else:
            audio_embed = torch.sum(audio_embed, dim=0)
        if self.use_audio_out_embed_projector:
            audio_embed = self.audio_out_embed_projector(audio_embed)
        return audio_embed

    def _apply_audio_tower(self, audio_features, audio_feature_attention_mask):
        """Apply the audio tower to the audio features"""

        if audio_features.shape[0] == 0:
            if torch.is_grad_enabled():
                # FIXME!!!!!!!!
                # This is a hack to ensure that the forward+backward pass of audio_tower and audio_encoder_proj get triggered.
                # The monkey patch won't overwrite the backward pass of nn.Module.
                audio_outputs = _whisper_encoder_zero_shape_forward(
                    self.audio_tower, audio_features, attention_mask=None, check_seq_length=False
                )
                selected_audio_feature = audio_outputs.last_hidden_state
                audio_features_embed = self.audio_encoder_proj(selected_audio_feature)
                audio_feat_out_lengths = None
                return audio_features_embed, audio_feat_out_lengths
            else:
                return None, None

        audio_feat_lengths, audio_feat_out_lengths = self.audio_tower._get_feat_extract_output_lengths(
            audio_feature_attention_mask.sum(-1)
        )
        batch_size, _, max_mel_seq_len = audio_features.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1
        # Create a sequence tensor of shape (batch_size, max_seq_len)
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # Create mask
        padding_mask = seq_range < lengths_expand

        if self.config._attn_implementation != "flash_attention_2":
            audio_attention_mask = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
        else:
            audio_attention_mask = padding_mask

        audio_outputs = self.audio_tower(audio_features, attention_mask=audio_attention_mask)
        selected_audio_feature = audio_outputs.last_hidden_state
        audio_features_embed = self.audio_encoder_proj(selected_audio_feature)

        return audio_features_embed, audio_feat_out_lengths

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def _prepare_all_static_kv_cache_masks(self, hidden_states, attention_mask, audio_out_mask, past_key_values):
        target_length = hidden_states.shape[1]
        cur_pos = audio_out_mask.shape[1]
        min_dtype = torch.finfo(hidden_states.dtype).min
        assert len(attention_mask.shape) == 4, "Only support SDPA for now"
        kv_cache_len = past_key_values.get_max_cache_shape()
        audio_out_mask_padded = torch.nn.functional.pad(audio_out_mask, (0, kv_cache_len - cur_pos), value=True)
        fast_forward_attention_mask = attention_mask.masked_fill(
            audio_out_mask_padded[:, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]].reshape(
                audio_out_mask_padded.shape[0], 1, target_length, 1
            )
            | audio_out_mask_padded.reshape(audio_out_mask_padded.shape[0], 1, 1, audio_out_mask_padded.shape[1]),
            min_dtype,
        )

        no_audio_out_mask = ~audio_out_mask
        no_audio_out_mask = torch.nn.functional.pad(
            no_audio_out_mask, (0, kv_cache_len - audio_out_mask.shape[1]), value=False
        )
        no_audio_out_mask = no_audio_out_mask[
            :, audio_out_mask.shape[1] - target_length : audio_out_mask.shape[1]
        ].reshape(audio_out_mask.shape[0], 1, target_length, 1) | no_audio_out_mask.reshape(
            audio_out_mask.shape[0], 1, 1, kv_cache_len
        )
        audio_attention_mask = attention_mask.masked_fill(no_audio_out_mask, min_dtype)
        return fast_forward_attention_mask, audio_attention_mask

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        audio_discrete_codes_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]],
        use_cache: bool,
        audio_attention_mask: torch.Tensor,
        fast_forward_attention_mask: torch.Tensor,
        output_attentions: bool,
        output_hidden_states: bool,
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
    ):
        # create position embeddings to be shared across the decoder layers
        # When past_key_values is passed in, we need to offset the position ids when calculating the position embeddings.
        # Therefore, cache_position is used.
        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if isinstance(decoder_layer, HiggsAudioDualFFNDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    position_ids=position_ids,
                    audio_out_mask=audio_discrete_codes_mask,
                    is_decoding_audio_token=is_decoding_audio_token,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    is_using_cuda_graph=is_using_cuda_graph,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        return hidden_states, all_hidden_states, all_self_attns

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        label_ids: Optional[torch.LongTensor] = None,
        label_audio_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_audio_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cache_audio_discrete_codes_mask: Optional[torch.LongTensor] = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        reward = None,
        audio_features = None,
        **kwargs
        #**kwargs
    ):

        target_device = input_ids.device

        inputs_embeds = self.embed_tokens(input_ids)

        if self.config.encode_audio_in_tokens:
            if audio_in_ids is not None and audio_in_ids.shape[-1] > 0:
                audio_in_ids = audio_in_ids.to(target_device)
            else:
                audio_in_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
            audio_in_embed = self._embed_audio_ids(audio_in_ids)
        else:
            audio_in_embed = None

        if audio_out_ids is not None and audio_out_ids.shape[-1] > 0:
            audio_out_ids = audio_out_ids.to(target_device)
        else:
            audio_out_ids = torch.zeros((self.audio_num_codebooks, 0), device=target_device, dtype=torch.long)
        audio_out_embed = self._embed_audio_ids(audio_out_ids)

        round_to = 1 if use_cache else 8
        left_padding = True if use_cache or input_ids.shape[0] == 1 else False
        (
            inputs_embeds,
            attention_mask,
            labels,
            position_ids,
            input_ids,
            audio_in_mask,
            audio_in_discrete_codes_mask,
            audio_out_mask,
        ) = merge_input_ids_with_audio_features(
            audio_in_embed,
            audio_in_ids_start,
            audio_out_embed,
            audio_out_ids_start,
            self.audio_in_token_idx,
            self.audio_out_token_idx,
            inputs_embeds,
            input_ids,
            attention_mask,
            label_ids,
            pad_token_id=self.padding_idx,
            round_to=round_to,
            left_padding=left_padding,
        )

        # re-check if we use the correct kv cache bucket after
        # the input_embeds has been merged with audio features
        if past_key_values_buckets is not None and inputs_embeds.shape[1] > past_key_values.get_max_cache_shape():
            past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                inputs_embeds.shape[1], None, past_key_values_buckets
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            if isinstance(past_key_values, StaticCache) and past_seen_tokens >= past_key_values.get_max_cache_shape():
                raise ValueError(
                    f"The current sequence length ({past_seen_tokens}) exceeds "
                    f"the maximum cache shape. "
                    f"Please consider increasing the cache size."
                )

        # Use torch compile
        use_static_cache = isinstance(past_key_values, StaticCache)

        # Apply the LLM component
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        audio_discrete_codes_mask = audio_in_discrete_codes_mask | audio_out_mask
        if cache_audio_discrete_codes_mask is not None and use_cache:
            audio_discrete_codes_mask = torch.concat(
                [cache_audio_discrete_codes_mask, audio_discrete_codes_mask], dim=1
            )

        # Generate the audio attention mask outside the layer to avoid recompilation
        if use_static_cache:
            fast_forward_attention_mask, audio_attention_mask = self._prepare_all_static_kv_cache_masks(
                hidden_states, causal_mask, audio_discrete_codes_mask, past_key_values
            )
            # Set the audio out mask to the last token
            if hidden_states.shape[1] == 1:
                audio_discrete_codes_mask = audio_discrete_codes_mask[:, -1:]
                audio_discrete_codes_mask = audio_discrete_codes_mask.reshape((-1, 1)).contiguous()
                is_decoding_audio_token = audio_discrete_codes_mask.item()
            else:
                is_decoding_audio_token = False

        if (
            past_key_values is not None
            and past_key_values.get_max_cache_shape() in self.decode_graph_runners
            and (input_ids.shape[-1] == 1)
        ):
            _forward_core = self.decode_graph_runners[past_key_values.get_max_cache_shape()][is_decoding_audio_token]
            is_using_cuda_graph = True
        else:
            _forward_core = self._forward_core
            is_using_cuda_graph = False

        hidden_states = _forward_core(
            hidden_states=hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            audio_discrete_codes_mask=audio_discrete_codes_mask,
            is_decoding_audio_token=is_decoding_audio_token if use_static_cache else None,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_attention_mask=audio_attention_mask if use_static_cache else None,
            fast_forward_attention_mask=fast_forward_attention_mask if use_static_cache else None,
            is_using_cuda_graph=is_using_cuda_graph,
            output_hidden_states = False,
            output_attentions = False
        )
        #print(hidden_states)
        hidden_states = self.norm(hidden_states[0])

        # Apply the audio decoder projector
        logits, audio_logits = (
            self.audio_decoder_proj(
                hidden_states,
                audio_out_mask,
                label_audio_ids=label_audio_ids,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_audio_hidden_states=output_audio_hidden_states,
                cache_position=cache_position,
            )
        )

        if audio_logits is not None:
            audio_logits = audio_logits.view(
                audio_logits.shape[0], self.audio_num_codebooks, self.audio_codebook_size
            ).float()

        next_cache = past_key_values if use_cache else None

        ret = HiggsAudioModelOutputWithPast(
            logits=logits,
            audio_logits=audio_logits,
            past_key_values=next_cache,
            audio_out_mask = audio_out_mask,
            audio_in_discrete_codes_mask = audio_in_discrete_codes_mask
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            outputs = ret.to_tuple()
            return outputs

        return ret

    # Overwrite GenerationMixin._update_model_kwargs_for_generation
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
        extend_attention_mask: bool = True,
    ) -> Dict[str, Any]:
        """Update the model kwargs for each step."""
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if extend_attention_mask:
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        if "cache_audio_discrete_codes_mask" in model_kwargs:
            if model_kwargs["cache_audio_discrete_codes_mask"] is None:
                model_kwargs["cache_audio_discrete_codes_mask"] = (
                    outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask
                )
            else:
                model_kwargs["cache_audio_discrete_codes_mask"] = torch.concat(
                    [
                        model_kwargs["cache_audio_discrete_codes_mask"],
                        outputs.audio_in_discrete_codes_mask | outputs.audio_out_mask,
                    ],
                    1,
                )

        return model_kwargs

    def _copy_kv_cache(self, from_cache: Cache, to_cache: Cache):
        num_layers = self.config.text_config.num_hidden_layers
        if self.config.audio_dual_ffn_layers is not None:
            num_layers += len(self.config.audio_dual_ffn_layers)
        """ Copy the key-value pairs from one cache to another. """
        for layer_idx in range(num_layers):
            from_cache_size = from_cache.get_max_cache_shape()
            assert to_cache.get_max_cache_shape() >= from_cache_size, (
                f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
            )
            to_cache.key_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.key_cache[layer_idx]
            to_cache.value_cache[layer_idx][:, :, :from_cache_size, :] = from_cache.value_cache[layer_idx]

    def _prepare_kv_cache(
        self,
        current_sequence_length: int,
        current_past_key_values_bucket: Optional[int],
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> Tuple[Optional[Cache], Optional[int]]:
        """Prepare the KV cache for the current sequence length."""
        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:
                # Promote to the next KV cache bucket, copy the current KV cache bucket
                # to the new one.
                if current_past_key_values_bucket is not None and cache_length != current_past_key_values_bucket:
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket], past_key_values_buckets[cache_length]
                    )

                return past_key_values_buckets[cache_length], cache_length

        raise ValueError(
            f"The current sequence length {current_sequence_length} is larger than "
            f"all past key values buckets {past_key_values_buckets.keys()}."
        )

    def _sample_audio_tokens(
        self,
        hidden_states: torch.Tensor,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""

        # parameters related to repetition aware sampling
        ras_win_len = generation_config.generation_kwargs.get("ras_win_len", None)
        ras_win_max_num_repeat = generation_config.generation_kwargs.get("ras_win_max_num_repeat", 2)
        audio_eos_token_id = generation_config.generation_kwargs.get("audio_eos_token_id", None)
        # In the audio generation mode, we sample from audio_logits and keep updating audio_out_ids.
        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        # TopP, TopK logits processor supports empty input_ids
        next_audio_token_scores = logits_processor(None, next_audio_token_logits)

        # token selection
        if do_sample:
            # next_audio_token_scores has been applied top_p, top_k, and temperature.
            probs = nn.functional.softmax(next_audio_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_audio_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        # next_tokens: (num_codebooks, )
        if ras_win_len is not None:
            # check if there are repetitions over a window of tokens.
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

            # if we saw repeated tokens in the most recent window of tokens, resample without temperature.
            row_indices = torch.nonzero(rep_num >= ras_win_max_num_repeat).squeeze(1)
            resampled_next_tokens = (
                next_audio_token_logits[row_indices]
                .softmax(dim=-1)
                .multinomial(1, replacement=True, generator=torch_generator)
                .squeeze(1)
            )
            next_audio_tokens[row_indices] = resampled_next_tokens

        # Force the next text tokens to be <|AUDIO_OUT|> in audio generation mode
        next_tokens = torch.full(
            (audio_logits.shape[0],),
            self.config.audio_out_token_idx,
            dtype=torch.long,
            device=device,
        )

        # Handle delay_pattern
        if self.use_delay_pattern:
            if num_delay + 1 < next_audio_tokens.shape[0]:
                next_audio_tokens[(num_delay + 1) :] = self.config.audio_stream_bos_id
                num_delay += 1
            if num_remaining_delays is not None:
                next_audio_tokens[: (self.audio_num_codebooks - num_remaining_delays)] = (
                    self.config.audio_stream_eos_id
                )
                num_remaining_delays -= 1
            else:
                all_eos_indices = (next_audio_tokens == self.config.audio_stream_eos_id).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    next_audio_tokens[:last_eos_idx] = self.config.audio_stream_eos_id
                    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1
            if num_remaining_delays is not None and num_remaining_delays <= 0:
                next_tokens[...] = audio_eos_token_id
                num_delay = 0
                num_remaining_delays = None

        return (
            next_tokens,
            next_audio_tokens,
            next_audio_token_logits,
            next_audio_token_scores,
            num_delay,
            num_remaining_delays,
        )

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        do_sample: bool,
        logits_processor: LogitsProcessorList,
        device: torch.device,
        generation_mode: GenerationMode,
        torch_generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""
        # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config.audio_num_codebooks,),
                self.config.audio_stream_bos_id,
                dtype=torch.long,
                device=device,
            )
        else:
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1, generator=torch_generator).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_audio_tokens = None

        return next_tokens, next_audio_tokens, next_token_logits, next_token_scores

    # Built on top of GenerationMixin._sample.
    # We revise the implementation to support generating both audio / text.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        past_key_values_buckets: Optional[OrderedDict[int, Cache]],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for joint text/audio models using **multinomial sampling**.

        This function may also be revised to support generating samples from HiggsAudio-like end-to-end text/audio models built on top of LLMs.
        If the input_ids ends with <|audio_out_bos|>, we will switch to the audio-generation mode.

        ```
        ...<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|>
        ```

        Otherwise, we will keep generating the text tokens.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        assert input_ids.shape[0] == 1, "Only support batch_size=1 in _sample()"
        audio_out_bos_token_id = generation_config.generation_kwargs.get("audio_out_bos_token_id", None)

        # torch generator for sampling
        seed = generation_config.generation_kwargs.get("seed", None)
        if seed is not None:
            torch_generator = torch.Generator(device=input_ids.device).manual_seed(seed)
        else:
            torch_generator = None

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        # Used to track which past_key_va
        self.current_past_key_values_bucket = None

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        if generation_config.use_cache:
            model_kwargs["cache_audio_discrete_codes_mask"] = None

        init_model_input = True
        num_delay = 0
        num_remaining_delays = None
        audio_sequences = []
        # A tensor to keep track of all the audio placeholder tokens.
        input_ids_full = input_ids.clone()

        # Initialize the audio variables based on the input prompt.
        if input_ids[0][-1] == self.config.audio_out_token_idx:
            audio_sequences = [model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1] :]]
            if self.use_delay_pattern:
                num_delay = (
                    self.audio_num_codebooks
                    - (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_bos_id).sum()
                )
                all_eos_indices = (model_kwargs["audio_out_ids"][:, -1] == self.config.audio_stream_eos_id).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1

        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device, cur_len=cur_len, max_length=max_length
        ):
            # Check which multimodal stage we are in
            # FIXME: Assume single input generation
            if input_ids[0][-1] == audio_out_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
            else:
                generation_mode = GenerationMode.TEXT

            is_audio_generation_mode = generation_mode == GenerationMode.AUDIO_IN_PROGRESS

            if init_model_input or not generation_config.use_cache:
                model_inputs = {"input_ids": input_ids, **model_kwargs}
            else:
                model_inputs = {"input_ids": input_ids[:, -1:], **model_kwargs}

                if is_audio_generation_mode and generation_config.use_cache:
                    model_inputs["audio_out_ids"] = model_kwargs["audio_out_ids"][:, -1:]
                    model_inputs["audio_out_ids_start"] = torch.tensor([0], dtype=torch.long, device=input_ids.device)
                elif not is_audio_generation_mode:
                    del model_inputs["audio_out_ids"]
                    del model_inputs["audio_out_ids_start"]

                if generation_config.use_cache:
                    if "audio_features" in model_inputs and model_inputs["audio_features"] is not None:
                        model_inputs["audio_features"] = model_inputs["audio_features"][:0, ...]
                        model_inputs["audio_feature_attention_mask"] = model_inputs["audio_feature_attention_mask"][
                            :0, ...
                        ]

                    if "audio_in_ids" in model_inputs and model_inputs["audio_in_ids"] is not None:
                        model_inputs["audio_in_ids"] = None
                        model_inputs["audio_in_ids_start"] = None

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            if past_key_values_buckets is not None:
                past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                    cur_len, self.current_past_key_values_bucket, past_key_values_buckets
                )
                if past_key_values is not None:
                    model_inputs.update({"past_key_values": past_key_values})
                model_inputs["past_key_values_buckets"] = past_key_values_buckets

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            # Update the actual sequence length after the first forward pass
            if init_model_input and past_key_values_buckets is not None:
                cur_len = past_key_values_buckets[self.current_past_key_values_bucket].get_seq_length().item()

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                extend_attention_mask=True,
            )

            # After the first forward pass, we can set init_model_input to False.
            init_model_input = False

            if synced_gpus and this_peer_finished:
                continue

            if is_audio_generation_mode:
                # In audio generation mode, we sample the audio tokens from audio logits.
                # It might also generate the audio eos token to end the audio generation.
                (
                    next_tokens,
                    next_audio_tokens,
                    next_audio_token_logits,
                    next_audio_token_scores,
                    num_delay,
                    num_remaining_delays,
                ) = self._sample_audio_tokens(
                    hidden_states=outputs.audio_hidden_states,
                    audio_logits=outputs.audio_logits,
                    audio_out_ids=model_kwargs["audio_out_ids"],
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    torch_generator=torch_generator,
                    generation_config=generation_config,
                    num_delay=num_delay,
                    num_remaining_delays=num_remaining_delays,
                )

                # update generated ids, model inputs, and length for next step
                model_kwargs["audio_out_ids"] = torch.cat(
                    [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
                )
                audio_sequences[-1] = torch.cat([audio_sequences[-1], next_audio_tokens[:, None]], dim=-1)

                if streamer is not None:
                    streamer.put(next_audio_tokens.cpu())
            else:
                # In text generation mode, we sample the text tokens from text logits.
                # It might also generate the audio placeholder token to start the audio generation.
                next_tokens, next_audio_tokens, next_token_logits, next_token_scores = self._sample_text_tokens(
                    input_ids=input_ids,
                    logits=outputs.logits,
                    do_sample=do_sample,
                    logits_processor=logits_processor,
                    device=input_ids.device,
                    generation_mode=generation_mode,
                    torch_generator=torch_generator,
                )

                if streamer is not None:
                    streamer.put(next_tokens.cpu())

                if next_audio_tokens is not None:
                    # If the token is audio bos token, we will generate the audio placeholder token
                    # and the corrensponding audio stream bos token to start the audio generation.
                    audio_sequences.append(next_audio_tokens[:, None])
                    if streamer is not None:
                        streamer.put(next_audio_tokens.cpu())
                    if model_kwargs["audio_out_ids"] is None or model_kwargs["audio_out_ids"].shape[0] == 0:
                        # Initialize audio_out_ids
                        model_kwargs["audio_out_ids"] = next_audio_tokens[:, None]
                        model_kwargs["audio_out_ids_start"] = torch.tensor(
                            [0], dtype=torch.long, device=input_ids.device
                        )
                    else:
                        model_kwargs["audio_out_ids_start"] = torch.concat(
                            [
                                model_kwargs["audio_out_ids_start"],
                                torch.tensor(
                                    [model_kwargs["audio_out_ids"].shape[1]], dtype=torch.long, device=input_ids.device
                                ),
                            ],
                            dim=0,
                        )
                        model_kwargs["audio_out_ids"] = torch.concat(
                            [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=1
                        )

            if return_dict_in_generate:
                if output_scores:
                    if is_audio_generation_mode:
                        scores += (next_audio_token_scores,)
                    else:
                        scores += (next_token_scores,)
                if output_logits:
                    if is_audio_generation_mode:
                        raw_logits += (next_audio_token_logits,)
                    else:
                        raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if "tokenizer_length" in generation_config.generation_kwargs:
                tokenizer_length = generation_config.generation_kwargs["tokenizer_length"]
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(
                        f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior."
                    )

            # update generated ids, model inputs, and length for next step
            if not is_audio_generation_mode or next_tokens[0] != self.audio_out_token_idx:
                # We only add one <|AUDIO_OUT|> token to the input_ids for simplicity.
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids_full, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioGenerationOutput(
                sequences=input_ids,
                audio_sequences=audio_sequences,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids, audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_features: Optional[torch.FloatTensor] = None,
        audio_feature_attention_mask: Optional[torch.BoolTensor] = None,
        audio_in_ids: Optional[torch.LongTensor] = None,
        audio_in_ids_start: Optional[torch.LongTensor] = None,
        audio_out_ids: Optional[torch.LongTensor] = None,
        audio_out_ids_start: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        The generate function in huggingface generally follows these steps:

        for sample_step in 1, 2, 3, 4, 5, ...
            ...

        """
        # Right now, it's a very simplified version of generate, we should revisit this after our model architecture stabilizes.
        assert input_ids.shape[0] == 1, (
            "Currently HiggsAudioModel.generate() only supports batch_size=1. See the implementation of "
        )
        generation_config, kwargs = self._prepare_generation_config(kwargs.pop("generation_config", None), **kwargs)
        if audio_out_bos_token_id is not None:
            generation_config.generation_kwargs["audio_out_bos_token_id"] = audio_out_bos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = self.audio_out_bos_token_id
            except:
                generation_config.generation_kwargs["audio_out_bos_token_id"] = None

        if audio_eos_token_id is not None:
            generation_config.generation_kwargs["audio_eos_token_id"] = audio_eos_token_id
        else:
            try:
                generation_config.generation_kwargs["audio_eos_token_id"] = self.audio_eos_token_id
            except:
                generation_config.generation_kwargs["audio_eos_token_id"] = None

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

        generation_config.generation_kwargs["ras_win_len"] = kwargs.pop("ras_win_len", None)
        generation_config.generation_kwargs["ras_win_max_num_repeat"] = kwargs.pop("ras_win_max_num_repeat", 2)
        # Set generation seed if determinstic generation is required
        if seed is not None:
            generation_config.generation_kwargs["seed"] = seed

        # Store tokenizer in generation config if it is in kwargs without popping it
        if "tokenizer" in kwargs:
            generation_config.generation_kwargs["tokenizer_length"] = len(kwargs["tokenizer"])

        # input_ids: [bsz, seq_len]
        # The merging of audio features happens inside the forward path. The input_ids does not need to change.
        # TODO: prepare the final input embeddings to improve generation performance
        input_ids_length = input_ids.shape[-1]
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=None,
            inputs_tensor=None,
            input_ids_length=input_ids_length,
        )
        assert generation_config.num_beams == 1, "Currently, we only support beam search with num_beams=1"
        return_dict_in_generate = generation_config.return_dict_in_generate
        output_scores = generation_config.output_scores

        # When attn_implement is spda or flash-attention, it will create causal mask automatically.
        attention_mask = kwargs.pop("attention_mask", None)
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            past_key_values=past_key_values,
            generation_config=generation_config,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            past_key_values_buckets=past_key_values_buckets,
            **kwargs,
        )

    def parameter_count_per_component(self):
        """Count the number of parameters per component in the model.

        HiggsAudio has the following main components:
            audio_tower: For mapping audio features to hidden states),
            llm_embed: The size of embedding layer of the LLM
            llm_non_embed: The size of non-embedding layer of the LLM
            audio_adapter: The overall size of additional layers for audio generation

        """
        trainable_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }
        total_stats = {
            "audio_tower": 0,
            "llm_embed": 0,
            "llm_non_embed": 0,
            "audio_embed": 0,
            "audio_adapter": 0,
            "overall": 0,
        }

        total_stats["overall"] = count_parameters(self, trainable_only=False)
        trainable_stats["overall"] = count_parameters(self, trainable_only=True)

        for mod in [self.audio_tower]:
            if mod is not None:
                total_stats["audio_tower"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_tower"] += count_parameters(mod, trainable_only=True)

        total_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=False)
        trainable_stats["llm_embed"] = count_parameters(self.embed_tokens, trainable_only=True)

        total_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=False)
        trainable_stats["audio_embed"] = count_parameters(self.audio_codebook_embeddings, trainable_only=True)

        # Calculate number of parameters for LLM
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                total_param_count = count_parameters(layer, trainable_only=False)
                total_trainable_param_count = count_parameters(layer, trainable_only=True)
                total_stats["llm_non_embed"] += total_param_count
                trainable_stats["llm_non_embed"] += total_trainable_param_count
                if not layer.fast_forward:
                    audio_mlp_param_count = count_parameters(layer.audio_mlp, trainable_only=False)
                    audio_mlp_trainable_param_count = count_parameters(layer.audio_mlp, trainable_only=True)

                    audio_norm_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=False
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=False)
                    audio_norm_trainable_param_count = count_parameters(
                        layer.audio_post_attention_layernorm, trainable_only=True
                    ) + count_parameters(layer.audio_input_layernorm, trainable_only=True)
                    total_stats["llm_non_embed"] -= audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["llm_non_embed"] -= (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )
                    total_stats["audio_adapter"] += audio_mlp_param_count + audio_norm_param_count
                    trainable_stats["audio_adapter"] += (
                        audio_mlp_trainable_param_count + audio_norm_trainable_param_count
                    )

                    if layer.use_audio_attention:
                        audio_attn_param_count = count_parameters(
                            layer.audio_attn, trainable_only=False
                        ) + count_parameters(layer.audio_post_audio_attn_layer_norm, trainable_only=False)
                        audio_attn_trainable_param_count = count_parameters(
                            layer.audio_attn, trainable_only=True
                        ) + count_parameters(layer.audio_post_audio_attn_layer_norm, trainable_only=True)
                        total_stats["llm_non_embed"] -= audio_attn_param_count
                        trainable_stats["llm_non_embed"] -= audio_attn_trainable_param_count
                        total_stats["audio_adapter"] += audio_attn_param_count
                        trainable_stats["audio_adapter"] += audio_attn_trainable_param_count
            else:
                total_stats["llm_non_embed"] += count_parameters(layer, trainable_only=False)
                trainable_stats["llm_non_embed"] += count_parameters(layer, trainable_only=True)
        total_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=False)
        trainable_stats["llm_non_embed"] += count_parameters(self.norm, trainable_only=True)

        total_stats["audio_adapter"] += count_parameters(self.audio_decoder_proj.audio_lm_head, trainable_only=False)
        trainable_stats["audio_adapter"] += count_parameters(
            self.audio_decoder_proj.audio_lm_head, trainable_only=True
        )
        total_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=False)
        trainable_stats["llm_embed"] += count_parameters(self.audio_decoder_proj.text_lm_head, trainable_only=True)

        other_audio_modules = [self.audio_encoder_proj]
        if self.use_audio_out_embed_projector:
            other_audio_modules.append(self.audio_out_embed_projector)

        for mod in other_audio_modules:
            if mod is not None:
                total_stats["audio_adapter"] += count_parameters(mod, trainable_only=False)
                trainable_stats["audio_adapter"] += count_parameters(mod, trainable_only=True)
        return {"trainable": trainable_stats, "total": total_stats}

    def set_skip_audio_tower(self):
        self.config.skip_audio_tower = True
        self.config.encode_whisper_embed = False

    def set_encode_audio_in_tokens(self):
        self.config.encode_audio_in_tokens = True

    def freeze_audio_tower(self):
        if self.audio_tower is not None:
            for param in self.audio_tower.parameters():
                param.requires_grad = False

    def freeze_audio_encoder_proj(self):
        if self.audio_encoder_proj is not None:
            for param in self.audio_encoder_proj.parameters():
                param.requires_grad = False

    def freeze_llm(self, freeze_embed=True, freeze_embed_until_idx: Optional[int] = None):
        for layer in self.layers:
            if isinstance(layer, HiggsAudioDualFFNDecoderLayer):
                for param in layer.self_attn.parameters():
                    param.requires_grad = False
                for param in layer.mlp.parameters():
                    param.requires_grad = False

                for param in layer.post_attention_layernorm.parameters():
                    param.requires_grad = False

                for param in layer.input_layernorm.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        for param in self.norm.parameters():
            param.requires_grad = False

        if freeze_embed:
            if freeze_embed_until_idx is None:
                for param in self.embed_tokens.parameters():
                    param.requires_grad = False
            else:
                assert isinstance(self.embed_tokens, nn.Embedding)
                self.embed_tokens = PartiallyFrozenEmbedding(
                    original_embedding=self.embed_tokens, freeze_until_idx=freeze_embed_until_idx
                )

    def freeze_text_head(self, freeze_text_head_until_idx: Optional[int] = None):
        """Freeze the final text head"""
        if freeze_text_head_until_idx is None:
            for param in self.audio_decoder_proj.text_lm_head.parameters():
                param.requires_grad = False

        else:
            assert isinstance(self.audio_decoder_proj.text_lm_head, nn.Linear)
            self.audio_decoder_proj.text_lm_head = PartiallyFrozenLinear(
                original_linear=self.audio_decoder_proj.text_lm_head, freeze_until_idx=freeze_text_head_until_idx
            )

    @classmethod
    def merge_weights_from_checkpoint(cls, checkpoint_dir: str, merged_output_dir: str, *model_args, **kwargs):
        # For users' convenience, we merge back embedding and text_lm_head if they are splitted
        splitted_model = super().from_pretrained(
            checkpoint_dir,
            *model_args,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            **{**kwargs, "state_dict": None},  # Prevent auto-loading state_dict
        )

        # Load all safetensor shards
        state_dict = {}
        shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))

        for shard_path in shard_paths:
            shard_dict = load_file(shard_path)  # Load each shard
            state_dict.update(shard_dict)  # Merge into a single dict

        # Merge weights
        if (
            "audio_decoder_proj.text_lm_head.linear_frozen.weight" in state_dict
            and "audio_decoder_proj.text_lm_head.linear_trainable.weight" in state_dict
        ):
            state_dict["audio_decoder_proj.text_lm_head.weight"] = torch.cat(
                [
                    state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"],
                    state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["audio_decoder_proj.text_lm_head.linear_frozen.weight"]
            del state_dict["audio_decoder_proj.text_lm_head.linear_trainable.weight"]

        if (
            "embed_tokens.embedding_frozen.weight" in state_dict
            and "embed_tokens.embedding_trainable.weight" in state_dict
        ):
            state_dict["embed_tokens.weight"] = torch.cat(
                [
                    state_dict["embed_tokens.embedding_frozen.weight"],
                    state_dict["embed_tokens.embedding_trainable.weight"],
                ],
                dim=0,
            )

            del state_dict["embed_tokens.embedding_frozen.weight"]
            del state_dict["embed_tokens.embedding_trainable.weight"]

        # Load the final state_dict
        splitted_model.load_state_dict(state_dict, strict=True)

        if merged_output_dir:
            splitted_model.save_pretrained(merged_output_dir, is_main_process=True, state_dict=state_dict)

    @torch.inference_mode()
    def capture_model(self, past_key_values: list[Union[Cache, List[torch.FloatTensor]]]) -> None:
        """Capture CUDA graphs for the model's forward pass with different KV cache lengths.

        Args:
            past_key_values: List of KV caches to capture graphs for
        """
        for past_key_value in past_key_values:
            kv_cache_length = past_key_value.get_max_cache_shape()
            # We capture two graphs, one for decoding audio tokens and one for decoding text tokens
            for is_decoding_audio_token in [True, False]:
                runner = CUDAGraphRunner(self._forward_core)

                # Create dummy inputs for graph capture
                batch_size = 1
                hidden_dim = self.config.hidden_size

                hidden_states = torch.zeros(
                    (batch_size, 1, hidden_dim), dtype=None, device=self.device
                )
                causal_mask = torch.ones(
                    (batch_size, 1, 1, kv_cache_length), dtype=None, device=self.device
                )
                position_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                audio_discrete_codes_mask = torch.tensor(
                    [[is_decoding_audio_token]], dtype=torch.bool, device=self.device
                )
                cache_position = torch.tensor([kv_cache_length - 1], dtype=torch.long, device=self.device)
                audio_attention_mask = torch.ones_like(causal_mask)
                fast_forward_attention_mask = torch.ones_like(causal_mask)

                runner.capture(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    position_ids=position_ids,
                    audio_discrete_codes_mask=audio_discrete_codes_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_value,
                    use_cache=True,
                    audio_attention_mask=audio_attention_mask,
                    fast_forward_attention_mask=fast_forward_attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    is_decoding_audio_token=is_decoding_audio_token,
                    is_using_cuda_graph=True,
                    #stream=torch.cuda.Stream(device=self.device),
                )

                self.decode_graph_runners[kv_cache_length][is_decoding_audio_token] = runner
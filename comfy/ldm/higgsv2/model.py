from __future__ import annotations

from comfy.text_encoders.llama import (
    RMSNorm, MLP, Attention, LlamaRoPE, Llama2Config
)

from comfy.autoregressive_sampling import GenerationConfig, apply_logits_processing, check_stopping_criteria
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.ldm.modules.attention import optimized_attention
from .cuda_graph_runner import CUDAGraphRunner
from .preprocess import _ceil_to_nearest

import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from typing import Optional, Tuple, Union, List

class GenerationMode(Enum):
    TEXT = 0
    AUDIO_INIT = 1
    AUDIO_IN_PROGRESS = 2

def _ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    is_training: bool = False,
) -> bool:

    _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
    key_value_length = query_length + past_key_values_length

    ignore_causal_mask = False

    if attention_mask is None:
        if (is_training and (query_length == 1 or key_value_length == query_length)):
            ignore_causal_mask = True
    elif sliding_window is None or key_value_length < sliding_window:
        if len(attention_mask.shape) == 4:
            return False
        elif torch.all(attention_mask == 1):
            if query_length == 1 or key_value_length == query_length:
                ignore_causal_mask = True

    return ignore_causal_mask

def categorical_sample(probs, generator = None):
    u = torch.rand((probs.size(0), 1), device = probs.device, generator = generator)
    cdf = probs.cumsum(dim = -1)
    return (u < cdf).float().argmax(dim = -1)

@dataclass
class HiggsAudioModelOutputWithPast:
    logits: Optional[torch.FloatTensor] = None
    audio_logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    audio_in_discrete_codes_mask: Optional[torch.BoolTensor] = None
    audio_out_mask: Optional[torch.BoolTensor] = None

@torch.jit.script
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
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

# refactored to decrease the length
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

        if embed.dtype != final_embedding.dtype:
            embed = embed.to(final_embedding.dtype)

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

class HiggsAudioDualFFNDecoderLayer(nn.Module):

    def __init__(
        self, config, llama_config, layer_idx: int, fast_forward: bool = False, device = None, dtype = None,
    ):
        super().__init__()
        text_config = config["text_config"]
        self.hidden_size = text_config["hidden_size"]
        self.layer_idx = layer_idx
        self.self_attn = Attention(config = llama_config, layer_idx = layer_idx, device = device, dtype = dtype)

        self.mlp = MLP(llama_config)

        if not fast_forward:
            self.audio_mlp = MLP(llama_config, device = device, dtype = dtype)
            self.audio_input_layernorm = RMSNorm(text_config["hidden_size"], eps = text_config["rms_norm_eps"], device = device, dtype = dtype)
            self.audio_post_attention_layernorm = RMSNorm(text_config["hidden_size"], eps = text_config["rms_norm_eps"], device = device, dtype = dtype)

        self.fast_forward = fast_forward

        self.input_layernorm = RMSNorm(text_config["hidden_size"], eps = text_config["rms_norm_eps"], device = device, dtype = dtype)
        self.post_attention_layernorm = RMSNorm(text_config["hidden_size"], eps = text_config["rms_norm_eps"], device = device, dtype = dtype)

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

        if is_decoding_audio_token and self.fast_forward:
            return (hidden_states,)

        has_audio_out = audio_out_mask is not None and audio_out_mask.shape[0] > 0

        audio_out_mask_sq = audio_out_mask

        small_input = target_length <= 2048
        optimized_attention = optimized_attention_for_device(hidden_states.device, small_input = small_input)

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
                attention_mask = attention_mask * ~audio_out_mask
            elif len(attention_mask.shape) == 4:

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

        hidden_states, present_key_value = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            freqs_cis = position_embeddings,
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

                mlp_dtype = next(iter(self.mlp.parameters())).dtype
                if text_hidden_states.dtype != mlp_dtype:
                    text_hidden_states = text_hidden_states.to(mlp_dtype)

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

class HiggsAudioDecoderProjector(nn.Module):

    def __init__(self, config, device = None, dtype = None, operations = None):
        super().__init__()

        self.text_lm_head = operations.Linear(config["text_config"]["hidden_size"], config["text_config"]["vocab_size"], bias=False, device = device, dtype = dtype)
        self.audio_lm_head = operations.Linear(
            config["text_config"]["hidden_size"], config["audio_num_codebooks"] * (config["audio_codebook_size"] + 2), bias=False, device = device, dtype = dtype
        )

    def forward(self, hidden_states, audio_out_mask, **kwargs):
        logits = self.text_lm_head(hidden_states)
        audio_logits = self.audio_lm_head(hidden_states[audio_out_mask])
        return logits, audio_logits

class HiggsAudioModel(nn.Module):

    _supports_cache_class = True
    _supports_static_cache = True

    def __init__(self, device = None, dtype = None, operations = None, **kwargs):
        super().__init__()

        self.padding_idx = kwargs["pad_token_id"]
        self.audio_in_token_idx = kwargs["audio_in_token_idx"]
        self.audio_out_token_idx = kwargs["audio_out_token_idx"]
        self.audio_out_bos_token_id = kwargs.get("audio_out_bos_token_id", None)
        self.audio_eos_token_id = kwargs.get("audio_eos_token_id", None)
        self.vocab_size = kwargs["text_config"]["vocab_size"]
        self.audio_num_codebooks = kwargs["audio_num_codebooks"]
        self.use_delay_pattern = kwargs["use_delay_pattern"]

        # for autoregressive sampling
        self.num_hidden_layers = kwargs["text_config"]["num_hidden_layers"]
        self.cache_config = kwargs["text_config"]
        self.hidden_dim = kwargs["text_config"]["hidden_size"]
        self.max_seq_len = kwargs["text_config"]["max_position_embeddings"]
        self.use_kv_buckets = kwargs.get("use_kv_buckets", False)

        self.dtype = dtype
        self.device = device
        self.config = kwargs

        self.generation_config = GenerationConfig.from_model_config(kwargs)
        self.generation_config.cache_implementation = self.cache_implementation = "static"

        self.audio_out_bos_token_id = 128013
        self.audio_eos_token_id = 128012

        text_config = kwargs["text_config"]
        llama_config = Llama2Config(num_attention_heads = text_config["num_attention_heads"],
                            num_key_value_heads = text_config["num_key_value_heads"],
                            hidden_size = text_config["hidden_size"],
                            head_dim = text_config["head_dim"],
                            qkv_bias = text_config["mlp_bias"],
                            intermediate_size = text_config["intermediate_size"])

        self.embed_tokens = operations.Embedding(self.vocab_size, kwargs["text_config"]["hidden_size"], self.padding_idx, device = device, dtype = dtype)
        self.attn_implementation = optimized_attention.__name__

        if kwargs["audio_adapter_type"] == "dual_ffn_fast_forward":
            layer_idx = 0
            layers = []
            for j in range(self.num_hidden_layers):
                if j in kwargs["audio_dual_ffn_layers"]:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(
                            kwargs,
                            llama_config,
                            layer_idx,
                            fast_forward=False,
                            device = device, dtype = dtype
                        )
                    )
                    layer_idx += 1
                else:
                    layers.append(
                        HiggsAudioDualFFNDecoderLayer(kwargs, llama_config, layer_idx, fast_forward=True, device = device, dtype = dtype)
                    )
                    layer_idx += 1
            self.layers = nn.ModuleList(layers)
        else:
            raise NotImplementedError(f"Audio adapter type {kwargs['audio_adapter_type']} not implemented.")

        self.num_activation_checkpointing_layers = len(self.layers)

        self.decode_graph_runners = defaultdict(dict[bool, CUDAGraphRunner])
        self.norm = RMSNorm(
            kwargs["text_config"]["hidden_size"], eps = kwargs["text_config"]["rms_norm_eps"]
        )
        self.rotary_emb = LlamaRoPE(config = llama_config)

        self.audio_tower = None
        self.audio_encoder_proj = None

        self.audio_decoder_proj = HiggsAudioDecoderProjector(
            kwargs, device=device, dtype=dtype, operations=operations
        )
        self.audio_codebook_size = kwargs["audio_codebook_size"] + 2

        self.audio_codebook_embeddings = operations.Embedding(
            kwargs["audio_num_codebooks"] * self.audio_codebook_size,
            kwargs["text_config"]["hidden_size"],
        )

        self.audio_codebook_weights = (
            torch.ones(kwargs["audio_num_codebooks"]) / kwargs["audio_num_codebooks"]
        )

    def _sample_audio_tokens(
        self,
        audio_logits: torch.Tensor,
        audio_out_ids: torch.Tensor,
        logits_processing_list,
        device: torch.device,
        torch_generator: Optional[torch.Generator],
        generation_config: GenerationConfig,
        num_delay: int,
        num_remaining_delays: Optional[int],
        is_using_cuda_graphs,
        do_sample = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[int]]:
        """Sample audio tokens and its corresponding text tokens from the logits"""

        ras_win_len = generation_config.generation_kwargs.get("ras_win_len", None)
        ras_win_max_num_repeat = generation_config.generation_kwargs.get("ras_win_max_num_repeat", 2)
        audio_eos_token_id = generation_config.generation_kwargs.get("audio_eos_token_id", None)

        next_audio_token_logits = audio_logits.clone()[-1, :, :].float().to(device)
        next_audio_token_scores = apply_logits_processing(next_audio_token_logits, logits_processing_list)

        if do_sample:
            probs = nn.functional.softmax(next_audio_token_scores, dim = -1)
            # torch.multinomial doesn't work with cuda graphs, replaced with mathematically eqv. fn
            if not is_using_cuda_graphs:
                next_audio_tokens = torch.multinomial(probs, num_samples = 1, generator=torch_generator).squeeze(1)
            else:
                next_audio_tokens = categorical_sample(probs, generator = torch_generator)
        else:
            next_audio_tokens = torch.argmax(next_audio_token_scores, dim=-1)

        if ras_win_len is not None:
            rep_num = (audio_out_ids[:, -ras_win_len:] == next_audio_tokens.unsqueeze(1)).sum(dim=1)

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
            self.config["audio_out_token_idx"],
            dtype=torch.long,
            device=device,
        )

        # Handle delay_pattern
        if self.use_delay_pattern:
            if num_delay + 1 < next_audio_tokens.shape[0]:
                next_audio_tokens[(num_delay + 1) :] = self.config["audio_stream_bos_id"]
                num_delay += 1
            if num_remaining_delays is not None:
                next_audio_tokens[: (self.audio_num_codebooks - num_remaining_delays)] = (
                    self.config["audio_stream_eos_id"]
                )
                num_remaining_delays -= 1
            else:
                all_eos_indices = (next_audio_tokens == self.config["audio_stream_eos_id"]).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    next_audio_tokens[:last_eos_idx] = self.config["audio_stream_eos_id"]
                    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1
            if num_remaining_delays is not None and num_remaining_delays <= 0:
                next_tokens[...] = audio_eos_token_id
                num_delay = 0
                num_remaining_delays = None

        return (
            next_tokens,
            next_audio_tokens,
            num_delay,
            num_remaining_delays,
        )

    def _sample_text_tokens(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processing_list,
        device: torch.device,
        generation_mode: GenerationMode,
    ) -> torch.Tensor:
        """Sample text tokens from the logits"""

        next_token_logits = logits.clone()[:, -1, :].float()
        next_token_logits = next_token_logits.to(input_ids.device)

        # pre-process distribution
        next_token_scores = apply_logits_processing(next_token_logits, logits_processing_list)

        if generation_mode == GenerationMode.AUDIO_INIT:
            # See the audio bos token, we should start generating audio tokens
            next_tokens = torch.full(
                (input_ids.shape[0],),
                self.audio_out_token_idx,
                dtype=torch.long,
                device=device,
            )
            next_audio_tokens = torch.full(
                (self.config["audio_num_codebooks"],),
                self.config["audio_stream_bos_id"],
                dtype=torch.long,
                device=device,
            )
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            next_audio_tokens = None

        return next_tokens, next_audio_tokens

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.attn_implementation == "attention_flash":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        if self.attn_implementation == "attention_pytorch" and not using_static_cache and not output_attentions:
            if _ignore_causal_mask_sdpa(
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
            if hasattr(past_key_values, "get_max_length"):
                target_length = past_key_values.get_max_length()
            else:
                target_length = past_key_values.max_cache_len
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

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
            self.attn_implementation == "attention_pytorch"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            causal_mask =  causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True))

        return causal_mask

    def _embed_audio_ids(self, audio_ids):
        codebook_shift = (
            torch.arange(self.config["audio_num_codebooks"], device=audio_ids.device) * self.audio_codebook_size
        )
        audio_embed = self.audio_codebook_embeddings(audio_ids + codebook_shift.unsqueeze(-1))
        audio_embed = torch.sum(audio_embed, dim=0)
        return audio_embed

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
        is_decoding_audio_token: Optional[bool] = None,
        is_using_cuda_graph: Optional[bool] = False,
    ):

        position_id_offset = cache_position[0] if use_cache else 0
        position_embeddings = self.rotary_emb(hidden_states, position_ids + position_id_offset)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                audio_attention_mask=audio_attention_mask,
                fast_forward_attention_mask=fast_forward_attention_mask,
                position_ids=position_ids,
                audio_out_mask=audio_discrete_codes_mask,
                is_decoding_audio_token=is_decoding_audio_token,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                is_using_cuda_graph=is_using_cuda_graph,
            )

            hidden_states = layer_outputs[0]

        return hidden_states

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
        is_using_cuda_graphs: bool = None,
        **kwargs
    ):

        target_device = input_ids.device

        inputs_embeds = self.embed_tokens(input_ids)

        if self.config["encode_audio_in_tokens"]:
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
            and is_using_cuda_graphs
        ):
            _forward_core = self.decode_graph_runners[past_key_values.get_max_cache_shape()][is_decoding_audio_token]
            local_cuda_graph = True
        else:
            _forward_core = self._forward_core
            local_cuda_graph = False

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
            is_using_cuda_graph = local_cuda_graph,
        )
        hidden_states = self.norm(hidden_states)

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

        return ret

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
    ):
        model_kwargs["past_key_values"] = outputs.past_key_values

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
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
        from_cache_size = from_cache.get_max_cache_shape()
        assert to_cache.get_max_cache_shape() >= from_cache_size, (
            f"The target cache size {to_cache.get_max_cache_shape()} is smaller than the source cache size {from_cache_size}."
        )

        n_layers = self.num_hidden_layers

        for i in range(n_layers):
            from_layer = from_cache.layers[i]
            to_layer = to_cache.layers[i]

            # lazy init
            if getattr(to_layer, "keys", None) is None:
                to_layer.keys = torch.zeros(
                    (self.cache_config.max_batch, self.cache_config.num_key_value_heads,
                     to_cache.get_max_cache_shape(),
                     self.cache_config.head_dim),
                    device=self.device, dtype=self.dtype
                )

            if getattr(to_layer, "values", None) is None:
                to_layer.values = torch.zeros(
                    (self.cache_config.max_batch, self.cache_config.num_key_value_heads,
                     to_cache.get_max_cache_shape(),
                     self.cache_config.head_dim),
                    device=self.device, dtype=self.dtype
                )

            seq_len = from_cache_size
            to_layer.keys[:, :, :seq_len, :] = from_layer.keys
            to_layer.values[:, :, :seq_len, :] = from_layer.values

    def _prepare_kv_cache(
        self,
        current_sequence_length: int,
        current_past_key_values_bucket: Optional[int],
        past_key_values_buckets: OrderedDict[int, Cache],
    ) -> Tuple[Optional[Cache], Optional[int]]:

        for cache_length in past_key_values_buckets.keys():
            if cache_length >= current_sequence_length:

                if current_past_key_values_bucket is not None and cache_length != current_past_key_values_bucket:
                    self._copy_kv_cache(
                        past_key_values_buckets[current_past_key_values_bucket], past_key_values_buckets[cache_length]
                    )

                return past_key_values_buckets[cache_length], cache_length

        raise ValueError(
            f"The current sequence length {current_sequence_length} is larger than "
            f"all past key values buckets {past_key_values_buckets.keys()}."
        )

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processing_list,
        generation_config: GenerationConfig,
        past_key_values_buckets: Optional[OrderedDict[int, Cache]],
        **model_kwargs,
    ):

        # code supports only non-mixed batchs

        audio_out_bos_token_id = generation_config.generation_kwargs.get("audio_out_bos_token_id", None)

        # torch generator for sampling
        torch_generator = model_kwargs.pop("torch_generator", None)
        # pbar for sampling
        pbar = model_kwargs.pop("pbar", None)

        # init values
        pad_token_id = generation_config.pad_token_id
        # Used to track which past_key_va
        self.current_past_key_values_bucket = None
        max_length = generation_config.max_length

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=input_ids.device)
        if generation_config.use_cache:
            model_kwargs["cache_audio_discrete_codes_mask"] = None

        do_sample = generation_config.do_sample
        is_using_cuda_graphs = generation_config.is_using_cuda_graphs

        init_model_input = True
        num_delay = 0
        num_remaining_delays = None
        audio_sequences = []
        # A tensor to keep track of all the audio placeholder tokens.
        input_ids_full = input_ids.clone()

        # Initialize the audio variables based on the input prompt.
        if input_ids[0][-1] == self.config["audio_out_token_idx"]:
            audio_sequences = [model_kwargs["audio_out_ids"][:, model_kwargs["audio_out_ids_start"][-1] :]]
            if self.use_delay_pattern:
                num_delay = (
                    self.audio_num_codebooks
                    - (model_kwargs["audio_out_ids"][:, -1] == self.config["audio_stream_bos_id"]).sum()
                )
                all_eos_indices = (model_kwargs["audio_out_ids"][:, -1] == self.config['audio_stream_eos_id']).nonzero()
                if torch.numel(all_eos_indices) > 0:
                    all_eos_indices = all_eos_indices[0]
                    last_eos_idx = all_eos_indices[-1]
                    num_remaining_delays = self.audio_num_codebooks - last_eos_idx - 1

        while not this_peer_finished:
            eos_token_tensor = torch.tensor([self.config["text_config"]["eos_token_id"]], device=input_ids.device)

            if input_ids[0][-1] == audio_out_bos_token_id:
                generation_mode = GenerationMode.AUDIO_INIT
            elif input_ids[0][-1] == self.audio_out_token_idx:
                generation_mode = GenerationMode.AUDIO_IN_PROGRESS
                eos_token_tensor = torch.tensor([self.config["audio_eos_token_id"]], device=input_ids.device)
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

            if past_key_values_buckets is not None:
                past_key_values, self.current_past_key_values_bucket = self._prepare_kv_cache(
                    cur_len, self.current_past_key_values_bucket, past_key_values_buckets
                )
                if past_key_values is not None:
                    model_inputs.update({"past_key_values": past_key_values})
                model_inputs["past_key_values_buckets"] = past_key_values_buckets

            outputs = self(**model_inputs, is_using_cuda_graphs = is_using_cuda_graphs, return_dict=True)

            # Update the actual sequence length after the first forward pass
            if init_model_input and past_key_values_buckets is not None:
                cur_len = past_key_values_buckets[self.current_past_key_values_bucket].get_seq_length().item()

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
            )

            init_model_input = False

            if this_peer_finished:
                continue

            if is_audio_generation_mode:
                (
                    next_tokens,
                    next_audio_tokens,
                    num_delay,
                    num_remaining_delays,
                ) = self._sample_audio_tokens(
                    audio_logits=outputs.audio_logits,
                    audio_out_ids=model_kwargs["audio_out_ids"],
                    logits_processing_list=logits_processing_list,
                    device=input_ids.device,
                    torch_generator=torch_generator,
                    generation_config=generation_config,
                    num_delay=num_delay,
                    num_remaining_delays=num_remaining_delays,
                    do_sample = do_sample,
                    is_using_cuda_graphs = is_using_cuda_graphs
                )

                # update generated ids, model inputs, and length for next step
                model_kwargs["audio_out_ids"] = torch.cat(
                    [model_kwargs["audio_out_ids"], next_audio_tokens[:, None]], dim=-1
                )
                audio_sequences[-1] = torch.cat([audio_sequences[-1], next_audio_tokens[:, None]], dim=-1)

            else:
                next_tokens, next_audio_tokens = self._sample_text_tokens(
                    input_ids=input_ids,
                    logits=outputs.logits,
                    logits_processing_list=logits_processing_list,
                    device=input_ids.device,
                    generation_mode=generation_mode,
                )

                if next_audio_tokens is not None:
                    audio_sequences.append(next_audio_tokens[:, None])
                    if model_kwargs["audio_out_ids"] is None or model_kwargs["audio_out_ids"].shape[0] == 0:
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

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences)

            if "tokenizer_length" in generation_config.generation_kwargs:
                tokenizer_length = generation_config.generation_kwargs["tokenizer_length"]
                if torch.max(next_tokens) >= tokenizer_length:
                    raise ValueError(
                        f"Next generated token has max value {torch.max(next_tokens)} which is greater than the tokenizer's vocabulary size {tokenizer_length}, this is undesired behavior."
                    )

            if pbar is not None:
                pbar.update(1)

            if not is_audio_generation_mode or next_tokens[0] != self.audio_out_token_idx:
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            input_ids_full = torch.cat([input_ids_full, next_tokens[:, None]], dim=-1)
            finished, unfinished_sequences = check_stopping_criteria(input_ids_full, max_length, eos_token = eos_token_tensor)
            this_peer_finished = finished.all()
            cur_len += 1

            del outputs
            torch.cuda.empty_cache()

        if pbar is not None:
            if pbar.total != pbar.current:
                pbar.update(pbar.total - pbar.current)

        return audio_sequences

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audio_out_bos_token_id: int = None,
        audio_eos_token_id: int = None,
        generation_config = None,
        generation_functions = None,
        **kwargs,
    ):

        if generation_config is None:
            generation_config = GenerationConfig()

        generation_config, kwargs = generation_functions._prepare_generation_config(generation_config, **kwargs)
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

        generation_config.generation_kwargs["ras_win_len"] = 7
        generation_config.generation_kwargs["ras_win_max_num_repeat"] = kwargs.pop("ras_win_max_num_repeat", 2)

        if "tokenizer" in kwargs:
            generation_config.generation_kwargs["tokenizer_length"] = len(kwargs["tokenizer"])

        input_ids_length = input_ids.shape[-1]
        generation_config = generation_functions._prepare_generated_length(
            generation_config=generation_config,
            input_ids_length=input_ids_length,
        )

        return generation_config

    @torch.inference_mode()
    def capture_model(self, past_key_values: list[Union[Cache, List[torch.FloatTensor]]]) -> None:
        for past_key_value in past_key_values:
            kv_cache_length = past_key_value.get_max_cache_shape()
            for is_decoding_audio_token in [True, False]:
                runner = CUDAGraphRunner(self._forward_core)

                batch_size = 1
                hidden_dim = self.config["hidden_size"]

                hidden_states = torch.zeros(
                    (batch_size, 1, hidden_dim), dtype=self.dtype, device=self.device
                )
                causal_mask = torch.ones(
                    (batch_size, 1, 1, kv_cache_length), dtype=self.dtype, device=self.device
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
                    is_decoding_audio_token=is_decoding_audio_token,
                    is_using_cuda_graph=True,
                )

                self.decode_graph_runners[kv_cache_length][is_decoding_audio_token] = runner

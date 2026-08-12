import dataclasses
import hashlib

import torch
from torch import nn

import comfy.model_management
import comfy.ops
import comfy.utils
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import Llama2_, Qwen3_8BConfig

from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS


CFG_SCALE = 1.5
CFG_TOP_K = 50
C0_VOCAB_SIZE = 16384
MAX_PROMPT_TOKENS = 5000
MAX_AUDIO_FRAMES = 9000


def derive_seed(seed, *parts):
    digest = hashlib.blake2b(digest_size=8, person=b"minimax-ttm")
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        value = str(part).encode("utf-8")
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def sample_topk(logits, generator):
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9)
    threshold = torch.topk(values, 50, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probabilities = torch.nan_to_num(torch.softmax(values, dim=-1), nan=0.0)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


class RVQAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dtype, device, operations):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        batch, length, hidden_size = x.shape
        q = self.q_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        mask = torch.full((length, length), torch.finfo(q.dtype).min, device=q.device, dtype=q.dtype).triu_(1)
        attention = optimized_attention_for_device(q.device, mask=True, small_input=True)
        out = attention(q, k, v, self.num_heads, mask=mask, skip_reshape=True)
        return self.o_proj(out)


class RVQRMSNorm(nn.Module):
    def __init__(self, hidden_size, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), comfy.ops.cast_to_input(self.weight, x), 1e-6)


class RVQMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dtype, device, operations):
        super().__init__()
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RVQDecoderBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dtype, device, operations):
        super().__init__()
        self.input_layernorm = RVQRMSNorm(hidden_size, dtype, device)
        self.self_attn = RVQAttention(hidden_size, num_heads, dtype, device, operations)
        self.post_attention_layernorm = RVQRMSNorm(hidden_size, dtype, device)
        self.mlp = RVQMLP(hidden_size, intermediate_size, dtype, device, operations)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class RVQDepthDecoder(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        audio_vocab_size = int(config["audio_vocab_size"])
        num_codebooks = int(config["audio_num_codebooks"])
        self.projection = operations.Linear(hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.pos_embedding = operations.Embedding(16, hidden_size, dtype=dtype, device=device)
        self.audio_heads = nn.ModuleList([
            operations.Linear(hidden_size, audio_vocab_size, bias=False, dtype=dtype, device=device)
            for _ in range(num_codebooks - 1)
        ])
        self.layers = nn.ModuleList([
            RVQDecoderBlock(
                hidden_size,
                int(config["decoder_num_heads"]),
                int(config["decoder_intermediate_size"]),
                dtype,
                device,
                operations,
            )
            for _ in range(int(config["decoder_num_layers"]))
        ])
        self.norm = RVQRMSNorm(hidden_size, dtype, device)

    def forward(self, sequence):
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        x = sequence + self.pos_embedding(positions, out_dtype=sequence.dtype).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MiniMaxMusic3AR(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        config_fields = {field.name for field in dataclasses.fields(Qwen3_8BConfig)}
        qwen_config = Qwen3_8BConfig(**{key: value for key, value in config.items() if key in config_fields})
        qwen_config.lm_head = False
        self.model = Llama2_(qwen_config, device=device, dtype=dtype, ops=operations)
        self.model.lm_head = operations.Linear(qwen_config.hidden_size, qwen_config.vocab_size, bias=False, dtype=dtype, device=device)
        self.model.audio_extra_embedding = operations.Embedding(
            int(config["audio_vocab_size"]) * (int(config["audio_num_codebooks"]) - 1),
            qwen_config.hidden_size,
            dtype=dtype,
            device=device,
        )
        self.model.audio_decoder = RVQDepthDecoder(config, dtype, device, operations)
        self.audio_vocab_size = int(config["audio_vocab_size"])
        self.num_codebooks = int(config["audio_num_codebooks"])
        self.embedding_scale = self.num_codebooks ** -0.5

    def _init_kv_cache(self, batch, length, device, execution_dtype):
        config = self.model.config
        return [
            (
                torch.empty((batch, config.num_key_value_heads, length, config.head_dim), device=device, dtype=execution_dtype),
                torch.empty((batch, config.num_key_value_heads, length, config.head_dim), device=device, dtype=execution_dtype),
                0,
            )
            for _ in range(config.num_hidden_layers)
        ]

    def _guided_c0(self, logits, cfg_scale, top_k):
        conditioned = logits[0:1].float()
        unconditioned = logits[1:2].float()
        guided = unconditioned + (conditioned - unconditioned) * cfg_scale
        threshold = torch.topk(conditioned, top_k, dim=-1).values[..., -1, None]
        return guided.masked_fill(conditioned < threshold, -float("inf"))

    def _depth_codes(self, hidden, c0, generator, execution_dtype, cfg_scale):
        decoder = self.model.audio_decoder
        sequence = [decoder.projection(hidden).unsqueeze(1)]
        c0_embed = self.model.embed_tokens(c0 + AUDIO_CODE_OFFSET, out_dtype=execution_dtype)
        sequence.append(decoder.projection(c0_embed).unsqueeze(1))
        codes = [c0]
        hidden_parts = []
        for index in range(1, self.num_codebooks):
            out = decoder(torch.cat(sequence, dim=1))[:, -1]
            hidden_parts.append(out[:1].detach())
            logits = decoder.audio_heads[index - 1](out)
            conditioned = logits[:1].float()
            unconditioned = logits[1:2].float()
            code = sample_topk(unconditioned + (conditioned - unconditioned) * cfg_scale, generator).repeat(2)
            codes.append(code)
            if index < self.num_codebooks - 1:
                embedding = self.model.audio_extra_embedding(
                    code + (index - 1) * self.audio_vocab_size,
                    out_dtype=execution_dtype,
                )
                sequence.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)

    def _embed_audio_frame(self, codes, execution_dtype):
        c0 = self.model.embed_tokens(codes[:, 0] + AUDIO_CODE_OFFSET, out_dtype=execution_dtype)
        offsets = torch.arange(self.num_codebooks - 1, device=codes.device) * self.audio_vocab_size
        extra = self.model.audio_extra_embedding(codes[:, 1:] + offsets.unsqueeze(0), out_dtype=execution_dtype).sum(dim=1)
        return ((c0 + extra) * self.embedding_scale).unsqueeze(1)

    def generate(self, input_ids, seed, max_audio_frames, device, cfg_scale=CFG_SCALE, top_k=CFG_TOP_K):
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens > MAX_PROMPT_TOKENS:
            raise ValueError(f"MiniMax Music3 prompt has {prompt_tokens} tokens; maximum is {MAX_PROMPT_TOKENS}")

        input_ids = input_ids.to(device)
        if comfy.model_management.should_use_bf16(device):
            execution_dtype = torch.bfloat16
        else:
            execution_dtype = torch.float32
        unconditioned = input_ids.clone()
        unconditioned[:, 1:-2] = SPECIAL_TOKEN_IDS["<|audio_cfg|>"]
        text_ids = torch.cat((input_ids, unconditioned), dim=0)
        text_embeds = self.model.embed_tokens(text_ids, out_dtype=execution_dtype)
        decode_limit = min(int(max_audio_frames), MAX_AUDIO_FRAMES)
        past = self._init_kv_cache(2, prompt_tokens + decode_limit + 1, device, execution_dtype)
        output = self.model(None, embeds=text_embeds, past_key_values=past, dtype=execution_dtype)
        last_hidden = output[0][:, -1]
        past = output[2]

        generator = torch.Generator(device=device).manual_seed(derive_seed(seed, "ar"))
        hidden_frames = []
        progress = comfy.utils.ProgressBar(decode_limit)
        stop_token = SPECIAL_TOKEN_IDS["<|audio_end|>"]

        for frame_index in comfy.utils.model_trange(decode_limit + 1, desc="AR sampling"):
            comfy.model_management.throw_exception_if_processing_interrupted()
            logits = self.model.lm_head(last_hidden).float()
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[:, AUDIO_CODE_OFFSET:AUDIO_CODE_OFFSET + C0_VOCAB_SIZE] = False
            mask[:, stop_token] = False
            logits = logits.masked_fill(mask, -float("inf"))
            guided = self._guided_c0(logits, cfg_scale, top_k).masked_fill(mask[:1], -float("inf"))
            code_or_stop = sample_topk(guided, generator)
            if int(code_or_stop.item()) == stop_token:
                break

            c0 = code_or_stop - AUDIO_CODE_OFFSET
            feedback_codes, depth_hidden = self._depth_codes(last_hidden, c0.repeat(2), generator, execution_dtype, cfg_scale)
            frame_hidden = torch.cat((last_hidden[:1].detach(), depth_hidden), dim=-1)
            if frame_index > 0:
                hidden_frames.append(frame_hidden[0].to(device="cpu", copy=True))
                progress.update_absolute(len(hidden_frames))
                if len(hidden_frames) >= decode_limit:
                    break

            feedback = self._embed_audio_frame(feedback_codes, execution_dtype)
            output = self.model(None, embeds=feedback, past_key_values=past, dtype=execution_dtype)
            last_hidden = output[0][:, -1]
            past = output[2]

        if not hidden_frames:
            raise ValueError("MiniMax Music3 generated zero audio frames")
        return torch.stack(hidden_frames)

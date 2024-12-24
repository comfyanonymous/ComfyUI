import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Any

from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.model_management
import comfy.ldm.common_dit

import comfy.model_management

@dataclass
class Llama2Config:
    vocab_size: int = 128320
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        return comfy.ldm.common_dit.rms_norm(x, self.weight, self.eps)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def precompute_freqs_cis(head_dim, seq_len, theta, device=None):
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (theta_numerator / head_dim))

    position_ids = torch.arange(0, seq_len, device=device).unsqueeze(0)

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return (cos, sin)


def apply_rope(xq, xk, freqs_cis):
    cos = freqs_cis[0].unsqueeze(1)
    sin = freqs_cis[1].unsqueeze(1)
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads

        ops = ops or nn
        self.q_proj = ops.Linear(config.hidden_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, device=device, dtype=dtype)
        self.o_proj = ops.Linear(config.hidden_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output)

class MLP(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        ops = ops or nn
        self.gate_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ops.Linear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ops.Linear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
    ):
        # Self Attention
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
        )
        x = residual + x

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x

class Llama2_(nn.Module):
    def __init__(self, config, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = ops.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype
        )
        self.layers = nn.ModuleList([
            TransformerBlock(config, device=device, dtype=dtype, ops=ops)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        # self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        x = self.embed_tokens(x, out_dtype=dtype)

        freqs_cis = precompute_freqs_cis(self.config.hidden_size // self.config.num_attention_heads,
                                         x.shape[1],
                                         self.config.rope_theta,
                                         device=x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), float("-inf"))

        causal_mask = torch.empty(x.shape[1], x.shape[1], dtype=x.dtype, device=x.device).fill_(float("-inf")).triu_(1)
        if mask is not None:
            mask += causal_mask
        else:
            mask = causal_mask
        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        intermediate = None
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
            )
            if i == intermediate_output:
                intermediate = x.clone()

        x = self.norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)

        return x, intermediate


class Llama2(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Llama2Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def forward(self, input_ids, *args, **kwargs):
        return self.model(input_ids, *args, **kwargs)

import torch
import torch.nn as nn
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
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = False

@dataclass
class Qwen25_3BConfig:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True

@dataclass
class Gemma2_2B_Config:
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    transformer_type: str = "gemma2"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, add=False, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.add = add

    def forward(self, x: torch.Tensor):
        w = self.weight
        if self.add:
            w = w + 1.0

        return comfy.ldm.common_dit.rms_norm(x, w, self.eps)



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

        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        ops = ops or nn
        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

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
        if config.mlp_activation == "silu":
            self.activation = torch.nn.functional.silu
        elif config.mlp_activation == "gelu_pytorch_tanh":
            self.activation = lambda a: torch.nn.functional.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))

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

class TransformerBlockGemma2(nn.Module):
    def __init__(self, config: Llama2Config, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

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

        x = self.post_attention_layernorm(x)
        x = residual + x

        # MLP
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
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
        if self.config.transformer_type == "gemma2":
            transformer = TransformerBlockGemma2
            self.normalize_in = True
        else:
            transformer = TransformerBlock
            self.normalize_in = False

        self.layers = nn.ModuleList([
            transformer(config, device=device, dtype=dtype, ops=ops)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        # self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        if self.normalize_in:
            x *= self.config.hidden_size ** 0.5

        freqs_cis = precompute_freqs_cis(self.config.head_dim,
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
        all_intermediate = None
        if intermediate_output is not None:
            if intermediate_output == "all":
                all_intermediate = []
                intermediate_output = None
            elif intermediate_output < 0:
                intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                all_intermediate.append(x.unsqueeze(1).clone())
            x = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
            )
            if i == intermediate_output:
                intermediate = x.clone()

        x = self.norm(x)
        if all_intermediate is not None:
            all_intermediate.append(x.unsqueeze(1).clone())

        if all_intermediate is not None:
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)

        return x, intermediate

class BaseLlama:
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.embed_tokens = embeddings

    def forward(self, input_ids, *args, **kwargs):
        return self.model(input_ids, *args, **kwargs)


class Llama2(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Llama2Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Qwen25_3B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_3BConfig(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma2_2B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma2_2B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

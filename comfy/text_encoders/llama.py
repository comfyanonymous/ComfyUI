import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Any
import math
import logging

from comfy.ldm.modules.attention import optimized_attention_for_device
import comfy.model_management
import comfy.ldm.common_dit

import comfy.model_management
from . import qwen_vl

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
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None

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
    rope_dims = None
    q_norm = None
    k_norm = None
    rope_scale = None

@dataclass
class Qwen25_7BVLI_Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]
    q_norm = None
    k_norm = None
    rope_scale = None

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
    rope_dims = None
    q_norm = None
    k_norm = None
    sliding_attention = None
    rope_scale = None

@dataclass
class Gemma3_4B_Config:
    vocab_size: int = 262208
    hidden_size: int = 2560
    intermediate_size: int = 10240
    num_hidden_layers: int = 34
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta = [10000.0, 1000000.0]
    transformer_type: str = "gemma3"
    head_dim = 256
    rms_norm_add = True
    mlp_activation = "gelu_pytorch_tanh"
    qkv_bias = False
    rope_dims = None
    q_norm = "gemma3"
    k_norm = "gemma3"
    sliding_attention = [False, False, False, False, False, 1024]
    rope_scale = [1.0, 8.0]

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


def precompute_freqs_cis(head_dim, position_ids, theta, rope_scale=None, rope_dims=None, device=None):
    if not isinstance(theta, list):
        theta = [theta]

    out = []
    for index, t in enumerate(theta):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (t ** (theta_numerator / head_dim))

        if rope_scale is not None:
            if isinstance(rope_scale, list):
                inv_freq /= rope_scale[index]
            else:
                inv_freq /= rope_scale

        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        if rope_dims is not None and position_ids.shape[0] > 1:
            mrope_section = rope_dims * 2
            cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
            sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        out.append((cos, sin))

    if len(out) == 1:
        return out[0]

    return out


def apply_rope(xq, xk, freqs_cis):
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed.to(org_dtype), k_embed.to(org_dtype)


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

        self.q_norm = None
        self.k_norm = None

        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

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

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

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
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
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
    def __init__(self, config: Llama2Config, index, device=None, dtype=None, ops: Any = None):
        super().__init__()
        self.self_attn = Attention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

        if config.sliding_attention is not None:  # TODO: implement. (Not that necessary since models are trained on less than 1024 tokens)
            self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]
        else:
            self.sliding_attention = False

        self.transformer_type = config.transformer_type

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        optimized_attention=None,
    ):
        if self.transformer_type == 'gemma3':
            if self.sliding_attention:
                if x.shape[1] > self.sliding_attention:
                    logging.warning("Warning: sliding attention not implemented, results may be incorrect")
                freqs_cis = freqs_cis[1]
            else:
                freqs_cis = freqs_cis[0]

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
        if self.config.transformer_type == "gemma2" or self.config.transformer_type == "gemma3":
            transformer = TransformerBlockGemma2
            self.normalize_in = True
        else:
            transformer = TransformerBlock
            self.normalize_in = False

        self.layers = nn.ModuleList([
            transformer(config, index=i, device=device, dtype=dtype, ops=ops)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        # self.lm_head = ops.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, position_ids=None, embeds_info=[]):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        if self.normalize_in:
            x *= self.config.hidden_size ** 0.5

        if position_ids is None:
            position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)

        freqs_cis = precompute_freqs_cis(self.config.head_dim,
                                         position_ids,
                                         self.config.rope_theta,
                                         self.config.rope_scale,
                                         self.config.rope_dims,
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

class Qwen25_7BVLI(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen25_7BVLI_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.visual = qwen_vl.Qwen2VLVisionTransformer(hidden_size=1280, output_hidden_size=config.hidden_size, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    def preprocess_embed(self, embed, device):
        if embed["type"] == "image":
            image, grid = qwen_vl.process_qwen2vl_images(embed["data"])
            return self.visual(image.to(device, dtype=torch.float32), grid), grid
        return None, None

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None, embeds_info=[]):
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") == "image":
                grid = e.get("extra", None)
                start = e.get("index")
                if position_ids is None:
                    position_ids = torch.zeros((3, embeds.shape[1]), device=embeds.device)
                    position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
                end = e.get("size") + start
                len_max = int(grid.max()) // 2
                start_next = len_max + start
                position_ids[:, end:] = torch.arange(start_next + offset, start_next + (embeds.shape[1] - end) + offset, device=embeds.device)
                position_ids[0, start:end] = start + offset
                max_d = int(grid[0][1]) // 2
                position_ids[1, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
                max_d = int(grid[0][2]) // 2
                position_ids[2, start:end] = torch.arange(start + offset, start + max_d + offset, device=embeds.device).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
                offset += len_max - (end - start)

        if grid is None:
            position_ids = None

        return super().forward(x, attention_mask=attention_mask, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=final_layer_norm_intermediate, dtype=dtype, position_ids=position_ids)

class Gemma2_2B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma2_2B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

class Gemma3_4B(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Gemma3_4B_Config(**config_dict)
        self.num_layers = config.num_hidden_layers

        self.model = Llama2_(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

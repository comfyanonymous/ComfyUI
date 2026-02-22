import math
import torch
import torch.nn as nn

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.image_encoders.dino2 import LayerScale as DINOv3ViTLayerScale

class DINOv3ViTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias, device, dtype, operations):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, device=device, dtype=dtype)
        self.down_proj = operations.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias, device=device, dtype=dtype)
        self.act_fn = torch.nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(q, k, cos, sin, **kwargs):
    num_tokens = q.shape[-2]
    num_patches = sin.shape[-2]
    num_prefix_tokens = num_tokens - num_patches

    q_prefix_tokens, q_patches = q.split((num_prefix_tokens, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((num_prefix_tokens, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)

    return q, k

class DINOv3ViTAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, device, dtype, operations):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.k_proj = operations.Linear(self.embed_dim, self.embed_dim, bias=False, device=device, dtype=dtype) # key_bias = False
        self.v_proj = operations.Linear(self.embed_dim, self.embed_dim, bias=True, device=device, dtype=dtype)

        self.q_proj = operations.Linear(self.embed_dim, self.embed_dim, bias=True, device=device, dtype=dtype)
        self.o_proj = operations.Linear(self.embed_dim, self.embed_dim, bias=True, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        batch_size, patches, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn = optimized_attention_for_device(query_states.device, mask=False)

        attn_output = attn(
            query_states, key_states, value_states, self.num_heads, attention_mask, skip_reshape=True, skip_output_reshape=True
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

class DINOv3ViTGatedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias, device, dtype, operations):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, device=device, dtype=dtype)
        self.up_proj = operations.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias, device=device, dtype=dtype)
        self.down_proj = operations.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias, device=device, dtype=dtype)
        self.act_fn = torch.nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def get_patches_center_coordinates(
    num_patches_h: int, num_patches_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:

    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    coords = 2.0 * coords - 1.0
    return coords

class DINOv3ViTRopePositionEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, rope_theta, hidden_size, num_attention_heads, image_size, patch_size, device, dtype):
        super().__init__()
        self.base = rope_theta
        self.head_dim = hidden_size // num_attention_heads
        self.num_patches_h = image_size // patch_size
        self.num_patches_w = image_size // patch_size
        self.patch_size = patch_size

        inv_freq = 1 / self.base ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        device = pixel_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.amp.autocast(device_type = device_type, enabled=False):
            patch_coords = get_patches_center_coordinates(
                num_patches_h, num_patches_w, dtype=torch.float32, device=device
            )

            self.inv_freq = self.inv_freq.to(device)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
            angles = angles.flatten(1, 2)
            angles = angles.tile(2)

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        dtype = pixel_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class DINOv3ViTEmbeddings(nn.Module):
    def __init__(self, hidden_size, num_register_tokens, num_channels, patch_size, dtype, device, operations):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size, device=device, dtype=dtype))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size, device=device, dtype=dtype))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, hidden_size, device=device, dtype=dtype))
        self.patch_embeddings = operations.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype
        )

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor | None = None):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.weight.dtype

        patch_embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)

        if bool_masked_pos is not None:
            mask_token = self.mask_token.to(patch_embeddings.dtype)
            patch_embeddings = torch.where(bool_masked_pos.unsqueeze(-1), mask_token, patch_embeddings)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        device = patch_embeddings.device
        cls_token = cls_token.to(device)
        register_tokens = register_tokens.to(device)
        embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)

        return embeddings

class DINOv3ViTLayer(nn.Module):

    def __init__(self, hidden_size, layer_norm_eps, use_gated_mlp, mlp_bias, intermediate_size, num_attention_heads,
                 device, dtype, operations):
        super().__init__()

        self.norm1 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.attention = DINOv3ViTAttention(hidden_size, num_attention_heads, device=device, dtype=dtype, operations=operations)
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size, device=device, dtype=dtype, operations=None)

        self.norm2 = operations.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)

        if use_gated_mlp:
            self.mlp = DINOv3ViTGatedMLP(hidden_size, intermediate_size, mlp_bias, device=device, dtype=dtype, operations=operations)
        else:
            self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size=intermediate_size, mlp_bias=mlp_bias, device=device, dtype=dtype, operations=operations)
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size, device=device, dtype=dtype, operations=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.layer_scale1(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.layer_scale2(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class DINOv3ViTModel(nn.Module):
    def __init__(self, config, dtype, device, operations):
        super().__init__()
        if dtype == torch.float16 and comfy.model_management.should_use_bf16(device, prioritize_performance=False):
            dtype = torch.bfloat16
        num_hidden_layers = config["num_hidden_layers"]
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
        num_register_tokens = config["num_register_tokens"]
        intermediate_size = config["intermediate_size"]
        layer_norm_eps = config["layer_norm_eps"]
        num_channels = config["num_channels"]
        patch_size = config["patch_size"]
        rope_theta = config["rope_theta"]

        self.embeddings = DINOv3ViTEmbeddings(
            hidden_size, num_register_tokens, num_channels=num_channels, patch_size=patch_size, dtype=dtype, device=device, operations=operations
        )
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(
            rope_theta, hidden_size, num_attention_heads, image_size=512, patch_size=patch_size, dtype=dtype, device=device
        )
        self.layer = nn.ModuleList(
            [DINOv3ViTLayer(hidden_size, layer_norm_eps, use_gated_mlp=False, mlp_bias=True,
                            intermediate_size=intermediate_size,num_attention_heads = num_attention_heads,
                            dtype=dtype, device=device, operations=operations)
            for _ in range(num_hidden_layers)])
        self.norm = operations.LayerNorm(hidden_size, eps=layer_norm_eps, dtype=dtype, device=device)

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        **kwargs,
    ):

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        position_embeddings = self.rope_embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )

        norm = self.norm.to(hidden_states.device)
        sequence_output = norm(hidden_states)
        pooled_output = sequence_output[:, 0, :]

        return sequence_output, None, pooled_output, None

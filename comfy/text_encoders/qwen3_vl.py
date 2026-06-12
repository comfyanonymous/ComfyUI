"""Generic Qwen3-VL multimodal stack.

Sibling of `comfy.text_encoders.qwen_vl` (which only ships the Qwen2-VL vision
tower). Qwen3-VL differs from Qwen2-VL in: full attention vision blocks,
GELU MLP via `linear_fc{1,2}`, LayerNorm (not RMSNorm), learned `pos_embed`,
and a deepstack-merger contract that additively injects intermediate vision
features into specific decoder layers at visual-token positions.

Public exports:
  - `Qwen3VLConfig`               — dataclass for the Qwen3-VL text decoder
  - `Qwen3VLVisionConfig`         — dataclass for the Qwen3-VL vision tower
  - `Qwen3VLVisionModel`          — vision tower; forward returns
                                    `(image_features, deepstack_features)`
  - `Qwen3VLDecoder`              — forked Llama2-style decoder with per-layer
                                    deepstack residual injection
  - `Qwen3VLBase`                 — outer wrapper holding `model.{language_model,
                                    visual}` plus root `lm_head` to bijectively
                                    match a `model.*` / `lm_head` checkpoint
  - `process_qwen3vl_image`       — preprocess one (1, H, W, C) image in [0,1]
                                    into (flatten_patches, grid_thw)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.llama import (
    MLP,
    RMSNorm,
    apply_rope,
    precompute_freqs_cis,
)


# Defaults track the JoyImageEdit checkpoint (text_encoder/config.json) but the
# class is intended for any Qwen3-VL deployment; override fields as needed.
@dataclass
class Qwen3VLConfig:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    transformer_type: str = "llama"
    head_dim: int = 128
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    rope_dims: Tuple[int, int, int] = (24, 20, 20)
    interleaved_mrope: bool = True
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = True
    stop_tokens: Tuple[int, int] = (151643, 151645)
    # Decoder layer indices that receive deepstack residuals from the vision
    # tower. transformers' `Qwen3VLTextModel` injects merger outputs after
    # decoder layers ``range(len(deepstack_visual_embeds))`` — i.e. after the
    # first 3 layers (0, 1, 2) for the standard 3-merger setup, regardless of
    # the vision-side ``deepstack_visual_indexes=[8, 16, 24]``. The decoder
    # injection layers and the vision tap layers are distinct concepts; they
    # share the count (3) but not the indices.
    deepstack_decoder_inject_layers: Tuple[int, ...] = (0, 1, 2)


@dataclass
class Qwen3VLVisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    out_hidden_size: int = 4096
    num_heads: int = 16
    depth: int = 27
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: Tuple[int, ...] = (8, 16, 24)
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    min_pixels: int = 65536
    max_pixels: int = 16777216


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def process_qwen3vl_image(
    image: torch.Tensor,
    min_pixels: int = 65536,
    max_pixels: int = 16777216,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
):
    """Resize, normalize and patch-flatten a single (B=1, H, W, C) image tensor in [0, 1].

    Returns ``(flatten_patches, grid_thw)`` ready for `Qwen3VLVisionModel.forward`.
    Mirrors `Qwen2VLImageProcessorFast` (used by the Qwen3VLProcessor): bucket
    size to a multiple of ``patch_size*merge_size``, clamp by min/max pixels,
    bicubic resize, normalize by mean/std, then unfold into temporal*spatial
    patches using a single-frame temporal repeat.
    """
    if image_mean is None:
        image_mean = [0.5, 0.5, 0.5]
    if image_std is None:
        image_std = [0.5, 0.5, 0.5]

    if image.dim() == 3:
        image = image.unsqueeze(0)
    batch, height, width, channels = image.shape
    if batch != 1:
        raise ValueError("process_qwen3vl_image expects one image (B=1) at a time.")
    device = image.device

    image = image.permute(0, 3, 1, 2)  # (1, C, H, W)
    img = image[0]

    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    img_resized = F.interpolate(
        img.unsqueeze(0), size=(h_bar, w_bar), mode="bicubic", align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)

    normalized = img_resized.clone()
    for c in range(3):
        normalized[c] = (img_resized[c] - image_mean[c]) / image_std[c]

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.long)

    # Single-frame inputs are duplicated along T to fill the 2-frame temporal
    # patch kernel; matches Qwen2VLImageProcessorFast for static images.
    pixel_values = normalized.unsqueeze(0).repeat(temporal_patch_size, 1, 1, 1)
    grid_t = 1
    channel = pixel_values.shape[1]
    patches = pixel_values.reshape(
        grid_t, temporal_patch_size, channel,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flatten_patches, grid_thw


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------

class _Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, hidden_size, patch_size, temporal_patch_size, in_channels=3,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = hidden_size
        self.proj = ops.Conv3d(
            in_channels, hidden_size,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=True, device=device, dtype=dtype,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size,
        )
        hidden_states = self.proj(hidden_states)
        return hidden_states.view(-1, self.embed_dim)


class _Qwen3VLVisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None, ops=None):
        super().__init__()
        self.linear_fc1 = ops.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class _Qwen3VLVisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = ops.Linear(hidden_size, hidden_size * 3, bias=True, device=device, dtype=dtype)
        self.proj = ops.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)

    def forward(self, hidden_states, position_embeddings, cu_seqlens, optimized_attention):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 0, 2, 3).unbind(0)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(-2).float()
        sin = sin.unsqueeze(-2).float()
        q_orig_dtype = q.dtype
        q_f = q.float()
        k_f = k.float()
        q_rot = torch.cat((-q_f[..., q_f.shape[-1] // 2:], q_f[..., : q_f.shape[-1] // 2]), dim=-1)
        k_rot = torch.cat((-k_f[..., k_f.shape[-1] // 2:], k_f[..., : k_f.shape[-1] // 2]), dim=-1)
        q = ((q_f * cos) + (q_rot * sin)).to(q_orig_dtype)
        k = ((k_f * cos) + (k_rot * sin)).to(q_orig_dtype)

        q = q.transpose(0, 1).unsqueeze(0)  # (1, H, S, D)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # Per-image full attention: split by cu_seqlens and run independently.
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
        outs = [optimized_attention(qq, kk, vv, self.num_heads, skip_reshape=True) for qq, kk, vv in zip(*splits)]
        out = torch.cat(outs, dim=1)
        out = out.reshape(seq_length, -1)
        return self.proj(out)


class _Qwen3VLVisionBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, device=None, dtype=None, ops=None):
        super().__init__()
        self.norm1 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.norm2 = ops.LayerNorm(hidden_size, eps=1e-6, device=device, dtype=dtype)
        self.attn = _Qwen3VLVisionAttention(hidden_size, num_heads, device=device, dtype=dtype, ops=ops)
        self.mlp = _Qwen3VLVisionMLP(hidden_size, intermediate_size, device=device, dtype=dtype, ops=ops)

    def forward(self, hidden_states, position_embeddings, cu_seqlens, optimized_attention):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), position_embeddings, cu_seqlens, optimized_attention,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class _Qwen3VLPatchMerger(nn.Module):
    def __init__(self, hidden_size, out_hidden_size, spatial_merge_size,
                 use_postshuffle_norm, device=None, dtype=None, ops=None):
        super().__init__()
        merged_size = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = merged_size if use_postshuffle_norm else hidden_size
        self.norm = ops.LayerNorm(norm_dim, eps=1e-6, device=device, dtype=dtype)
        self.linear_fc1 = ops.Linear(merged_size, merged_size, bias=True, device=device, dtype=dtype)
        self.linear_fc2 = ops.Linear(merged_size, out_hidden_size, bias=True, device=device, dtype=dtype)
        self.merged_size = merged_size

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        x = self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="none"))
        return x


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision tower.

    forward returns ``(image_features, deepstack_features)`` where
    ``image_features`` is the merger output ``(N_merged, out_hidden_size)`` and
    ``deepstack_features`` is a list of per-merger outputs (same shape) — one
    per index in ``deepstack_visual_indexes``. The caller is responsible for
    additively injecting each ``deepstack_features[k]`` into language-model
    hidden states at the matching layer at visual-token positions.
    """

    def __init__(self, config: Optional[Qwen3VLVisionConfig] = None,
                 device=None, dtype=None, ops=None, **kwargs):
        super().__init__()
        if config is None:
            config = Qwen3VLVisionConfig(**kwargs)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.head_dim = config.hidden_size // config.num_heads
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)

        self.patch_embed = _Qwen3VLVisionPatchEmbed(
            config.hidden_size, config.patch_size, config.temporal_patch_size, in_channels=3,
            device=device, dtype=dtype, ops=ops,
        )
        self.pos_embed = ops.Embedding(config.num_position_embeddings, config.hidden_size,
                                       device=device, dtype=dtype)
        self.blocks = nn.ModuleList([
            _Qwen3VLVisionBlock(config.hidden_size, config.intermediate_size, config.num_heads,
                                device=device, dtype=dtype, ops=ops)
            for _ in range(config.depth)
        ])
        self.merger = _Qwen3VLPatchMerger(
            config.hidden_size, config.out_hidden_size, config.spatial_merge_size,
            use_postshuffle_norm=False, device=device, dtype=dtype, ops=ops,
        )
        self.deepstack_merger_list = nn.ModuleList([
            _Qwen3VLPatchMerger(
                config.hidden_size, config.out_hidden_size, config.spatial_merge_size,
                use_postshuffle_norm=True, device=device, dtype=dtype, ops=ops,
            ) for _ in range(len(self.deepstack_visual_indexes))
        ])

    def _rotary_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        device = self.pos_embed.weight.device
        dim = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        seq = torch.arange(max_hw, device=device, dtype=inv_freq.dtype)
        freq_table = torch.outer(seq, inv_freq)

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra = torch.arange(merge_size, device=device)
            row_idx = (block_rows[:, None, None, None] * merge_size + intra[None, None, :, None]).expand(
                merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = (block_cols[None, :, None, None] * merge_size + intra[None, None, None, :]).expand(
                merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            n = coords.shape[0]
            pos_ids[offset: offset + n] = coords
            offset += n
        return freq_table[pos_ids].flatten(1)

    def _fast_pos_embed_interpolate(self, grid_thw):
        # Bilinear interpolation over the learned `pos_embed` grid into the
        # actual (grid_h, grid_w) requested by this image.
        grid_thw_list = grid_thw.tolist()
        device = self.pos_embed.weight.device
        idx_lists = [[] for _ in range(4)]
        weight_lists = [[] for _ in range(4)]
        grid_hs = [r[1] for r in grid_thw_list]
        grid_ws = [r[2] for r in grid_thw_list]
        grid_ts = [r[0] for r in grid_thw_list]
        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            hf = h_idxs.int()
            wf = w_idxs.int()
            hc = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            wc = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - hf
            dw = w_idxs - wf
            base_h = hf * self.num_grid_per_side
            base_h_ceil = hc * self.num_grid_per_side
            indices = [
                (base_h[None].T + wf[None]).flatten(),
                (base_h[None].T + wc[None]).flatten(),
                (base_h_ceil[None].T + wf[None]).flatten(),
                (base_h_ceil[None].T + wc[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_lists[i].extend(indices[i].tolist())
                weight_lists[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_lists, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_lists, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos = patch_pos.split([h * w for h, w in zip(grid_hs, grid_ws)])
        out = []
        merge_size = self.spatial_merge_size
        for pe, t, h, w in zip(patch_pos, grid_ts, grid_hs, grid_ws):
            pe = pe.repeat(t, 1)
            pe = (pe.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                  .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            out.append(pe)
        return torch.cat(out)

    def forward(self, pixel_values, grid_thw):
        optimized_attention = optimized_attention_for_device(pixel_values.device, mask=False, small_input=True)
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)

        rotary_pos_emb = self._rotary_pos_emb(grid_thw).to(hidden_states.device)
        seq_len = hidden_states.size(0)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        deepstack_features: List[torch.Tensor] = []
        deepstack_set = set(self.deepstack_visual_indexes)
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, position_embeddings, cu_seqlens, optimized_attention)
            if layer_num in deepstack_set:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](hidden_states))

        if len(deepstack_features) != len(self.deepstack_visual_indexes):
            raise RuntimeError(
                f"Qwen3VLVisionModel: produced {len(deepstack_features)} deepstack features "
                f"but configured for {len(self.deepstack_visual_indexes)}; "
                f"deepstack_visual_indexes={self.deepstack_visual_indexes} contained an "
                f"out-of-range layer."
            )

        image_features = self.merger(hidden_states)
        return image_features, deepstack_features


# ---------------------------------------------------------------------------
# Decoder (forked from Llama2_) with deepstack residual injection
# ---------------------------------------------------------------------------

class _Qwen3VLAttention(nn.Module):
    """Qwen3-VL self-attention. Equivalent to `comfy.text_encoders.llama.Attention`
    with `q_norm/k_norm = "gemma3"` and `qkv_bias = False`; forked here only so
    that `Qwen3VLDecoder` does not depend on the private `Attention` symbol of
    `llama.py` (which is intentionally not part of its public surface).
    """

    def __init__(self, config: Qwen3VLConfig, device=None, dtype=None, ops=None):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim

        self.q_proj = ops.Linear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ops.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ops.Linear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)

        if config.q_norm == "gemma3":
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.q_norm = None
        if config.k_norm == "gemma3":
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        else:
            self.k_norm = None

    def forward(self, hidden_states, attention_mask, freqs_cis, optimized_attention):
        batch_size, seq_length, _ = hidden_states.shape

        xq = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        xk = xk.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        xv = xv.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        output = optimized_attention(xq, xk, xv, self.num_heads, mask=attention_mask, skip_reshape=True)
        return self.o_proj(output)


class _Qwen3VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLConfig, device=None, dtype=None, ops=None):
        super().__init__()
        self.self_attn = _Qwen3VLAttention(config, device=device, dtype=dtype, ops=ops)
        self.mlp = MLP(config, device=device, dtype=dtype, ops=ops)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device, dtype=dtype)

    def forward(self, x, attention_mask, freqs_cis, optimized_attention):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            hidden_states=x,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            optimized_attention=optimized_attention,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen3VLDecoder(nn.Module):
    """Forked Llama2-style decoder for Qwen3-VL.

    Constructor surface is compatible with `comfy.text_encoders.llama.Llama2_`
    (config dataclass + ``device/dtype/ops``). Forward signature additionally
    accepts ``deepstack_residuals`` and ``deepstack_layer_indices`` to enable
    the Qwen3-VL deepstack injection that vanilla `Llama2_` does not support.

    Deepstack contract:
      ``deepstack_residuals`` is a list of full-sequence tensors, each of shape
      ``(B, seq_len, hidden_size)``, with **zeros at non-visual positions** and
      the corresponding ``deepstack_merger_list[k]`` output at visual-token
      positions. Index ``k`` in ``deepstack_residuals`` is added into the
      hidden state **after decoder layer**
      ``deepstack_layer_indices[k]`` runs (matching transformers'
      ``Qwen3VLTextModel`` semantics). Lengths of the two lists must match;
      indices must be in ``[0, num_hidden_layers)``. Mismatch raises.
    """

    def __init__(self, config: Qwen3VLConfig, device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = ops.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            _Qwen3VLDecoderLayer(config, device=device, dtype=dtype, ops=ops)
            for _ in range(config.num_hidden_layers)
        ])

        if config.final_norm:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add,
                                device=device, dtype=dtype)
        else:
            self.norm = None

    def compute_freqs_cis(self, position_ids, device):
        return precompute_freqs_cis(
            self.config.head_dim,
            position_ids,
            self.config.rope_theta,
            self.config.rope_scale,
            list(self.config.rope_dims) if self.config.rope_dims is not None else None,
            interleaved_mrope=getattr(self.config, "interleaved_mrope", False),
            device=device,
        )

    def forward(
        self,
        x,
        attention_mask=None,
        embeds=None,
        num_tokens=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None,
        position_ids=None,
        embeds_info=(),
        deepstack_residuals=None,
        deepstack_layer_indices=None,
        # Forward-compat with `Llama2_.forward` signature; not used here
        # (this fork doesn't implement KV-cache generation).
        past_key_values=None,
        input_ids=None,
    ):
        if embeds is not None:
            x = embeds
        else:
            x = self.embed_tokens(x, out_dtype=dtype)

        seq_len = x.shape[1]

        # Validate deepstack arguments up front. No silent fallbacks.
        if deepstack_residuals is not None or deepstack_layer_indices is not None:
            if deepstack_residuals is None or deepstack_layer_indices is None:
                raise ValueError(
                    "Qwen3VLDecoder.forward: deepstack_residuals and "
                    "deepstack_layer_indices must be supplied together "
                    f"(got residuals={'set' if deepstack_residuals is not None else 'None'}, "
                    f"indices={'set' if deepstack_layer_indices is not None else 'None'})."
                )
            if len(deepstack_residuals) != len(deepstack_layer_indices):
                raise ValueError(
                    f"Qwen3VLDecoder.forward: deepstack_residuals has length "
                    f"{len(deepstack_residuals)} but deepstack_layer_indices has length "
                    f"{len(deepstack_layer_indices)}; the two must match 1:1."
                )
            for k, idx in enumerate(deepstack_layer_indices):
                if not (0 <= idx < len(self.layers)):
                    raise ValueError(
                        f"Qwen3VLDecoder.forward: deepstack_layer_indices[{k}]={idx} "
                        f"out of range for {len(self.layers)} decoder layers."
                    )
                r = deepstack_residuals[k]
                if r.shape[0] != x.shape[0] or r.shape[1] != seq_len or r.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"Qwen3VLDecoder.forward: deepstack_residuals[{k}].shape={tuple(r.shape)} "
                        f"does not match (B, seq_len, hidden_size)={tuple(x.shape)}."
                    )
            inject_at = {int(layer_idx): k for k, layer_idx in enumerate(deepstack_layer_indices)}
        else:
            inject_at = {}

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        freqs_cis = self.compute_freqs_cis(position_ids, x.device)

        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(
                attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1])
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=x.dtype, device=x.device).fill_(
                torch.finfo(x.dtype).min / 4).triu_(1)
            if mask is not None:
                mask += causal_mask
            else:
                mask = causal_mask

        optimized_attention = optimized_attention_for_device(x.device, mask=mask is not None, small_input=True)

        intermediate = None
        all_intermediate = None
        only_layers = None
        resolved_intermediate_output = intermediate_output
        if intermediate_output is not None:
            if isinstance(intermediate_output, list):
                all_intermediate = []
                only_layers = set(intermediate_output)
            elif intermediate_output == "all":
                all_intermediate = []
                resolved_intermediate_output = None
            elif intermediate_output < 0:
                resolved_intermediate_output = len(self.layers) + intermediate_output

        for i, layer in enumerate(self.layers):
            if all_intermediate is not None:
                if only_layers is None or (i in only_layers):
                    all_intermediate.append(x.unsqueeze(1).clone())

            x = layer(
                x=x,
                attention_mask=mask,
                freqs_cis=freqs_cis,
                optimized_attention=optimized_attention,
            )

            if i == resolved_intermediate_output:
                intermediate = x.clone()

            if i in inject_at:
                # Additive injection at visual-token positions; non-visual
                # positions in the residual tensor are zero. Applied AFTER
                # the decoder layer.
                x = x + deepstack_residuals[inject_at[i]].to(dtype=x.dtype)

        if self.norm is not None:
            x = self.norm(x)

        if all_intermediate is not None:
            if only_layers is None or ((len(self.layers)) in only_layers):
                all_intermediate.append(x.unsqueeze(1).clone())
            intermediate = torch.cat(all_intermediate, dim=1)

        if intermediate is not None and final_layer_norm_intermediate and self.norm is not None:
            intermediate = self.norm(intermediate)

        return x, intermediate


# ---------------------------------------------------------------------------
# Outer wrapper
# ---------------------------------------------------------------------------

class _Qwen3VLInnerModel(nn.Module):
    """Holds ``language_model`` and ``visual`` so checkpoint keys match the
    ``model.language_model.*`` / ``model.visual.*`` namespace produced by
    ``Qwen3VLForConditionalGeneration``.
    """

    def __init__(self, config: Qwen3VLConfig, vision_config: Qwen3VLVisionConfig,
                 device=None, dtype=None, ops=None):
        super().__init__()
        self.config = config
        self.language_model = Qwen3VLDecoder(config, device=device, dtype=dtype, ops=ops)
        self.visual = Qwen3VLVisionModel(vision_config, device=device, dtype=dtype, ops=ops)

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    def forward(self, *args, **kwargs):
        return self.language_model.forward(*args, **kwargs)


class Qwen3VLBase(torch.nn.Module):
    """Generic Qwen3-VL multimodal stack with the
    ``model.{language_model,visual}`` + root ``lm_head`` namespace.

    Subclasses are expected to plug in 3D MRoPE position-id construction (for
    image-token blocks) by overriding ``forward`` or
    ``build_image_position_ids`` to consume the ``embeds_info`` list produced
    by ``comfy.sd1_clip.SDClipModel.process_tokens``. Plain text-only callers
    can use ``forward`` directly.
    """

    def __init__(self, config_dict, dtype, device, operations,
                 config_cls=Qwen3VLConfig, vision_config_cls=Qwen3VLVisionConfig,
                 vision_config_dict: Optional[dict] = None):
        super().__init__()
        config = config_cls(**config_dict)
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.dtype = dtype

        if vision_config_dict is None:
            vision_config = vision_config_cls()
        else:
            vision_config = vision_config_cls(**vision_config_dict)

        if len(vision_config.deepstack_visual_indexes) != len(config.deepstack_decoder_inject_layers):
            raise ValueError(
                f"Qwen3VLBase: vision_config has "
                f"{len(vision_config.deepstack_visual_indexes)} deepstack mergers "
                f"but text config has {len(config.deepstack_decoder_inject_layers)} "
                f"deepstack injection layers; lengths must match."
            )

        self.model = _Qwen3VLInnerModel(config, vision_config, device=device, dtype=dtype, ops=operations)
        # `lm_head` lives at the root of a Qwen3VLForConditionalGeneration
        # checkpoint. Required for clean state-dict loading even when callers
        # only use the encoder for hidden states.
        if config.lm_head:
            self.lm_head = operations.Linear(
                config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype,
            )

    # --- Public surface mirroring `comfy.text_encoders.llama.BaseLlama` ----

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.model.language_model.embed_tokens = embeddings

    # --- Vision / preprocessing -----------------------------------------------

    def preprocess_embed(self, embed, device):
        """Run the vision tower for one ``{"type": "image", "data": tensor}``
        embed and return ``(merged_features, extra)`` where ``extra`` is a
        dict ``{"grid": grid_thw, "deepstack": deepstack_features}``. The
        ``deepstack`` list has one tensor per
        ``vision_config.deepstack_visual_indexes`` entry, each of shape
        ``(N_merged, hidden_size)`` — same shape as ``merged_features``.
        """
        if embed["type"] != "image":
            return None, None
        pixel_values, grid_thw = process_qwen3vl_image(embed["data"])
        pixel_values = pixel_values.to(device, dtype=torch.float32)
        grid_thw = grid_thw.to(device)
        merged, deepstack = self.model.visual(pixel_values, grid_thw)
        return merged, {"grid": grid_thw, "deepstack": deepstack}

    # --- Position ids ---------------------------------------------------------

    def build_position_ids(self, embeds, attention_mask, embeds_info):
        """Build the (3, seq_len) MRoPE position-id matrix for an embed sequence
        that may contain image-token blocks. Mirrors
        `comfy.text_encoders.llama.Qwen25_7BVLI.forward`'s position-id logic
        but reads ``grid`` from ``e["extra"]["grid"]`` rather than
        ``e["extra"]`` directly.
        """
        grid = None
        position_ids = None
        offset = 0
        for e in embeds_info:
            if e.get("type") != "image":
                continue
            extra = e.get("extra", None)
            if not isinstance(extra, dict) or "grid" not in extra:
                raise ValueError(
                    "Qwen3VLBase.build_position_ids: image embed extra is missing 'grid'."
                )
            grid = extra["grid"]
            start = e.get("index")
            if position_ids is None:
                position_ids = torch.ones((3, embeds.shape[1]), device=embeds.device, dtype=torch.long)
                position_ids[:, :start] = torch.arange(0, start, device=embeds.device)
            end = e.get("size") + start
            len_max = int(grid.max()) // 2
            start_next = len_max + start
            if attention_mask is not None:
                after_mask = attention_mask[0, end:]
                text_positions = after_mask.cumsum(0) - 1 + start_next + offset
                position_ids[:, end:] = torch.where(
                    after_mask.bool(), text_positions, position_ids[0, end:],
                )
            else:
                position_ids[:, end:] = torch.arange(
                    start_next + offset, start_next + (embeds.shape[1] - end) + offset,
                    device=embeds.device,
                )
            position_ids[0, start:end] = start + offset
            max_d = int(grid[0][1]) // 2
            position_ids[1, start:end] = torch.arange(
                start + offset, start + max_d + offset, device=embeds.device,
            ).unsqueeze(1).repeat(1, math.ceil((end - start) / max_d)).flatten(0)[:end - start]
            max_d = int(grid[0][2]) // 2
            position_ids[2, start:end] = torch.arange(
                start + offset, start + max_d + offset, device=embeds.device,
            ).unsqueeze(0).repeat(math.ceil((end - start) / max_d), 1).flatten(0)[:end - start]
            offset += len_max - (end - start)

        return position_ids if grid is not None else None

    # --- Deepstack residual construction --------------------------------------

    def build_deepstack_residuals(self, embeds, embeds_info):
        """Construct the per-merger zero-padded residual tensors that
        `Qwen3VLDecoder.forward` expects. Returns
        ``(residuals, layer_indices)`` or ``(None, None)`` if no images are
        present in the sequence.

        Each residual has shape ``(B, seq_len, hidden_size)``, with the
        corresponding deepstack feature placed at visual-token positions and
        zeros elsewhere. If multiple images share one batch, all of them
        contribute residuals in order.
        """
        num_mergers = len(self.config.deepstack_decoder_inject_layers)
        any_image = any(e.get("type") == "image" for e in embeds_info)
        if not any_image:
            return None, None

        B, seq_len, hidden_size = embeds.shape
        residuals = [
            torch.zeros((B, seq_len, hidden_size), device=embeds.device, dtype=embeds.dtype)
            for _ in range(num_mergers)
        ]
        for e in embeds_info:
            if e.get("type") != "image":
                continue
            extra = e.get("extra", None)
            if not isinstance(extra, dict) or "deepstack" not in extra:
                raise ValueError(
                    "Qwen3VLBase.build_deepstack_residuals: image embed extra is missing 'deepstack'."
                )
            ds_features = extra["deepstack"]
            if len(ds_features) != num_mergers:
                raise ValueError(
                    f"Qwen3VLBase.build_deepstack_residuals: expected {num_mergers} deepstack "
                    f"features per image but got {len(ds_features)}."
                )
            start = e.get("index")
            size = e.get("size")
            for k, feat in enumerate(ds_features):
                if feat.shape[0] != size:
                    raise ValueError(
                        f"Qwen3VLBase.build_deepstack_residuals: deepstack feature #{k} has "
                        f"{feat.shape[0]} tokens but image embed claims {size} positions."
                    )
                residuals[k][:, start:start + size, :] = feat.to(dtype=embeds.dtype).unsqueeze(0)

        return residuals, list(self.config.deepstack_decoder_inject_layers)

    # --- Forward --------------------------------------------------------------

    def forward(self, x, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=()):
        position_ids = self.build_position_ids(embeds, attention_mask, embeds_info) if embeds is not None else None
        deepstack_residuals, deepstack_layer_indices = (
            self.build_deepstack_residuals(embeds, embeds_info) if embeds is not None else (None, None)
        )
        return self.model(
            x,
            attention_mask=attention_mask,
            embeds=embeds,
            num_tokens=num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype,
            position_ids=position_ids,
            deepstack_residuals=deepstack_residuals,
            deepstack_layer_indices=deepstack_layer_indices,
        )

# DINOv3 ViT-H+ backbone for SAM 3D Body.
#
# Single-file consolidation of the inference path. SAM 3D Body only ships a
# `dinov3_vith16plus` checkpoint, so the architecture is hardcoded rather
# than reconstructed from Hydra-flavoured configs.
#
# Adapted from facebookresearch/dinov3 (DINOv3 License Agreement). Trimmed
# to what's actually exercised at inference: no multi-crop training path,
# no DINOHead, no causal blocks, no rmsnorm/Mlp variants, no rope shift /
# jitter / rescale (training-time augmentations).

#TODO: Unify with TRELLIS2

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention
from torch import Tensor, nn

# DINOv3 ViT-H+ architecture constants.
EMBED_DIM = 1280
DEPTH = 32
NUM_HEADS = 20
FFN_RATIO = 6.0
PATCH_SIZE = 16
LAYERSCALE_INIT = 1.0e-5
N_STORAGE_TOKENS = 4
LAYERNORM_EPS = 1e-5  # "layernormbf16" preset uses 1e-5
ROPE_BASE = 100.0

# RoPE (axial sin/cos, no learnable weights)

def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    return x * cos + _rotate_half(x) * sin


class RopePositionEmbedding(nn.Module):
    """Axial RoPE for 2D patch grids; periods buffer is deterministic."""

    def __init__(self, embed_dim: int, num_heads: int, dtype=torch.float32, device=None):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        D_head = embed_dim // num_heads
        # Periods are persistent so they round-trip through state_dict, but the
        # values are deterministic from D_head/base; load_state_dict will
        # overwrite this with the saved buffer either way.
        periods = ROPE_BASE ** (
            2 * torch.arange(D_head // 4, dtype=dtype, device=device) / (D_head // 2)
        )
        self.register_buffer("periods", periods, persistent=True)
        self._dtype = dtype

    def forward(self, H: int, W: int) -> Tuple[Tensor, Tensor]:
        device, dtype = self.periods.device, self._dtype
        coords_h = torch.arange(0.5, H, device=device, dtype=dtype) / H
        coords_w = torch.arange(0.5, W, device=device, dtype=dtype) / W
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0  # [HW, 2] in [-1, +1]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)  # [HW, D_head]
        return torch.sin(angles), torch.cos(angles)


def _apply_rope_to_qk(q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]):
    """Apply RoPE only to the patch-token slice (skip CLS + storage tokens)."""
    sin, cos = rope
    rope_dtype = sin.dtype
    q_dtype, k_dtype = q.dtype, k.dtype
    q = q.to(rope_dtype)
    k = k.to(rope_dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    q_pre, q_rope = q[..., :prefix, :], q[..., prefix:, :]
    k_pre, k_rope = k[..., :prefix, :], k[..., prefix:, :]
    q = torch.cat([q_pre, _apply_rope(q_rope, sin, cos)], dim=-2)
    k = torch.cat([k_pre, _apply_rope(k_rope, sin, cos)], dim=-2)
    return q.to(q_dtype), k.to(k_dtype)

# Layers

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float, device=None, dtype=None):
        super().__init__()
        self.gamma = nn.Parameter(
            torch.full((dim,), init_values, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SwiGLUFFN(nn.Module):
    """w3(silu(w1(x)) * w2(x))."""

    def __init__(self, in_features: int, hidden_features: int, align_to: int = 8,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        d = int(hidden_features * 2 / 3)
        h = d + (-d % align_to)
        self.w1 = ops.Linear(in_features, h, bias=True, device=device, dtype=dtype)
        self.w2 = ops.Linear(in_features, h, bias=True, device=device, dtype=dtype)
        self.w3 = ops.Linear(h, in_features, bias=True, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        self.num_heads = num_heads
        # DINOv3's `mask_k_bias` zeroes the K third of qkv.bias. The mask is
        # deterministic from out_features, so the loader applies it in-place
        # once after load_state_dict (see `apply_dinov3_qkv_bias_mask`) and the
        # forward stays a plain F.linear.
        self.qkv = ops.Linear(dim, dim * 3, bias=True, device=device, dtype=dtype)
        self.proj = ops.Linear(dim, dim, bias=True, device=device, dtype=dtype)

    def forward(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = _apply_rope_to_qk(q, k, rope)
        # low_precision_attention=False forces attention_sage (when enabled
        # globally in comfy) to fall back to pytorch SDPA. SAM 3D Body's
        # regression heads (camera projection, MHR rig math) are sensitive
        # to attention output precision; sage's int8/fp8 path drifts the
        # keypoints and mesh visibly.
        x = optimized_attention(
            q, k, v, self.num_heads, skip_reshape=True,
            low_precision_attention=False,
        )
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        self.norm1 = ops.LayerNorm(dim, eps=LAYERNORM_EPS, device=device, dtype=dtype)
        self.attn = SelfAttention(dim, num_heads, device=device, dtype=dtype, operations=operations)
        self.ls1 = LayerScale(dim, LAYERSCALE_INIT, device=device, dtype=dtype)
        self.norm2 = ops.LayerNorm(dim, eps=LAYERNORM_EPS, device=device, dtype=dtype)
        self.mlp = SwiGLUFFN(dim, int(dim * ffn_ratio), device=device, dtype=dtype, operations=operations)
        self.ls2 = LayerScale(dim, LAYERSCALE_INIT, device=device, dtype=dtype)

    def forward(self, x: Tensor, rope=None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE,
                 device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        self.proj = ops.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            device=device, dtype=dtype,
        )

# Encoder + wrapper

class _DinoEncoder(nn.Module):
    """Inner ViT module. Held under `Dinov3Backbone.encoder` so state_dict
    keys (`backbone.encoder.*`) match the upstream layout."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        ops = operations if operations is not None else nn
        self.patch_size = PATCH_SIZE
        self.embed_dim = EMBED_DIM

        self.patch_embed = PatchEmbed(
            embed_dim=EMBED_DIM, patch_size=PATCH_SIZE,
            device=device, dtype=dtype, operations=operations,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, EMBED_DIM, device=device, dtype=dtype))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, N_STORAGE_TOKENS, EMBED_DIM, device=device, dtype=dtype)
        )
        # The released config sets pos_embed_rope_dtype="fp32"; periods stays
        # in fp32 regardless of the backbone weight dtype.
        self.rope_embed = RopePositionEmbedding(EMBED_DIM, NUM_HEADS, dtype=torch.float32, device=device)

        self.blocks = nn.ModuleList([
            Block(EMBED_DIM, NUM_HEADS, FFN_RATIO, device=device, dtype=dtype, operations=operations)
            for _ in range(DEPTH)
        ])
        self.norm = ops.LayerNorm(EMBED_DIM, eps=LAYERNORM_EPS, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed.proj(x)  # (B, embed_dim, H, W)
        B, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # Prepend CLS + storage tokens.
        x = torch.cat([
            self.cls_token.expand(B, -1, -1),
            self.storage_tokens.expand(B, -1, -1),
            x,
        ], dim=1)

        rope = self.rope_embed(H=H, W=W)
        for blk in self.blocks:
            x = blk(x, rope)
        x = self.norm(x)

        # Drop CLS + storage tokens; reshape patch grid to (B, C, H, W).
        x = x[:, 1 + N_STORAGE_TOKENS :]
        return x.reshape(B, H, W, EMBED_DIM).permute(0, 3, 1, 2).contiguous()


class Dinov3Backbone(nn.Module):
    """Public backbone interface used by SAM3DBody."""

    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.encoder = _DinoEncoder(device=device, dtype=dtype, operations=operations)
        self.patch_size = PATCH_SIZE
        self.embed_dim = self.embed_dims = EMBED_DIM

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


def apply_dinov3_qkv_bias_mask(backbone: "Dinov3Backbone") -> None:
    """Zero the K third of every block's qkv.bias in-place.

    Implements DINOv3's `mask_k_bias` once at load time so the per-block forward
    stays a plain F.linear instead of cloning + slicing the bias every call.
    """
    for blk in backbone.encoder.blocks:
        qkv = blk.attn.qkv
        if qkv.bias is not None:
            o = qkv.out_features
            qkv.bias.data[o // 3 : 2 * o // 3] = 0

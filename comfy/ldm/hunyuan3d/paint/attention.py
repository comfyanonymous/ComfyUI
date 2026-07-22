# Torch-native implementation of the Hunyuan3D 2.1 paint (hunyuan3d-paintpbr-v2-1)
# multiview attention stack: reimplements the architecture the released checkpoint
# was trained with (reference attention, material-dimension self attention, and 3D
# PoseRoPE) without a diffusers/xformers dependency. Module and parameter names
# follow the checkpoint's state_dict layout so the released weights load directly.
# The 1D rotary-embedding tables follow diffusers' (Apache-2.0) implementation,
# which the reference model also uses.

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Rotary position embedding utilities (PoseRoPE)
# ---------------------------------------------------------------------------
def get_1d_rotary_pos_embed(dim: int, pos: torch.Tensor, theta: float = 10000.0):
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device)[: (dim // 2)] / dim))
    freqs = torch.outer(pos, freqs)
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()
    return freqs_cos, freqs_sin


def get_3d_rotary_pos_embed(position: torch.Tensor, embed_dim: int, voxel_resolution: int, theta: int = 10000):
    """Per-token 3D RoPE tables. PoseRoPE splits the head dim 3:3:2 over x/y/z."""
    assert position.shape[-1] == 3
    axis_dims = (embed_dim // 8 * 3, embed_dim // 8 * 3, embed_dim // 8 * 2)

    grid = torch.arange(voxel_resolution, dtype=torch.float32, device=position.device)
    tables = {dim: get_1d_rotary_pos_embed(dim, grid, theta=theta) for dim in set(axis_dims)}

    flat = position.reshape(-1, 3)
    cos_parts = []
    sin_parts = []
    for axis, dim in enumerate(axis_dims):
        table_cos, table_sin = tables[dim]
        cos_parts.append(table_cos[flat[:, axis]])
        sin_parts.append(table_sin[flat[:, axis]])

    shape = (*position.shape[:-1], embed_dim)
    return torch.cat(cos_parts, dim=-1).reshape(shape), torch.cat(sin_parts, dim=-1).reshape(shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]):
    cos, sin = freqs_cis
    cos = cos.to(x.device).unsqueeze(1)
    sin = sin.to(x.device).unsqueeze(1)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _sdpa(query, key, value, heads):
    """Multi-head scaled dot product attention.

    query/key/value arrive as (B, L, inner_dim). The value projection may carry a
    different inner width (used by the reference-attention material split), so its
    head dim is derived independently.
    """
    b = query.shape[0]
    head_dim = query.shape[-1] // heads
    v_head_dim = value.shape[-1] // heads

    query = query.view(b, -1, heads, head_dim).transpose(1, 2)
    key = key.view(b, -1, heads, head_dim).transpose(1, 2)
    value = value.view(b, -1, heads, v_head_dim).transpose(1, 2)

    hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    hidden = hidden.transpose(1, 2).reshape(b, -1, heads * v_head_dim)
    return hidden


# ---------------------------------------------------------------------------
# PBR material heads (stored under attn.processor.* to match the checkpoint)
# ---------------------------------------------------------------------------
class SelfAttnProcessor(nn.Module):
    """Extra per-material projections for material-dimension self attention.

    Holds q/k/v/out linears for every non-albedo PBR token; albedo reuses the parent
    attention's own to_q/to_k/to_v/to_out. Mirrors ``SelfAttnProcessor2_0`` params.
    """

    def __init__(self, query_dim, inner_dim, cross_attention_dim, pbr_setting, bias=False,
                 out_bias=True, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.pbr_setting = pbr_setting
        for token in pbr_setting:
            if token == "albedo":
                continue
            self.add_module(f"to_q_{token}", operations.Linear(query_dim, inner_dim, bias=bias, dtype=dtype, device=device))
            self.add_module(f"to_k_{token}", operations.Linear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype, device=device))
            self.add_module(f"to_v_{token}", operations.Linear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype, device=device))
            self.add_module(f"to_out_{token}", nn.ModuleList([
                operations.Linear(inner_dim, query_dim, bias=out_bias, dtype=dtype, device=device),
                nn.Dropout(dropout),
            ]))


class RefAttnProcessor(nn.Module):
    """Extra per-material value/out projections for reference attention.

    Query/key are shared (computed once from the albedo stream); only the value and
    output projections are material specific. Mirrors ``RefAttnProcessor2_0`` params.
    """

    def __init__(self, query_dim, inner_dim, cross_attention_dim, pbr_setting, bias=False,
                 out_bias=True, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.pbr_setting = pbr_setting
        for token in pbr_setting:
            if token == "albedo":
                continue
            self.add_module(f"to_v_{token}", operations.Linear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype, device=device))
            self.add_module(f"to_out_{token}", nn.ModuleList([
                operations.Linear(inner_dim, query_dim, bias=out_bias, dtype=dtype, device=device),
                nn.Dropout(dropout),
            ]))


class Attention(nn.Module):
    """Minimal torch-native attention matching diffusers' ``Attention`` key layout.

    Only the pieces exercised by the paint UNet are implemented: to_q/to_k/to_v/to_out
    plus an optional ``processor`` submodule carrying the PBR material heads.
    """

    def __init__(self, query_dim, heads, dim_head, cross_attention_dim=None, bias=False,
                 out_bias=True, dropout=0.0, processor=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.query_dim = query_dim
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(self.cross_attention_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(self.cross_attention_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, query_dim, bias=out_bias, dtype=dtype, device=device),
            nn.Dropout(dropout),
        ])
        self.processor = processor

    # -- standard cross/self attention (multiview base, dino, plain self) -----
    def forward(self, hidden_states, encoder_hidden_states=None):
        enc = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = self.to_q(hidden_states)
        key = self.to_k(enc)
        value = self.to_v(enc)
        out = _sdpa(query, key, value, self.heads)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    # -- multiview attention with 3D RoPE ------------------------------------
    def forward_multiview(self, hidden_states, position_indices=None, n_pbrs=1):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        b = query.shape[0]
        query = query.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(b, -1, self.heads, self.dim_head).transpose(1, 2)

        if position_indices is not None:
            if self.dim_head in position_indices:
                image_rotary_emb = position_indices[self.dim_head]
            else:
                image_rotary_emb = get_3d_rotary_pos_embed(
                    rearrange(position_indices["voxel_indices"].unsqueeze(1).repeat(1, n_pbrs, 1, 1),
                              "b n_pbrs l c -> (b n_pbrs) l c"),
                    self.dim_head,
                    voxel_resolution=position_indices["voxel_resolution"],
                )
                position_indices[self.dim_head] = image_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden = hidden.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        hidden = self.to_out[0](hidden)
        hidden = self.to_out[1](hidden)
        return hidden

    # -- material-dimension self attention (attn1) ---------------------------
    def forward_material_self(self, hidden_states, pbr_setting):
        # hidden_states: (b, n_pbr, n, l, c)
        b = hidden_states.size(0)
        pbr_hidden_states = torch.split(hidden_states, 1, dim=1)
        results = []
        for token, pbr_hs in zip(pbr_setting, pbr_hidden_states):
            hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c")
            if token == "albedo":
                to_q, to_k, to_v, to_out = self.to_q, self.to_k, self.to_v, self.to_out
            else:
                to_q = getattr(self.processor, f"to_q_{token}")
                to_k = getattr(self.processor, f"to_k_{token}")
                to_v = getattr(self.processor, f"to_v_{token}")
                to_out = getattr(self.processor, f"to_out_{token}")
            out = _sdpa(to_q(hs), to_k(hs), to_v(hs), self.heads)
            out = to_out[0](out)
            out = to_out[1](out)
            results.append(rearrange(out, "(b n_pbrs n) l c -> b n_pbrs n l c", b=b, n_pbrs=1))
        return torch.cat(results, dim=1)

    # -- reference attention (shared q/k, per-material v/out) -----------------
    def forward_ref(self, hidden_states, encoder_hidden_states, pbr_setting):
        # hidden_states: (b, n*l, c) albedo query ; encoder: (b, n_ref*l, c)
        #
        # Faithful to RefAttnProcessor2_0: ALL materials' value projections are
        # concatenated channel-wise and reshaped into heads of width
        # n_pbr*dim_head, so each attention head pairs its q/k slice with an
        # interleaved mix of albedo/mr value channels (head h sees concat
        # channels [h*n_pbr*dim_head, (h+1)*n_pbr*dim_head)). That scrambled
        # packing is what the released weights were trained with - computing
        # each material's attention separately with matched head slices is
        # mathematically "cleaner" but weight-incompatible (verified against
        # captured reference activations; see the paint parity harness).
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        values = [self.to_v(encoder_hidden_states)]
        for token in pbr_setting:
            if token != "albedo":
                values.append(getattr(self.processor, f"to_v_{token}")(encoder_hidden_states))
        value = torch.cat(values, dim=-1)  # (b, n_ref*l, n_pbr*inner_dim)

        b = query.shape[0]
        n_pbr = len(pbr_setting)
        q = query.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = key.view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = value.view(b, -1, self.heads, n_pbr * self.dim_head).transpose(1, 2)

        hidden = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        outputs = []
        for i, token in enumerate(pbr_setting):
            to_out = self.to_out if token == "albedo" else getattr(self.processor, f"to_out_{token}")
            chunk = hidden[..., i * self.dim_head:(i + 1) * self.dim_head]
            chunk = chunk.transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
            chunk = to_out[0](chunk)
            chunk = to_out[1](chunk)
            outputs.append(chunk)
        return torch.stack(outputs, dim=1)  # (b, n_pbr, n*l, c)


# ---------------------------------------------------------------------------
# PoseRoPE position lookups: quantize each attention resolution's tokens to
# voxel indices by averaging the canonical-coordinate render over grid cells.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _mean_voxel_indices(position_maps: torch.Tensor, grid_resolution: int = 8, voxel_resolution: int = 128):
    """Average valid position samples per grid cell and quantize to voxel indices.

    ``position_maps`` is (B, N, 3, H, W) in [0, 1], where a pixel equal to 1.0 in
    every channel is background. Cells whose valid coverage falls below 1/16 of
    the cell area are zeroed rather than averaged from a handful of edge pixels.
    Returns (B, N, 3, g, g) long indices into a ``voxel_resolution``-sized grid.

    The pooling runs in float16 regardless of the model dtype, mirroring the
    reference ``compute_discrete_voxel_indice`` (which casts to half before
    averaging): the subsequent round() lands boundary cells on the same voxel
    index the released weights were trained with, and keeps the indices
    identical across fp16/bf16/fp32 inference.
    """
    position_maps = position_maps.half()
    b, n, channels, height, width = position_maps.shape
    g = grid_resolution
    assert height % g == 0 and width % g == 0
    cell_h, cell_w = height // g, width // g

    valid = (position_maps != 1).all(dim=2, keepdim=True).to(position_maps.dtype)

    def cell_sums(maps: torch.Tensor) -> torch.Tensor:
        ch = maps.shape[2]
        cells = maps.reshape(b, n, ch, g, cell_h, g, cell_w)
        return cells.sum(dim=(4, 6))

    sums = cell_sums(position_maps * valid)
    counts = cell_sums(valid)

    means = sums / counts.clamp(min=1)
    means = means.masked_fill(counts < (cell_h * cell_w) // 16, 0.0)
    return (means.clamp(0.0, 1.0) * (voxel_resolution - 1)).round().long()


@torch.no_grad()
def multires_voxel_indices(position_maps: torch.Tensor, grid_resolutions, voxel_resolutions):
    """Precompute the PoseRoPE lookup for each attention resolution, keyed by token count."""
    lookups = {}
    for grid_resolution, voxel_resolution in zip(grid_resolutions, voxel_resolutions):
        indices = _mean_voxel_indices(position_maps, grid_resolution, voxel_resolution)
        b, n = indices.shape[:2]
        tokens = indices.permute(0, 1, 3, 4, 2).reshape(b, -1, 3)
        lookups[tokens.shape[1]] = {
            "voxel_indices": tokens,
            "voxel_resolution": voxel_resolution,
        }
    return lookups

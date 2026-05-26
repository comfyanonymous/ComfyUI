import torch
import torch.nn as nn

from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import Mlp


def apply_adaln_(x, shift, scale):
    return x.addcmul_(x, scale).add_(shift)


def precompute_freqs_cis_2d(dim, height, width, theta=10000.0, scale=16.0, device=None, dtype=torch.float32):
    """2D RoPE with x/y axis frequencies interleaved at stride 2 across head dim.

    Returns Flux-format rotation matrices of shape [H*W, dim/2, 2, 2].
    Layout of head-dim pairs: [x_0, y_0, x_1, y_1, ..., x_{dim/4-1}, y_{dim/4-1}].
    """
    x_pos = torch.linspace(0, scale, width, device=device)
    y_pos = torch.linspace(0, scale, height, device=device)
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing="ij")
    x_pos = x_grid.reshape(-1)
    y_pos = y_grid.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device, dtype=torch.float32)[: (dim // 4)] / dim))
    x_freqs = torch.outer(x_pos, freqs)
    y_freqs = torch.outer(y_pos, freqs)
    freqs_interleaved = torch.stack([x_freqs, y_freqs], dim=-1).reshape(height * width, -1)
    cos = torch.cos(freqs_interleaved)
    sin = torch.sin(freqs_interleaved)
    out = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*cos.shape, 2, 2)
    return out.to(dtype=dtype)


def get_2d_sincos_pos_embed(embed_dim, height, width, device=None, dtype=torch.float32):
    """Torch port of MAE's 2D sin/cos absolute positional embedding for the pixel embedder.

    first half encodes W-coordinates, second half H.
    """
    assert embed_dim % 4 == 0
    grid_h = torch.arange(height, dtype=torch.float32, device=device)
    grid_w = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid_y = grid_y.reshape(-1)
    grid_x = grid_x.reshape(-1)
    omega = torch.arange(embed_dim // 4, dtype=torch.float32, device=device) / (embed_dim / 4.0)
    omega = 1.0 / (10000.0 ** omega)
    out_w = torch.outer(grid_x, omega)
    out_h = torch.outer(grid_y, omega)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    return torch.cat([emb_w, emb_h], dim=1).to(dtype=dtype)


class RotaryAttention(nn.Module):
    """Single-stream self-attention with rotary positional encoding (used inside PiTBlock)."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dtype=None, device=None, operations=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.proj = operations.Linear(dim, dim, dtype=dtype, device=device)

    def forward(self, x, pos, mask=None, transformer_options={}):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, pos[None, None])
        x = optimized_attention(q, k, v, H, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.proj(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.linear = operations.Linear(hidden_size, out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    """Linear projection used both for patchified-image tokens and text-feature tokens."""
    def __init__(self, in_chans, embed_dim, norm_layer=None, bias=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(in_chans, embed_dim, bias=bias, dtype=dtype, device=device)
        if norm_layer is not None:
            self.norm = operations.RMSNorm(embed_dim, eps=1e-6, dtype=dtype, device=device)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    """Pixel-level embedder: lifts each RGB pixel to hidden_size and packs into per-patch sequences."""
    def __init__(self, in_channels, hidden_size_output, use_pixel_abs_pos=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size_output = hidden_size_output
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = operations.Linear(self.in_channels, self.hidden_size_output, bias=True, dtype=dtype, device=device)

    def forward(self, inputs, patch_size):
        B, _, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        P2 = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = get_2d_sincos_pos_embed(self.hidden_size_output, H, W, device=x.device, dtype=x.dtype).view(H, W, self.hidden_size_output)
            x = x + pos_full.unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * Hs * Ws, P2, self.hidden_size_output)


class PiTBlock(nn.Module):
    """Pixel-level transformer block.

    Compresses each patch's P^2 pixel tokens → 1 attention token via a linear,
    runs global self-attention across patches with 2D RoPE, then expands back to P^2 tokens.
    Conditioning is per-pixel adaLN from the patch-level features.
    """
    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size, num_heads, mlp_ratio=4.0,
                 attn_hidden_size=None, attn_num_heads=None, rope_fn=None,
                 dtype=None, device=None, operations=None, mlp_chunks=1):
        super().__init__()
        self.pixel_dim = pixel_hidden_size
        self.context_dim = patch_hidden_size
        self.attn_dim = attn_hidden_size if attn_hidden_size is not None else patch_hidden_size
        self.num_heads = attn_num_heads if attn_num_heads is not None else num_heads
        assert self.attn_dim % self.num_heads == 0
        p2 = patch_size * patch_size
        self.compress_to_attn = operations.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True, dtype=dtype, device=device)
        self.expand_from_attn = operations.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True, dtype=dtype, device=device)
        self.norm1 = operations.RMSNorm(self.pixel_dim, eps=1e-6, dtype=dtype, device=device)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False,
                                    dtype=dtype, device=device, operations=operations)
        self.norm2 = operations.RMSNorm(self.pixel_dim, eps=1e-6, dtype=dtype, device=device)
        self.mlp = Mlp(self.pixel_dim, hidden_features=int(self.pixel_dim * mlp_ratio),
                       dtype=dtype, device=device, operations=operations)
        self.adaLN_modulation_msa = operations.Linear(self.context_dim, 3 * self.pixel_dim * p2, bias=True, dtype=dtype, device=device)
        self.adaLN_modulation_mlp = operations.Linear(self.context_dim, 3 * self.pixel_dim * p2, bias=True, dtype=dtype, device=device)
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d
        self.mlp_chunks = max(1, int(mlp_chunks))

    def _fetch_pos(self, height, width, device, dtype):
        return self._rope_fn(self.attn_dim // self.num_heads, height, width, device=device, dtype=dtype)

    def forward(self, x, s_cond, image_height, image_width, patch_size, mask=None, transformer_options={}):
        BL, P2, _ = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        L = Hs * Ws
        B = BL // L

        # Attention path uses only msa params; compute, use, free before mlp params allocate.
        msa_params = self.adaLN_modulation_msa(s_cond).view(BL, P2, 3 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa = msa_params.chunk(3, dim=-1)
        x_norm = apply_adaln_(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(BL, P2 * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(B, L, self.attn_dim)
        pos_comp = self._fetch_pos(Hs, Ws, x.device, x.dtype)
        attn_out = self.attn(x_comp, pos_comp, mask=mask, transformer_options=transformer_options)
        attn_flat = self.expand_from_attn(attn_out.view(B * L, self.attn_dim))
        attn_exp = attn_flat.view(BL, P2, self.pixel_dim)
        x = torch.addcmul(x, gate_msa, attn_exp)
        del msa_params, shift_msa, scale_msa, gate_msa

        mlp_params = self.adaLN_modulation_mlp(s_cond).view(BL, P2, 3 * self.pixel_dim)
        shift_mlp, scale_mlp, gate_mlp = mlp_params.chunk(3, dim=-1)
        gate_mlp = gate_mlp.contiguous()  # detach from mlp_params so the del below frees shift+scale storage before the MLP
        mlp_input = apply_adaln_(self.norm2(x), shift_mlp, scale_mlp)
        del mlp_params, shift_mlp, scale_mlp
        chunk_size = (BL + self.mlp_chunks - 1) // self.mlp_chunks
        for s in range(0, BL, chunk_size):
            e = min(s + chunk_size, BL)
            x[s:e].addcmul_(gate_mlp[s:e], self.mlp(mlp_input[s:e]))
        return x

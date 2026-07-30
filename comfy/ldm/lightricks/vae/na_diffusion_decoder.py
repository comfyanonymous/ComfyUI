"""LTX 2.4 diffusion video VAE decoder (NADiffusionDecoder) in pure PyTorch.

Port of the reference ``DiffusionVideoDecoder`` without the NATTEN dependency:
``natten.na3d`` is replaced by a tiled pure-torch 3D neighborhood attention
that reproduces NATTEN's semantics (window of exactly ``kernel_size`` per
query, shifted inward at grid boundaries, dilation 1).

Stages 1-4 deterministically upsample the latent into a context volume via
NA transformer blocks + linear pixel-shuffle upsamples. Stage 5 runs
``DiffusionNABlock``s that denoise patchified noised pixels ``x_t`` guided by
that context through AdaLN-Zero scale/shift. The 2.4 checkpoint is single-step
``x0``: one forward pass yields the pixels directly, no Euler loop.

State dict keys match the shipped checkpoints directly (fused ``attn.qkv``,
``t_embedder.mlp.{0,2}``, ``shared_adaln.proj``); no rename pass is needed.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from comfy.ldm.lightricks.model import get_timestep_embedding
from .causal_video_autoencoder import Encoder, processor

try:
    import comfy_kitchen
    # Fused NA in comfy_kitchen
    _kitchen_na3d = getattr(comfy_kitchen, "na3d", None)
except ImportError:
    _kitchen_na3d = None

# Target element count for one NA tile's [Nq, Nk] attention mask. Bounds both
# the mask allocation (~64 MB bf16 at 2**25) and, on CPU, the math-backend
# score materialization.
NA_SCORE_BUDGET = 2 ** 25
# Element budget for the stacked K/V copies of one batched SDPA call on CUDA
# (2**28 elements ~= 512 MB bf16 transient).
NA_KV_STACK_BUDGET = 2 ** 28
# Token chunk for the SwiGLU MLP (bounds the [chunk, hidden] workspace).
MLP_TOKEN_CHUNK = 65536


def rms_norm(x, weight, eps=1e-6):
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=eps)
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def patchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b c (f p) (h q) (w r) -> b (c p r q) f h w", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


def unpatchify(x, patch_size_hw, patch_size_t=1):
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=patch_size_t, q=patch_size_hw, r=patch_size_hw)


# --- Absolute per-axis RoPE (matches ltx-core rope.py numerics) ---

def default_rope_dim_split(head_dim):
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim, base=10000.0, device=None):
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    return (1.0 / torch.pow(torch.tensor(float(base), dtype=torch.float64, device=device), exponents)).to(torch.float32)


def _rot_abs_axis(xc, pos, inv, axis):
    """Absolute RoPE on one axis chunk ``xc[..., D]`` (D even) of a 6D
    ``(B, T, H, W, NH, HD)`` tensor. ``pos`` are (global) positions along
    ``axis``. Computed in fp32, returned in ``xc``'s dtype."""
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].float()
    xo = pairs[..., 1].float()
    shape = [1, 1, 1, 1, 1, inv.shape[0]]
    shape[axis] = pos.shape[0]
    ang = (pos[:, None].float() * inv[None, :]).reshape(shape)
    c = ang.cos()
    s = ang.sin()
    re = xe * c - xo * s
    ro = xe * s + xo * c
    return torch.stack([re, ro], dim=-1).reshape(xc.shape).to(out_dtype)


def apply_abs_rope(x, rope_split, inv_freqs, pos_t, pos_h, pos_w):
    """Rotate a ``(B, T, H, W, NH, HD)`` tensor with per-axis absolute RoPE."""
    d_t, d_h, _ = rope_split
    xt = _rot_abs_axis(x[..., :d_t], pos_t, inv_freqs[0], axis=1)
    xh = _rot_abs_axis(x[..., d_t:d_t + d_h], pos_h, inv_freqs[1], axis=2)
    xw = _rot_abs_axis(x[..., d_t + d_h:], pos_w, inv_freqs[2], axis=3)
    return torch.cat([xt, xh, xw], dim=-1)


# --- Pure-torch 3D neighborhood attention (NATTEN na3d semantics) ---

def _window_starts(length, kernel):
    """NATTEN window start per query index: centered, shifted inward at borders."""
    lo = max(length - kernel, 0)
    half = kernel // 2
    return [min(max(i - half, 0), lo) for i in range(length)]


def _rot_axis_tables(xc, cos, sin, axis):
    """Rotate one axis chunk ``xc[..., D]`` (D even) of a 6D ``(B,T,H,W,NH,HD)``
    tensor with precomputed fp32 cos/sin tables ``[len, D/2]``."""
    out_dtype = xc.dtype
    pairs = xc.reshape(*xc.shape[:-1], xc.shape[-1] // 2, 2)
    xe = pairs[..., 0].float()
    xo = pairs[..., 1].float()
    shape = [1, 1, 1, 1, 1, cos.shape[-1]]
    shape[axis] = cos.shape[0]
    c = cos.reshape(shape)
    s = sin.reshape(shape)
    re = xe * c - xo * s
    ro = xe * s + xo * c
    return torch.stack([re, ro], dim=-1).reshape(xc.shape).to(out_dtype)


def _rope_full(x, rope_split, inv_freqs):
    """Absolute RoPE over the full ``(B,T,H,W,NH,HD)`` tensor with global
    0-based positions, chunked over T to bound the fp32 transients."""
    batch, t, h, w, nh, hd = x.shape
    d_t, d_h, d_w = rope_split
    tables = []
    for length, inv in zip((t, h, w), inv_freqs):
        pos = torch.arange(length, dtype=torch.float32, device=x.device)
        ang = pos[:, None] * inv[None, :]
        tables.append((ang.cos(), ang.sin()))
    per_frame = h * w * nh * hd
    chunk = max(1, (2 ** 26) // max(per_frame, 1))
    out = torch.empty_like(x)
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        sl = x[:, t0:t1]
        parts = []
        if d_t:
            parts.append(_rot_axis_tables(sl[..., :d_t], tables[0][0][t0:t1], tables[0][1][t0:t1], axis=1))
        if d_h:
            parts.append(_rot_axis_tables(sl[..., d_t:d_t + d_h], tables[1][0], tables[1][1], axis=2))
        if d_w:
            parts.append(_rot_axis_tables(sl[..., d_t + d_h:], tables[2][0], tables[2][1], axis=3))
        out[:, t0:t1] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    return out


def _pick_tiles(dims, kernels):
    """Choose per-axis query-tile lengths so one tile's ``[Nq, Nk]`` mask stays
    under ``NA_SCORE_BUDGET`` elements."""
    tiles = list(dims)

    def cost(ts):
        nq = math.prod(ts)
        nk = math.prod(min(d, t + k - 1) for t, k, d in zip(ts, kernels, dims))
        return nq * nk

    while cost(tiles) > NA_SCORE_BUDGET and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / kernels[a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    return tiles


def _group_mask(rel_starts, kernels, dtype, device):
    """Additive ``[1, 1, Nq, Nk]`` mask for one tile-geometry group.

    ``rel_starts``: per-axis window starts relative to the key region origin.
    Built as the AND of three tiny per-axis membership masks, so the cost is
    one fill over Nq*Nk, not three broadcasted adds per tile."""
    bools = []
    for starts, kernel in zip(rel_starts, kernels):
        st = torch.tensor(starts, device=device)
        kj = torch.arange(int(st.max()) + kernel, device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < (st[:, None] + kernel)))
    visible = (bools[0][:, None, None, :, None, None]
               & bools[1][None, :, None, None, :, None]
               & bools[2][None, None, :, None, None, :])
    nq = visible.shape[0] * visible.shape[1] * visible.shape[2]
    nk = visible.shape[3] * visible.shape[4] * visible.shape[5]
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def na3d(q, k, v, kernel_size):
    """3D neighborhood attention, NATTEN ``na3d`` semantics, pure torch.

    Fallback used when comfy_kitchen's fused ``na3d`` is unavailable.
    ``q, k, v``: ``(B, T, H, W, NH, HD)``; ``q`` must already be scaled and
    positionally embedded. Tiles sharing the same window geometry are stacked
    into batched ``scaled_dot_product_attention`` calls (online softmax, no
    materialized score tensor), with one additive mask per geometry group.
    Kernels larger than an axis clamp to that axis (the window degenerates to
    full attention there), where NATTEN itself would raise.
    Returns ``(B, T, H, W, NH, HD)``.
    """
    batch, t, h, w, nh, hd = q.shape
    kt, kh, kw = (min(kernel_size[0], t), min(kernel_size[1], h), min(kernel_size[2], w))
    device = q.device

    tile_t, tile_h, tile_w = _pick_tiles((t, h, w), (kt, kh, kw))
    starts = (_window_starts(t, kt), _window_starts(h, kh), _window_starts(w, kw))

    # Group tiles by relative window geometry: interior tiles all share one
    # mask; boundary tiles form a handful of extra groups (<= 3 cases/axis).
    groups = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = starts[0][t0], starts[0][t1 - 1] + kt
        rel_t = tuple(s - rt0 for s in starts[0][t0:t1])
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = starts[1][h0], starts[1][h1 - 1] + kh
            rel_h = tuple(s - rh0 for s in starts[1][h0:h1])
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = starts[2][w0], starts[2][w1 - 1] + kw
                rel_w = tuple(s - rw0 for s in starts[2][w0:w1])
                groups.setdefault((rel_t, rel_h, rel_w), []).append((
                    (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                    (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                ))

    out = torch.empty((batch, t, h, w, nh, hd), device=device, dtype=v.dtype)
    for rel, tiles in groups.items():
        mask = _group_mask(rel, (kt, kh, kw), q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        if device.type == "cuda":
            # SDPA's efficient backend never materializes scores; bound the
            # stacked K/V copies instead.
            g_max = max(1, NA_KV_STACK_BUDGET // max(1, batch * nh * nk * hd * 2))
        else:
            g_max = 1  # CPU math backend materializes [G*B, NH, Nq, Nk]
        qs0, _ = tiles[0]
        tq, th, tw = (qs0[0].stop - qs0[0].start, qs0[1].stop - qs0[1].start, qs0[2].stop - qs0[2].start)
        for c0 in range(0, len(tiles), g_max):
            chunk = tiles[c0:c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            # [G, B, t, h, w, NH, HD] -> [G*B, NH, N, HD]
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nq, hd)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * batch, nh, nk, hd)
            o = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, batch, nh, tq, th, tw, hd).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]

    return out


class NeighborhoodAttention3D(nn.Module):
    """QKV (fused, matching checkpoint keys) + q/k RMSNorm + abs RoPE + NA."""

    def __init__(self, dim, kernel_size, head_dim=64, rope_base=10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim ** -0.5
        self.rope_split = default_rope_dim_split(head_dim)
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        batch, t, h, w, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q = self.q_norm(q.reshape(shape)) * self.scale
        k = self.k_norm(k.reshape(shape))
        v = v.reshape(shape)
        inv_freqs = tuple(rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
        q = _rope_full(q, self.rope_split, inv_freqs)
        k = _rope_full(k, self.rope_split, inv_freqs)
        if _kitchen_na3d is not None:
            out = _kitchen_na3d(q, k, v, list(self.kernel_size), None, 1.0)
        else:
            out = na3d(q, k, v, self.kernel_size)
        return self.proj(out.reshape(batch, t, h, w, self.dim))


class SwiGLU(nn.Module):
    """``w_down(silu(w_gate(x)) * w_up(x))``, chunked over tokens to bound the
    ``[chunk, hidden]`` workspace."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        out = torch.empty_like(x_flat)
        for i in range(0, x_flat.shape[0], MLP_TOKEN_CHUNK):
            chunk = x_flat[i:i + MLP_TOKEN_CHUNK]
            out[i:i + MLP_TOKEN_CHUNK] = self.w_down(F.silu(self.w_gate(chunk)) * self.w_up(chunk))
        return out.reshape(shape)


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds."""

    def __init__(self, dim, kernel_size, head_dim=64, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def modulate(x, scale, shift):
    return x * (1.0 + scale) + shift


class AdaLNZero(nn.Module):
    """``t_emb`` -> 7 (scale/shift/gate) chunks; gate slots unused (folded at export)."""

    NUM_CHUNKS = 7

    def __init__(self, dim, t_emb_dim):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb):
        h = self.proj(F.silu(t_emb))
        return tuple(c[:, None, None, None, :] for c in h.chunk(self.NUM_CHUNKS, dim=-1))


class DiffusionNABlock(nn.Module):
    """NA + SwiGLU with shared AdaLN-Zero scale/shift (ungated residuals)."""

    def __init__(self, dim, kernel_size, context_channels, head_dim=64, mlp_ratio=4.0):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x, latent_context, modulation):
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        x = x + self.context_proj(latent_context)
        x = x + self.attn(modulate(self.norm1(x), scale_msa, shift_msa))
        x = x + self.mlp(modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x


class LinearPixelShuffleUpsample(nn.Module):
    """Linear channel-expand, then channels-last pixel shuffle."""

    def __init__(self, in_channels, stride, out_channels_reduction_factor=1):
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, x, drop_leading_frame=True):
        x = self.proj(x)
        x = rearrange(
            x, "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
            p1=self.stride[0], p2=self.stride[1], p3=self.stride[2],
        )
        if self.stride[0] == 2 and drop_leading_frame:
            # The causal temporal pixel-shuffle duplicates the leading frame.
            x = x[:, 1:]
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal(256) -> MLP. ``mlp.{0,2}`` naming matches the checkpoint."""

    def __init__(self, t_emb_dim=384, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, t_emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim, bias=True),
        )

    def forward(self, timestep, dtype):
        emb = get_timestep_embedding(timestep.flatten(), self.freq_dim, flip_sin_to_cos=True,
                                     downscale_freq_shift=0, scale=1)
        return self.mlp(emb.to(dtype))


class NADiffusionDecoder(nn.Module):
    """Stages 1-4 (deterministic NA upsample) + stage-5 diffusion blocks.

    Input latent must already be un-normalized (the wrapper applies
    ``per_channel_statistics.un_normalize``, same as the conv VAE path).
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        head_dim=64,
        stage_channels=(2048, 1024, 512, 512, 256),
        stage_depths=(4, 6, 4, 2, 8),
        stage_kernels=((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)),
        upsamples=(((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
        stage5_kernel=(11, 11, 11),
        t_emb_dim=384,
        default_num_inference_steps=1,
        timestep_scale_multiplier=1000.0,
        model_output_type="x0",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps),
            persistent=False,
        )
        self.temporal_upscale = math.prod(s[0] for s, _ in upsamples)
        self.spatial_upscale = math.prod(s[1] for s, _ in upsamples) * patch_size
        # NATTEN-style last-frame border mitigation: replicate the last latent
        # frame through stages 1-4, crop the appendix off the context after.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(len(stage_channels) - 1):
            c = stage_channels[stage_i]
            self.det_stages.append(nn.ModuleList(
                [NABlock(c, stage_kernels[stage_i], head_dim=head_dim) for _ in range(stage_depths[stage_i])]
            ))
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(LinearPixelShuffleUpsample(c, stride, out_channels_reduction_factor=reduction))

        self.t_embedder = TimestepEmbedder(t_emb_dim=t_emb_dim)

        c5 = stage_channels[-1]
        self.context_channels = c5
        noised_pixel_channels = out_channels * (patch_size ** 2)
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim)
        self.diff_blocks = nn.ModuleList([
            DiffusionNABlock(c5, stage5_kernel, context_channels=c5, head_dim=head_dim)
            for _ in range(stage_depths[-1])
        ])
        self.norm_out = RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)

    def forward_pre_diffusion(self, z, drop_leading_frame=True, pad_trailing=True):
        """Stages 1-4: latent -> stage-5 context, channels-last.

        ``drop_leading_frame`` must be True only when ``z`` contains the
        latent's true temporal origin (t=0); tiled callers decoding a later
        temporal chunk pass False (the duplicate leading frame belongs solely
        to the origin chunk). ``pad_trailing`` only for chunks containing the
        latent's last frame."""
        n = self.trailing_pad_latent_frames if pad_trailing else 0
        if n > 0:
            z = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
        x = z.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
            x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        if n > 0:
            x = x[:, :-(n * self.temporal_upscale)]
        return x

    def forward_diff_step(self, context, x_t, t):
        x = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(x.permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, context, modulation)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward(self, z, generator=None, drop_leading_frame=True, pad_trailing=True):
        context = self.forward_pre_diffusion(z, drop_leading_frame=drop_leading_frame, pad_trailing=pad_trailing)
        batch, t5, h5, w5, _ = context.shape
        pixel_shape = (batch, self.out_channels, t5, h5 * self.patch_size, w5 * self.patch_size)
        x_t = torch.randn(pixel_shape, dtype=z.dtype, device=z.device, generator=generator)

        timesteps = self.default_inference_timesteps.to(z.device)
        num_steps = timesteps.shape[0]
        for i in range(num_steps):
            t_now = timesteps[i].expand(batch)
            model_out = self.forward_diff_step(context, x_t, t_now)
            if self.model_output_type == "x0":
                x0 = model_out
                if i == num_steps - 1:
                    return x0
                velocity = (x_t.float() - x0.float()) / timesteps[i]
            else:  # "v"
                velocity = model_out.float()
                if i == num_steps - 1:
                    return (x_t.float() - timesteps[i] * velocity).to(z.dtype)
            t_next = timesteps[i + 1] if i + 1 < num_steps else torch.zeros_like(timesteps[i])
            x_t = (x_t.float() - (timesteps[i] - t_next) * velocity).to(z.dtype)
        return x_t


LTX_24_VAE_CONFIG = {
    "_class_name": "CausalDiffusionVAE",
    "dims": 3,
    "model_output_type": "x0",
    "encoder": {
        "dims": 3,
        "in_channels": 3,
        "out_channels": 128,
        "blocks": [
            ["res_x", {"num_layers": 4}],
            ["compress_space_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 6}],
            ["compress_time_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 4}],
            ["compress_all_res", {"multiplier": 2}],
            ["res_x", {"num_layers": 2}],
            ["compress_all_res", {"multiplier": 1}],
            ["res_x", {"num_layers": 2}],
        ],
        "patch_size": 4,
        "latent_log_var": "constant",
        "norm_layer": "pixel_norm",
        "base_channels": 128,
        "spatial_padding_mode": "zeros",
    },
    "decoder": {
        "in_channels": 128,
        "out_channels": 3,
        "patch_size": 4,
        "head_dim": 64,
        "stage_channels": [2048, 1024, 512, 512, 256],
        "stage_depths": [4, 6, 4, 2, 8],
        "stage_kernels": [[3, 7, 7], [3, 7, 7], [3, 5, 5], [3, 5, 5], [11, 11, 11]],
        "upsamples": [[[1, 2, 2], 2], [[2, 1, 1], 2], [[2, 2, 2], 1], [[2, 2, 2], 2]],
        "stage5_kernel": [11, 11, 11],
        "timestep_scale_multiplier": 1000.0,
        "default_num_inference_steps": 1,
    },
}


class CausalDiffusionVAE(nn.Module):
    """LTX 2.4 video VAE: conv encoder (shared with the 2.0 arch) + NA
    diffusion decoder. Interface mirrors ``causal_video_autoencoder.VideoVAE``.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = LTX_24_VAE_CONFIG
        self.config = config
        enc = config.get("encoder", LTX_24_VAE_CONFIG["encoder"])
        dec = config.get("decoder", LTX_24_VAE_CONFIG["decoder"])
        dec_defaults = LTX_24_VAE_CONFIG["decoder"]

        self.encoder = Encoder(
            dims=enc.get("dims", 3),
            in_channels=enc.get("in_channels", 3),
            out_channels=enc.get("out_channels", 128),
            blocks=enc.get("blocks", LTX_24_VAE_CONFIG["encoder"]["blocks"]),
            patch_size=enc.get("patch_size", 4),
            latent_log_var=enc.get("latent_log_var", "constant"),
            norm_layer=enc.get("norm_layer", "pixel_norm"),
            spatial_padding_mode=enc.get("spatial_padding_mode", "zeros"),
            base_channels=enc.get("base_channels", 128),
        )

        self.decoder = NADiffusionDecoder(
            in_channels=dec.get("in_channels", 128),
            out_channels=dec.get("out_channels", 3),
            patch_size=dec.get("patch_size", 4),
            head_dim=dec.get("head_dim", 64),
            stage_channels=tuple(dec.get("stage_channels", dec_defaults["stage_channels"])),
            stage_depths=tuple(dec.get("stage_depths", dec_defaults["stage_depths"])),
            stage_kernels=tuple(tuple(k) for k in dec.get("stage_kernels", dec_defaults["stage_kernels"])),
            upsamples=tuple((tuple(s), r) for s, r in dec.get("upsamples", dec_defaults["upsamples"])),
            stage5_kernel=tuple(dec.get("stage5_kernel", dec_defaults["stage5_kernel"])),
            t_emb_dim=dec.get("t_emb_dim", 384),
            default_num_inference_steps=dec.get("default_num_inference_steps", 1),
            timestep_scale_multiplier=dec.get("timestep_scale_multiplier", 1000.0),
            model_output_type=config.get("model_output_type", "x0"),
        )

        self.per_channel_statistics = processor()

    def encode(self, x, device=None):
        x = x[:, :, :max(1, 1 + ((x.shape[2] - 1) // 8) * 8), :, :]
        means, logvar = torch.chunk(self.encoder(x, device=device), 2, dim=1)
        return self.per_channel_statistics.normalize(means)

    def decode(self, x):
        # Fixed-seed noise so decodes are reproducible TODO: expose?
        generator = torch.Generator(device=x.device)
        generator.manual_seed(0)
        return self.decoder(self.per_channel_statistics.un_normalize(x), generator=generator)

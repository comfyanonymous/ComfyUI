"""MiniMax H3 audio-video DiT.

Single-stream packed-token transformer denoising video (24ch, patch 1x2x2) and
stereo audio (32ch, 40 Hz) latents jointly, conditioned on Qwen3-VL layer-50
hidden states. The packed sequence is [text | cond rows | audio | video] for
t2va/fl2va and [text | reference blocks | audio | video] for ref2va, with full
bidirectional attention (batch 1, no padding, no varlen).

Timestep domain: the model receives the *video* sigma from the sampler and
derives per-token timesteps t = 1 - sigma internally; the audio stream runs on
its own shifted schedule (sigma_shift video 12.0 / audio 3.0), mapped from the
video sigma in closed form. The audio velocity is returned scaled by the
schedule map's derivative d(sigma_a)/d(sigma_v).
"""

import math

import torch
import torch.nn as nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


def time_shift_sigma(sigma, from_shift, to_shift):
    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma, from_shift, to_shift):
    """d(sigma_to)/d(sigma_from) at the same base-grid point.

    Scaling a stream's returned velocity by this slope makes the flat ODE that
    any sampler integrates on the from-schedule equal to that stream's true ODE
    on its own schedule.
    """
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (from_shift * (1.0 + (to_shift - 1.0) * base) ** 2)


def step_timesteps(sigma_v, shift_v, shift_a):
    sigma_v = sigma_v.float().clamp(min=1e-6)
    sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
    return float(1.0 - sigma_v), float(1.0 - sigma_a), sigma_v, sigma_a


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = torch.einsum("nthwcrpq->nctrhpwq", x)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    # origin + exclusive cumsum
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = operations.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, t, frame_rate=None):
        # t: [M] in [0, 1]; fp32 throughout, cos before sin
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frame_rate is not None:
            frame_rate = torch.as_tensor(frame_rate, dtype=torch.float32, device=t.device).flatten()
            if frame_rate.shape[0] == 1:
                frame_rate = frame_rate.expand(t.shape[0])
            fps_args = frame_rate[:, None] * freqs[None]
            emb.add_(torch.cat([torch.cos(fps_args), torch.sin(fps_args)], dim=-1))
        return self.proj_out(nn.functional.silu(self.proj_in(emb)))


def rope_rotation_table(angles, dtype):
    """[S, rot_dim] pair angles -> [S, 1, rot_dim/2, 2, 2] rotation matrices."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            rot = rope_freqs.shape[-3] * 2
            q[..., :rot], k[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
                q[..., :rot], k[..., :rot], rope_freqs)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
        return self.out_proj(out.squeeze(0))


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None, operations=None):
        super().__init__()
        self.fc1 = operations.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = operations.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        gate, up = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate).mul_(up))


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, dtype=None, device=None, operations=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def project(self, t_emb):
        # [M, t_dim] -> [M, modalities, expand, hidden]
        x = self.linear(nn.functional.silu(t_emb))
        return x.view(x.shape[0], self.modalities, self.expand, self.hidden)

    def forward(self, t_emb):
        x = self.project(t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, segments):
    # segments: [(start, stop, mod_row)] covering h contiguously. h is always a
    # freshly produced norm output, so modulate in place with broadcast rows.
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row]).add_(shift[row])
    return h


def _mod_gate(x, gate, other, segments):
    # other is the fresh attn/mlp output: accumulate the gated residual in place
    for a, b, row in segments:
        other[a:b].mul_(gate[row]).add_(x[a:b])
    return other


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x, transformer_options={}):
        # attn/mlp outputs are fresh: accumulate residuals in place
        x = self.attn(self.norm1(x), transformer_options=transformer_options).add_(x)
        return self.mlp(self.norm2(x)).add_(x)


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)])
        self.final_norm = operations.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)

    def forward(self, x, transformer_options={}):
        for block in self.blocks:
            x = block(x, transformer_options=transformer_options)
        return self.final_norm(x)


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 dtype=None, device=None, operations=None, adaln=True):
        super().__init__()
        self.hidden = hidden
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, dtype=dtype, device=device, operations=operations) if adaln else None

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, modulation=None):
        if modulation is None:
            mod_values = self.adaln_proj(t_emb)
        else:
            mod_values = (y.reshape(-1, self.hidden) for y in modulation.unbind(dim=0))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod_values
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, dtype=None, device=None, operations=None, adaln=True):
        super().__init__()
        self.hidden = hidden
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, dtype=dtype, device=device, operations=operations) if adaln else None
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_seg, audio_seg, modulation=None):
        # video_seg / audio_seg: (start, stop, timestep_row) of the target streams
        if modulation is None:
            mod_values = self.adaln_proj(t_emb)
        else:
            mod_values = (y.reshape(-1, self.hidden) for y in modulation.unbind(dim=0))
        shift, scale = mod_values
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


class ModulationBlock(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, dtype=None, device=None, operations=None):
        super().__init__()
        self.adaln_proj = AdalnProj(t_dim, hidden, expand, modalities, dtype=dtype, device=device, operations=operations)


class MiniMaxH3ModulationModel(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, timestep_input_dim=256,
                 time_embed_hidden_size=5376, time_embed_dim=2688,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                          dtype=torch.float32, device=device, operations=operations)
        self.blocks = nn.ModuleList([
            ModulationBlock(time_embed_dim, hidden_size, 6, 3, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)
        ])
        self.final_layer = ModulationBlock(time_embed_dim, hidden_size, 2, 1,
                                           dtype=torch.float32, device=device, operations=operations)

    def forward(self, timesteps, transformer_options={}):
        frame_rate = transformer_options.get("minimax_h3_frame_rate", None)
        t_emb = self.time_embedder(timesteps, frame_rate=frame_rate).to(self.dtype)
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), timesteps.device, transformer_options)
        modulation = None
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, timesteps.device, block)
            projected = block.adaln_proj.project(t_emb)
            if modulation is None:
                modulation = torch.empty((len(self.blocks), projected.shape[2], projected.shape[0], projected.shape[1], projected.shape[3]),
                                         dtype=projected.dtype, device=projected.device)
            modulation[i].copy_(projected.permute(2, 0, 1, 3))
            del projected
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, timesteps.device, None)
        final_modulation = self.final_layer.adaln_proj.project(t_emb).permute(2, 0, 1, 3).contiguous()
        return modulation, final_modulation


class MiniMaxH3ModulationCache:
    def __init__(self, timesteps, blocks, final):
        self.blocks = blocks
        self.final = final
        self.timestep_rows = {float(t): i for i, t in enumerate(timesteps)}

    def __call__(self, timesteps):
        try:
            rows = [self.timestep_rows[float(t)] for t in timesteps]
        except KeyError as e:
            raise RuntimeError(f"MiniMax H3 modulation cache does not contain timestep {e.args[0]}") from None
        rows = torch.tensor(rows, dtype=torch.long, device=self.blocks.device)
        device = timesteps.device

        def block(index):
            return self.blocks[index].index_select(1, rows).to(device)

        return block, self.final.index_select(1, rows).to(device)


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]  # (kind, n_rows)
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]  # per segment: [n, 3] float64 (t, h, w)

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            # fl2va: keyframe cond rows right after text, sharing the target spatial grid
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)  # [S, 3] float64
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        tags = torch.ones(self.seq_len, dtype=torch.long)  # text default
        tags[self.audio_pos] = 2
        tags[self.img_pos] = 0
        self.token_tags = tags
        # contiguous segment table (start, stop, kind); kinds: text / cond / ref_img /
        # ref_audio / audio / video — the packed sequence is uniform per segment in
        # (modality tag, timestep class), except the text span (tag runs resolved at
        # forward time from the presentation tags)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 image_model=None, split_modulation=False, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.split_modulation = split_modulation
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = operations.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = operations.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        self.time_embedder = None if split_modulation else TimeEmbedder(
            timestep_input_dim, time_embed_hidden_size, time_embed_dim,
            dtype=torch.float32, device=device, operations=operations)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, dtype=dtype, device=device, operations=operations,
                     adaln=not split_modulation)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, dtype=dtype, device=device, operations=operations,
                                      adaln=not split_modulation)
        self._layout_cache = {}

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds.

        Called once per sampling from extra_conds with the states already at the
        model's compute dtype (the input never changes across steps); forward
        accepts either form and only runs the refiner if needed.
        """
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def rope_freqs(self, position_ids, device, video_rows=None, temporal_scale=1.0, low_frequency_count=16):
        # [S, 3] float64 -> [S, 96] fp32
        pos = position_ids.to(torch.float32).to(device)
        inv = comfy.model_management.cast_to(self.rope.inv_freq, device=device)
        per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        if video_rows is not None and temporal_scale != 1.0 and low_frequency_count > 0:
            t_f[video_rows, -low_frequency_count:] *= temporal_scale
        half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
        return torch.cat((half, half), dim=-1)                 # [S, 96]

    def _layout(self, text_len, latent_t, latent_h, latent_w, audio_t, payload):
        keyframes = payload.get("keyframes")
        refs = payload.get("refs")
        key = (text_len, latent_t, latent_h, latent_w, audio_t,
               tuple((kf["resolved_frame_index"]) for kf in keyframes) if keyframes else None,
               tuple((b["kind"], b.get("latent_t", 0), b.get("latent_h", 0), b.get("latent_w", 0),
                      b.get("ref_audio_t", 0)) for b in refs) if refs else None)
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t,
                                  keyframes=keyframes, refs=refs,
                                  frame_count=payload.get("frame_count"))
            self._layout_cache = {key: layout}  # keep exactly one
        return layout

    def _cond_video_rows(self, payload, device):
        """Concatenated visual condition rows (normalized latents -> patchified), with condition noise augmentation."""
        rows = []
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        latents = payload.get("cond_video_latents", [])
        for i, z in enumerate(latents):
            r = patchify_video(z.to(torch.float32), self.patch_size)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def _cond_audio_rows(self, payload, device):
        rows = []
        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents", []):
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            rows.append(r.to(device))
        return torch.cat(rows, dim=0) if rows else None

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # model_base casts context to the compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        layout = self._layout(text_len, latent_t, lat_h, lat_w, audio_t, payload)

        # model_base passes model_sampling.timestep(sigma) = sigma * 1000
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        video_t = VISUAL_COND_TIMESTEP if transformer_options.get("minimax_h3_clean_video", False) else t_v
        seg_t = {"text": t_v, "video": video_t, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "ref_audio": max(t_a, aud_aug)}
        unique_t = sorted({t_v, t_a, video_t} | ({seg_t["cond"]} if has_vis_cond else set())
                          | ({seg_t["ref_audio"]} if has_aud_cond else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                # the presentation text span mixes tags (vision pads carry the
                # video modality); split into tag runs
                tags = text_tags.view(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        # ---- embed ----
        img_update = layout.img_update.to(device)
        audio_update = layout.audio_update.to(device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states),
                                             transformer_options=transformer_options)

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        timestep_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
        modulation_provider = transformer_options.get("minimax_h3_modulation", None)
        if modulation_provider is None:
            if self.split_modulation:
                raise RuntimeError("MiniMax H3 split transformer requires its modulation model")
            frame_rate = transformer_options.get("minimax_h3_frame_rate", None)
            t_emb = self.time_embedder(timestep_values, frame_rate=frame_rate).to(dtype)
            block_modulation = final_modulation = None
        else:
            if transformer_options.get("patches_replace", {}).get("dit", {}):
                raise RuntimeError("MiniMax H3 split modulation does not support block replacement patches")
            t_emb = None
            block_modulation, final_modulation = modulation_provider(timestep_values)
        # rotation table computed once per forward, consumed by the kitchen split-half rope
        rope_frame_rate = transformer_options.get("minimax_h3_rope_frame_rate", None)
        rope_end_timestep = transformer_options.get("minimax_h3_rope_end_timestep", 1.0)
        rope_scale = 1.0
        if rope_frame_rate is not None and rope_frame_rate != 24.0 and t_v <= rope_end_timestep:
            rope_scale = 24.0 / rope_frame_rate
            if transformer_options.get("minimax_h3_rope_smooth_taper", False) and rope_end_timestep > 0.0:
                rope_scale = 1.0 + (rope_scale - 1.0) * (1.0 - t_v / rope_end_timestep)
        rope_freqs = rope_rotation_table(self.rope_freqs(
            layout.position_ids, device,
            video_rows=(layout.token_tags == 0).to(device),
            temporal_scale=rope_scale,
            low_frequency_count=transformer_options.get("minimax_h3_rope_low_frequency_count", 16),
        ), dtype)

        # ---- blocks ----
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
        for i, block in enumerate(self.blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"], modulation=None)}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                modulation = None if block_modulation is None else block_modulation(i)
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, modulation=modulation)
        if prefetch_queue is not None:
            # drain: unpin the last block's prefetched weights (the queue only
            # cleans an entry when the following pop consumes it)
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

        # target streams are single contiguous segments (audio then video, last two)
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg, modulation=final_modulation)

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)

        # The sampler integrates the flat ODE dX/dsigma_v = (X - denoised)/sigma_v.
        # Scaling the audio velocity by d(sigma_a)/d(sigma_v) makes that ODE equal
        # to the audio stream's true ODE on its own shifted schedule.
        slope_a = time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
        return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]

"""PiD — Pixel Diffusion Decoder. Decodes a Flux/SD3/Flux2/Z-Image latent
directly to a 4x-upscaled image in 4 distilled flow-matching steps. PixDiT_T2I
body + LQ projection branch injected before each MMDiT patch block.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy.ldm.flux.math import rope

from .model import PixDiT_T2I


def precompute_freqs_cis_2d_ntk(dim: int, height: int, width: int,
                                ref_grid_h: int, ref_grid_w: int,
                                theta: float = 10000.0, scale: float = 16.0,
                                device=None, dtype=torch.float32):
    """NTK-aware 2D RoPE (rope_mode='ntk_aware' in upstream PiD).

    Per-axis theta = theta * (current/ref)^(dim_axis/(dim_axis-2)). Returns
    [H*W, dim/2, 2, 2] with x/y axis freqs interleaved at stride 2 (matches
    the head-dim layout PiD's Q/K weights expect).
    """
    dim_axis = dim // 2
    h_ntk = (height / ref_grid_h) ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0
    w_ntk = (width / ref_grid_w) ** (dim_axis / (dim_axis - 2)) if dim_axis > 2 else 1.0

    x_lin = torch.linspace(0, scale, width, device=device)
    y_lin = torch.linspace(0, scale, height, device=device)
    y_grid, x_grid = torch.meshgrid(y_lin, x_lin, indexing="ij")

    x_rope = rope(x_grid.reshape(1, -1), dim_axis, theta * w_ntk).squeeze(0)
    y_rope = rope(y_grid.reshape(1, -1), dim_axis, theta * h_ntk).squeeze(0)

    out = torch.stack([x_rope, y_rope], dim=2).reshape(height * width, dim // 2, 2, 2)
    return out.to(dtype=dtype)


class SigmaAwareGatePerTokenPerDim(nn.Module):
    """gate = sigmoid(content_proj(cat[x, lq]) - exp(log_alpha) * sigma); out = x + gate * lq.

    Trained init gives ~0.88 gate at sigma=0, ~0.05 at sigma=1.
    """

    def __init__(self, dim: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.content_proj = operations.Linear(dim * 2, dim, dtype=dtype, device=device)
        self.log_alpha = nn.Parameter(torch.empty((), dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, lq: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        content_logit = self.content_proj(torch.cat([x, lq], dim=-1))
        # log_alpha is a raw nn.Parameter -> doesn't auto-cast under dynamic VRAM.
        log_alpha = self.log_alpha.to(device=x.device, dtype=torch.float32)
        sigma_offset = -log_alpha.exp() * sigma.float().view(-1, 1, 1)
        gate = torch.sigmoid(content_logit + sigma_offset)
        return x + (gate * lq).to(x.dtype)


class ResBlock(nn.Module):
    """Pre-activation ResNet block: GN -> SiLU -> Conv -> GN -> SiLU -> Conv + skip."""

    def __init__(self, channels: int, num_groups: int = 4, dtype=None, device=None, operations=None):
        super().__init__()
        self.block = nn.Sequential(
            operations.GroupNorm(num_groups, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(channels, channels, kernel_size=3, padding=1, dtype=dtype, device=device),
            operations.GroupNorm(num_groups, channels, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(channels, channels, kernel_size=3, padding=1, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class LQProjection2D(nn.Module):
    """LQ latent -> per-block patch-aligned features for controlnet-style injection."""

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int = 512,
        out_dim: int = 1536,
        patch_size: int = 16,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        num_res_blocks: int = 4,
        num_outputs: int = 14,
        interval: int = 1,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        assert latent_channels > 0
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.sr_scale = sr_scale
        self.latent_spatial_down_factor = latent_spatial_down_factor
        self.num_outputs = num_outputs
        self.interval = interval

        z_to_patch_ratio = (sr_scale * latent_spatial_down_factor) / patch_size
        self.z_to_patch_ratio = z_to_patch_ratio
        if z_to_patch_ratio >= 1:
            self.latent_upsample_ratio = int(z_to_patch_ratio) if z_to_patch_ratio > 1 else 1
            self.latent_fold_factor = 0
            latent_proj_in_ch = latent_channels
        else:
            fold_factor = int(1 / z_to_patch_ratio)
            assert fold_factor * z_to_patch_ratio == 1.0
            self.latent_upsample_ratio = 0
            self.latent_fold_factor = fold_factor
            latent_proj_in_ch = latent_channels * fold_factor * fold_factor

        layers = [
            operations.Conv2d(latent_proj_in_ch, hidden_dim, kernel_size=3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dtype=dtype, device=device),
        ]
        for _ in range(num_res_blocks):
            layers.append(ResBlock(hidden_dim, dtype=dtype, device=device, operations=operations))
        self.latent_proj = nn.Sequential(*layers)

        self.output_heads = nn.ModuleList(
            [operations.Linear(hidden_dim, out_dim, dtype=dtype, device=device) for _ in range(num_outputs)]
        )
        self.gate_modules = nn.ModuleList(
            [SigmaAwareGatePerTokenPerDim(out_dim, dtype=dtype, device=device, operations=operations)
             for _ in range(num_outputs)]
        )

    def is_gate_active(self, block_idx: int) -> bool:
        return block_idx % self.interval == 0 if self.interval > 1 else True

    def output_index(self, block_idx: int) -> int:
        return block_idx // self.interval if self.interval > 1 else block_idx

    def gate(self, x: torch.Tensor, lq_feature: torch.Tensor, sigma: torch.Tensor, out_idx: int) -> torch.Tensor:
        return self.gate_modules[out_idx](x, lq_feature, sigma)

    def _align_latent_to_patch_grid(self, lq_latent: torch.Tensor, pH: int, pW: int) -> torch.Tensor:
        B, z_dim = lq_latent.shape[:2]
        if self.z_to_patch_ratio >= 1:
            if lq_latent.shape[2] != pH or lq_latent.shape[3] != pW:
                z_aligned = F.interpolate(lq_latent, size=(pH, pW), mode="nearest")
            else:
                z_aligned = lq_latent
        else:
            f = self.latent_fold_factor
            zH_expected, zW_expected = pH * f, pW * f
            if lq_latent.shape[2] != zH_expected or lq_latent.shape[3] != zW_expected:
                lq_latent = F.interpolate(lq_latent, size=(zH_expected, zW_expected), mode="nearest")
            z_aligned = lq_latent.reshape(B, z_dim, pH, f, pW, f).permute(0, 1, 3, 5, 2, 4)
            z_aligned = z_aligned.reshape(B, z_dim * f * f, pH, pW)
        return self.latent_proj(z_aligned)

    def forward(self, lq_latent: torch.Tensor, target_pH: int, target_pW: int) -> List[torch.Tensor]:
        feat = self._align_latent_to_patch_grid(lq_latent, target_pH, target_pW)
        tokens = feat.flatten(2).transpose(1, 2)
        return [head(tokens) for head in self.output_heads]


class PidNet(PixDiT_T2I):
    """PixDiT_T2I + LQ injection (one sigma-gated feature inserted before each patch block)."""

    def __init__(
        self,
        lq_latent_channels: int = 16,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        lq_interval: int = 1,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
        rope_ref_h: int = 1024, # NTK ref resolution in PIXEL units: 1024px / patch=16 -> grid_ref=64.
        rope_ref_w: int = 1024,
        image_model=None,
        dtype=None, device=None, operations=None,
        **pixdit_kwargs,
    ):
        super().__init__(dtype=dtype, device=device, operations=operations, **pixdit_kwargs)

        self.rope_ref_grid_h = int(rope_ref_h) // int(self.patch_size)
        self.rope_ref_grid_w = int(rope_ref_w) // int(self.patch_size)

        # Parent's PiTBlocks were built with plain RoPE — swap in NTK-aware.
        def _pit_rope_fn(head_dim, h, w):
            return precompute_freqs_cis_2d_ntk(head_dim, h, w, self.rope_ref_grid_h, self.rope_ref_grid_w)
        for blk in self.pixel_blocks:
            blk._rope_fn = _pit_rope_fn
            blk._pos_cache = {}

        num_lq_outputs = (self.patch_depth + lq_interval - 1) // lq_interval
        self.lq_proj = LQProjection2D(
            latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim,
            out_dim=self.hidden_size,
            patch_size=self.patch_size,
            sr_scale=sr_scale,
            latent_spatial_down_factor=latent_spatial_down_factor,
            num_res_blocks=lq_num_res_blocks,
            num_outputs=num_lq_outputs,
            interval=lq_interval,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _fetch_patch_pos(self, height, width, device, dtype):
        key = (height, width)
        pos = self._patch_pos_cache.get(key)
        if pos is None:
            pos = precompute_freqs_cis_2d_ntk(
                self.hidden_size // self.num_groups,
                height, width,
                self.rope_ref_grid_h, self.rope_ref_grid_w,
            )
            self._patch_pos_cache[key] = pos
        return pos.to(device=device, dtype=dtype)

    def _forward(self, x, timesteps, context=None, attention_mask=None, transformer_options={},
                 lq_latent=None, degrade_sigma=None, **kwargs):
        B, _, H, W = x.shape
        Hs = H // self.patch_size
        Ws = W // self.patch_size
        L = Hs * Ws

        if context is None or context.dim() != 3:
            raise ValueError("PidNet requires context [B, L, D]")
        if lq_latent is None:
            raise ValueError("PidNet requires lq_latent — attach via PiDConditioning")

        if degrade_sigma is None:
            degrade_sigma = torch.zeros(B, device=x.device, dtype=torch.float32)
        elif not isinstance(degrade_sigma, torch.Tensor):
            degrade_sigma = torch.tensor([float(degrade_sigma)] * B, device=x.device, dtype=torch.float32)
        else:
            degrade_sigma = degrade_sigma.to(device=x.device, dtype=torch.float32).reshape(-1)
            if degrade_sigma.numel() == 1 and B > 1:
                degrade_sigma = degrade_sigma.expand(B).contiguous()

        lq_latent = lq_latent.to(device=x.device, dtype=x.dtype)
        lq_features = self.lq_proj(lq_latent=lq_latent, target_pH=Hs, target_pW=Ws)

        pos_img = self._fetch_patch_pos(Hs, Ws, x.device, x.dtype)
        x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)

        t_emb = self.t_embedder(timesteps.view(-1), x.dtype).view(B, -1, self.hidden_size)

        Ltxt = min(context.shape[1], self.txt_max_length)
        y = context[:, :Ltxt, :]
        y_emb = self.y_embedder(y).view(B, Ltxt, self.hidden_size)
        # y_pos_embedding is raw nn.Parameter -> doesn't auto-cast under dynamic VRAM.
        y_emb = y_emb + self.y_pos_embedding[:, :Ltxt, :].to(device=y_emb.device, dtype=y_emb.dtype)

        condition = F.silu(t_emb)
        pos_txt = self._fetch_text_pos(Ltxt, x.device, x.dtype) if self.use_text_rope else None

        s = self.s_embedder(x_patches)
        for i, blk in enumerate(self.patch_blocks):
            if self.lq_proj.is_gate_active(i):
                out_idx = self.lq_proj.output_index(i)
                if out_idx < len(lq_features):
                    s = self.lq_proj.gate(s, lq_features[out_idx], degrade_sigma, out_idx)
            s, y_emb = blk(s, y_emb, condition, pos_img, pos_txt, None,
                           transformer_options=transformer_options)
        s = F.silu(t_emb + s)

        s_cond = s.view(B * L, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for blk in self.pixel_blocks:
            x_pixels = blk(x_pixels, s_cond, H, W, self.patch_size, mask=None,
                           transformer_options=transformer_options)

        x_pixels = self.final_layer(x_pixels)
        C_out = self.out_channels
        P2 = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(B, L, P2, C_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(B, C_out * P2, L)
        return F.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)

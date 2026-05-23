"""Lens denoising transformer (DiT)"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ldm.flux.layers
import comfy.patcher_extension
from comfy.ldm.modules.attention import optimized_attention


def _lens_time_proj(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    return comfy.ldm.flux.layers.timestep_embedding(t, dim)


def apply_rotary_emb_lens(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class LensEmbedRope(nn.Module):
    """Frame/H/W axial RoPE shared between image and text streams."""

    def __init__(self, theta: int = 10000, axes_dim=(8, 28, 28), scale_rope: bool = True) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        # Plain attributes (not buffers): register_buffer strips complex imag.
        self.pos_freqs = torch.cat(
            [self._rope_params(pos_index, d, theta) for d in self.axes_dim], dim=1
        )
        self.neg_freqs = torch.cat(
            [self._rope_params(neg_index, d, theta) for d in self.axes_dim], dim=1
        )
        self.rope_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(
            index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: Union[List[Tuple[int, int, int]], Tuple[int, int, int]],
        txt_seq_lens: Union[List[int], int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        if not isinstance(txt_seq_lens, list):
            txt_seq_lens = [txt_seq_lens]
        assert len(video_fhw) == 1, "video_fhw must have length 1"

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}_{device}"
            video_freq = self.rope_cache.get(rope_key)
            if video_freq is None:
                video_freq = self._compute_video_freqs(frame, height, width, idx=0).to(device)
                self.rope_cache[rope_key] = video_freq
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)
            vid_freqs.append(video_freq)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0
            ).view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0
            ).view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class _TimestepEmbedder(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.linear_1 = operations.Linear(in_channels, time_embed_dim, dtype=dtype, device=device)
        self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = F.silu(x)
        return self.linear_2(x)


class LensTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.timestep_embedder = _TimestepEmbedder(256, embedding_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = _lens_time_proj(timestep, 256)
        return self.timestep_embedder(proj.to(dtype=hidden_states.dtype))


class GateMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int, dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.w1 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)
        self.w2 = operations.Linear(hidden_dim, dim, bias=False, dtype=dtype, device=device)
        self.w3 = operations.Linear(dim, hidden_dim, bias=False, dtype=dtype, device=device)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x), inplace=True).mul_(self.w3(x)))


class LensJointAttention(nn.Module):
    """Joint image+text attention with fused QKV per stream."""

    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: Optional[int] = None,
        eps: float = 1e-5,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = self.inner_dim // dim_head
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.norm_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        self.img_qkv = operations.Linear(query_dim, 3 * self.inner_dim, bias=True, dtype=dtype, device=device)
        self.txt_qkv = operations.Linear(added_kv_proj_dim, 3 * self.inner_dim, bias=True, dtype=dtype, device=device)

        # ModuleList([Linear, Identity]) for state-dict key compatibility.
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, self.out_dim, bias=True, dtype=dtype, device=device),
            nn.Identity(),
        ])
        self.to_add_out = operations.Linear(self.inner_dim, query_dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        # image stream
        img_qkv = self.img_qkv(hidden_states).view(bsz, seq_img, 3, self.heads, self.dim_head)
        img_q, img_k, img_v = img_qkv.unbind(dim=2)
        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        img_v = img_v.contiguous()
        del img_qkv

        # text stream
        txt_qkv = self.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, self.heads, self.dim_head)
        txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)
        txt_v = txt_v.contiguous()
        del txt_qkv

        img_freqs, txt_freqs = image_rotary_emb
        if img_freqs.shape[0] < seq_img:
            raise ValueError(f"Image RoPE length {img_freqs.shape[0]} < {seq_img}")
        img_freqs = img_freqs[:seq_img]
        img_q = apply_rotary_emb_lens(img_q, img_freqs)
        img_k = apply_rotary_emb_lens(img_k, img_freqs)
        if seq_txt > 0:
            if txt_freqs.shape[0] < seq_txt:
                raise ValueError(f"Text RoPE length {txt_freqs.shape[0]} < {seq_txt}")
            txt_freqs = txt_freqs[:seq_txt]
            txt_q = apply_rotary_emb_lens(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_lens(txt_k, txt_freqs)

        # [B, S, H, D] → [B, H, S, D] for attention, dels to avoid VRAM peaks
        q = torch.cat([img_q, txt_q], dim=1).transpose(1, 2)
        del img_q, txt_q 
        k = torch.cat([img_k, txt_k], dim=1).transpose(1, 2)
        del img_k, txt_k
        v = torch.cat([img_v, txt_v], dim=1).transpose(1, 2)
        del img_v, txt_v

        if attention_mask is not None:
            expected = (bsz, 1, 1, seq_img + seq_txt)
            if attention_mask.shape != expected:
                raise ValueError(
                    f"attention_mask must be {expected}, got {tuple(attention_mask.shape)}"
                )
            attention_mask = attention_mask.to(q.dtype)

        out = optimized_attention(
            q, k, v, self.heads, mask=attention_mask, skip_reshape=True,
            transformer_options=transformer_options,
        )

        img_out = self.to_out[1](self.to_out[0](out[:, :seq_img, :]))
        txt_out = self.to_add_out(out[:, seq_img:, :])
        return img_out, txt_out


class LensTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        rms_norm: bool = True,
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()

        self.attn = LensJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=1e-5,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        if rms_norm:
            NormCls = operations.RMSNorm
            norm_kwargs = {}
        else:
            NormCls = operations.LayerNorm
            norm_kwargs = {"elementwise_affine": False}

        mlp_hidden = int(dim / 3 * 8)

        # Sequential(SiLU, Linear) so state-dict lands at img_mod.1.{weight,bias}.
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = NormCls(dim, eps=eps, dtype=dtype, device=device, **norm_kwargs)
        self.img_norm2 = NormCls(dim, eps=eps, dtype=dtype, device=device, **norm_kwargs)
        self.img_mlp = GateMLP(dim, mlp_hidden, dtype=dtype, device=device, operations=operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = NormCls(dim, eps=eps, dtype=dtype, device=device, **norm_kwargs)
        self.txt_norm2 = NormCls(dim, eps=eps, dtype=dtype, device=device, **norm_kwargs)
        self.txt_mlp = GateMLP(dim, mlp_hidden, dtype=dtype, device=device, operations=operations)

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            transformer_options=transformer_options,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states


class _AdaLayerNormContinuousNoAffine(nn.Module):
    """AdaLayerNormContinuous(elementwise_affine=False).

    The reference uses ``scale, shift = chunk(2)`` (scale first) — opposite
    to Flux's ``LastLayer``.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6,
                 dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.linear = operations.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=True, dtype=dtype, device=device
        )
        self.eps = eps
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        emb = self.linear(F.silu(conditioning))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = F.layer_norm(x, (self.embedding_dim,), None, None, self.eps)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LensTransformer2DModel(nn.Module):
    """Lens dual-stream MMDiT (48 blocks, inner_dim=1536, multi-layer text)."""

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: Optional[int] = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        enc_hidden_dim: int = 2880,
        axes_dims_rope: Tuple[int, int, int] = (8, 28, 28),
        rms_norm: bool = True,
        multi_layer_encoder_feature: bool = True,
        selected_layer_index: Tuple[int, ...] = (5, 11, 17, 23),
        image_model=None,  # unused; accepted for detection-side configs.
        dtype=None,
        device=None,
        operations=None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.multi_layer_encoder_feature = multi_layer_encoder_feature
        self.selected_layer_index = list(selected_layer_index)
        self.dtype = dtype

        self.pos_embed = LensEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = LensTimestepProjEmbeddings(
            embedding_dim=self.inner_dim, dtype=dtype, device=device, operations=operations
        )

        if self.multi_layer_encoder_feature:
            self.txt_norm = nn.ModuleList(
                [operations.RMSNorm(enc_hidden_dim, eps=1e-5, dtype=dtype, device=device)
                 for _ in self.selected_layer_index]
            )
            self.txt_in = operations.Linear(
                enc_hidden_dim * len(self.selected_layer_index),
                self.inner_dim, bias=True, dtype=dtype, device=device,
            )
        else:
            self.txt_norm = operations.RMSNorm(enc_hidden_dim, eps=1e-5, dtype=dtype, device=device)
            self.txt_in = operations.Linear(enc_hidden_dim, self.inner_dim, bias=True, dtype=dtype, device=device)

        self.img_in = operations.Linear(in_channels, self.inner_dim, bias=True, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            LensTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                eps=1e-6,
                rms_norm=rms_norm,
                dtype=dtype, device=device, operations=operations,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = _AdaLayerNormContinuousNoAffine(
            self.inner_dim, self.inner_dim, eps=1e-6,
            dtype=dtype, device=device, operations=operations,
        )
        self.proj_out = operations.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=True,
            dtype=dtype, device=device,
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, context: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                transformer_options: Optional[Dict[str, Any]] = None, **kwargs) -> torch.Tensor:
        if transformer_options is None:
            transformer_options = {}
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward, self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(x, timestep, context, attention_mask, transformer_options, **kwargs)

    def _forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        transformer_options: Optional[Dict[str, Any]] = None,
        control: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """ComfyUI bridge: ``(x[B,128,h,w], t[B], context[B,S,L*H], mask[B,S])``."""
        if transformer_options is None:
            transformer_options = {}
        transformer_options = transformer_options.copy()
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        B, C, h, w = x.shape
        hidden_states = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

        if self.multi_layer_encoder_feature:
            L = len(self.selected_layer_index)
            enc_dim = context.shape[-1] // L
            encoder_hidden_states = list(
                context.reshape(B, -1, L, enc_dim).unbind(dim=2)
            )
            text_seq_len = encoder_hidden_states[0].shape[1]
        else:
            encoder_hidden_states = context
            text_seq_len = context.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                (B, text_seq_len), dtype=torch.bool, device=x.device
            )

        img_len = h * w
        joint_mask = self._build_joint_attention_mask(attention_mask, img_len)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.multi_layer_encoder_feature:
            normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(L)]
            encoder_hidden_states = torch.cat(normed, dim=-1)
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({
                    "img": hidden_states,
                    "txt": encoder_hidden_states,
                    "transformer_options": transformer_options,
                })
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed([(1, h, w)], [text_seq_len], device=hidden_states.device)

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(
                        hidden_states=args["img"],
                        encoder_hidden_states=args["txt"],
                        temb=args["vec"],
                        image_rotary_emb=args["pe"],
                        attention_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                    )
                    return out
                out = blocks_replace[("double_block", i)](
                    {
                        "img": hidden_states,
                        "txt": encoder_hidden_states,
                        "vec": temb,
                        "pe": image_rotary_emb,
                        "attn_mask": joint_mask,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                encoder_hidden_states = out["txt"]
                hidden_states = out["img"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=joint_mask,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({
                        "img": hidden_states,
                        "txt": encoder_hidden_states,
                        "x": x,
                        "block_index": i,
                        "transformer_options": transformer_options,
                    })
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None:
                control_i = control.get("input")
                if control_i is not None and i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add

        hidden_states = self.norm_out(hidden_states, temb)
        out = self.proj_out(hidden_states)
        return out.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _build_joint_attention_mask(text_mask: torch.Tensor, img_len: int) -> torch.Tensor:
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()
        bsz = text_mask.shape[0]
        img_ones = torch.ones((bsz, img_len), dtype=torch.bool, device=text_mask.device)
        joint = torch.cat([img_ones, text_mask], dim=1)
        additive = torch.zeros_like(joint, dtype=torch.float32)
        additive.masked_fill_(~joint, torch.finfo(torch.float32).min)
        return additive[:, None, None, :]

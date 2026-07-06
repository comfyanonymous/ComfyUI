# https://github.com/jdopensource/JoyAI-Image-Edit (Apache 2.0)
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.patcher_extension
from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention


class FP32LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        out = F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps)
        return out.to(orig_dtype)


def _apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    ndim = xq.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq.shape)]
    cos = freqs_cis[0].view(*shape)
    sin = freqs_cis[1].view(*shape)

    def _rotate_half(x):
        x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
        return torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    xq_out = (xq.float() * cos + _rotate_half(xq) * sin).type_as(xq)
    xk_out = (xk.float() * cos + _rotate_half(xk) * sin).type_as(xk)
    return xq_out, xk_out


class JoyImageModulate(nn.Module):
    def __init__(self, hidden_size: int, factor: int, dtype=None, device=None):
        super().__init__()
        self.factor = factor
        self.modulate_table = nn.Parameter(
            torch.empty(1, factor, hidden_size, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> list:
        if x.ndim != 3:
            x = x.unsqueeze(1)
        table = comfy.ops.cast_to_input(self.modulate_table, x)
        return [o.squeeze(1) for o in (table + x).chunk(self.factor, dim=1)]


class JoyImageFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.net = nn.ModuleList([
            _GeluApproximate(dim, inner_dim, dtype=dtype, device=device, operations=operations),
            nn.Identity(),
            operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class _GeluApproximate(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=True, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class JoyImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        inner_dim = num_attention_heads * attention_head_dim

        self.img_attn_qkv = operations.Linear(dim, inner_dim * 3, bias=True, dtype=dtype, device=device)
        self.img_attn_q_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.img_attn_k_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.img_attn_proj = operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device)

        self.txt_attn_qkv = operations.Linear(dim, inner_dim * 3, bias=True, dtype=dtype, device=device)
        self.txt_attn_q_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.txt_attn_k_norm = operations.RMSNorm(attention_head_dim, eps=eps, dtype=dtype, device=device)
        self.txt_attn_proj = operations.Linear(inner_dim, dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        image_rotary_emb: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]],
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = self.num_attention_heads

        img_q, img_k, img_v = self.img_attn_qkv(img).chunk(3, dim=-1)
        txt_q, txt_k, txt_v = self.txt_attn_qkv(txt).chunk(3, dim=-1)

        img_q = img_q.unflatten(-1, (heads, -1))
        img_k = img_k.unflatten(-1, (heads, -1))
        img_v = img_v.unflatten(-1, (heads, -1))
        txt_q = txt_q.unflatten(-1, (heads, -1))
        txt_k = txt_k.unflatten(-1, (heads, -1))
        txt_v = txt_v.unflatten(-1, (heads, -1))

        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)
        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        if image_rotary_emb is not None:
            vis_freqs, txt_freqs = image_rotary_emb
            if vis_freqs is not None:
                img_q, img_k = _apply_rotary_emb(img_q, img_k, vis_freqs)
            if txt_freqs is not None:
                txt_q, txt_k = _apply_rotary_emb(txt_q, txt_k, txt_freqs)

        joint_q = torch.cat([img_q, txt_q], dim=1)
        joint_k = torch.cat([img_k, txt_k], dim=1)
        joint_v = torch.cat([img_v, txt_v], dim=1)

        joint_q = joint_q.flatten(2, 3)
        joint_k = joint_k.flatten(2, 3)
        joint_v = joint_v.flatten(2, 3)

        joint_out = optimized_attention(joint_q, joint_k, joint_v, heads=heads, transformer_options=transformer_options)
        joint_out = joint_out.to(joint_q.dtype)

        seq_img = img.shape[1]
        img_out = joint_out[:, :seq_img, :]
        txt_out = joint_out[:, seq_img:, :]

        img_out = self.img_attn_proj(img_out)
        txt_out = self.txt_attn_proj(txt_out)
        return img_out, txt_out


class JoyImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: float = 4.0,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        mlp_hidden_dim = int(dim * mlp_width_ratio)

        self.img_mod = JoyImageModulate(dim, factor=6, dtype=dtype, device=device)
        self.img_norm1 = FP32LayerNorm(dim, eps=eps)
        self.img_norm2 = FP32LayerNorm(dim, eps=eps)
        self.img_mlp = JoyImageFeedForward(dim, inner_dim=mlp_hidden_dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = JoyImageModulate(dim, factor=6, dtype=dtype, device=device)
        self.txt_norm1 = FP32LayerNorm(dim, eps=eps)
        self.txt_norm2 = FP32LayerNorm(dim, eps=eps)
        self.txt_mlp = JoyImageFeedForward(dim, inner_dim=mlp_hidden_dim, dtype=dtype, device=device, operations=operations)

        self.attn = JoyImageAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        transformer_options={},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(temb)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(temb)

        img_normed = self.img_norm1(hidden_states)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        img_modulated = img_normed * (1 + img_mod1_scale.unsqueeze(1)) + img_mod1_shift.unsqueeze(1)
        txt_modulated = txt_normed * (1 + txt_mod1_scale.unsqueeze(1)) + txt_mod1_shift.unsqueeze(1)

        img_attn, txt_attn = self.attn(img_modulated, txt_modulated, image_rotary_emb, transformer_options=transformer_options)

        hidden_states = hidden_states + img_attn * img_mod1_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + txt_attn * txt_mod1_gate.unsqueeze(1)

        img_ffn_normed = self.img_norm2(hidden_states)
        txt_ffn_normed = self.txt_norm2(encoder_hidden_states)
        img_ffn_input = img_ffn_normed * (1 + img_mod2_scale.unsqueeze(1)) + img_mod2_shift.unsqueeze(1)
        txt_ffn_input = txt_ffn_normed * (1 + txt_mod2_scale.unsqueeze(1)) + txt_mod2_shift.unsqueeze(1)
        hidden_states = hidden_states + self.img_mlp(img_ffn_input) * img_mod2_gate.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.txt_mlp(txt_ffn_input) * txt_mod2_gate.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class JoyImageTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim,
            time_embed_dim=dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.act_fn = nn.SiLU()
        self.time_proj = operations.Linear(dim, time_proj_dim, bias=True, dtype=dtype, device=device)
        self.text_embedder = _PixArtAlphaTextProjection(
            text_embed_dim, dim, dtype=dtype, device=device, operations=operations,
        )

    def forward(self, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = self.timesteps_proj(timestep)
        temb = self.time_embedder(timestep.to(dtype=encoder_hidden_states.dtype)).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return temb, timestep_proj, encoder_hidden_states


class _PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.linear_1 = operations.Linear(in_features, hidden_size, bias=True, dtype=dtype, device=device)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = operations.Linear(hidden_size, hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act_1(self.linear_1(caption)))


class JoyImageTransformer3DModel(nn.Module):
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 3072,
        num_attention_heads: int = 24,
        text_dim: int = 4096,
        mlp_width_ratio: float = 4.0,
        num_layers: int = 20,
        rope_dim_list: list = [16, 56, 56],
        rope_type: str = "rope",
        theta: int = 256,
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.out_channels = out_channels or in_channels
        self.patch_size = list(patch_size)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_dim_list = list(rope_dim_list)
        self.rope_type = rope_type
        self.theta = theta

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        attention_head_dim = hidden_size // num_attention_heads
        if sum(self.rope_dim_list) != attention_head_dim:
            raise ValueError(
                f"sum(rope_dim_list) ({sum(self.rope_dim_list)}) must equal head_dim ({attention_head_dim})"
            )

        self.img_in = operations.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=tuple(self.patch_size),
            stride=tuple(self.patch_size),
            dtype=dtype,
            device=device,
        )

        self.condition_embedder = JoyImageTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=256,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.double_blocks = nn.ModuleList([
            JoyImageTransformerBlock(
                dim=hidden_size,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_width_ratio=mlp_width_ratio,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(hidden_size, eps=1e-6)
        self.proj_out = operations.Linear(
            hidden_size,
            self.out_channels * math.prod(self.patch_size),
            bias=True,
            dtype=dtype,
            device=device,
        )

    def _get_rotary_pos_embed_for_range(
        self,
        start: Tuple[int, int, int],
        stop: Tuple[int, int, int],
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 3D RoPE for the patch grid range [start, stop) over (t, h, w). Token order after
        # reshape(-1) is (t, h, w), matching the img_in Conv3d flatten.
        head_dim = self.hidden_size // self.num_attention_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // 3 for _ in range(3)]
        if sum(rope_dim_list) != head_dim:
            raise ValueError("sum(rope_dim_list) should equal head_dim")

        grids = [torch.arange(start[i], stop[i], dtype=torch.float32, device=device) for i in range(3)]
        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=0)

        cos_parts, sin_parts = [], []
        for i, dim in enumerate(rope_dim_list):
            pos = mesh[i].reshape(-1)
            freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
            angles = torch.outer(pos, freqs)
            cos_parts.append(angles.cos().repeat_interleave(2, dim=1))
            sin_parts.append(angles.sin().repeat_interleave(2, dim=1))

        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    def get_rotary_pos_embed_for_components(
        self,
        component_sizes,
        device=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-component 3D RoPE. component_sizes is a list of (t, h, w) patch grid sizes in
        # sequence order [target, ref0, ref1, ...]; h/w restart at 0 for each component while t
        # continues from the running offset, giving every image its own temporal position band.
        cos_parts, sin_parts = [], []
        t_offset = 0
        for (t, h, w) in component_sizes:
            cos_emb, sin_emb = self._get_rotary_pos_embed_for_range(
                start=(t_offset, 0, 0),
                stop=(t_offset + t, h, w),
                device=device,
            )
            cos_parts.append(cos_emb)
            sin_parts.append(sin_emb)
            t_offset += t
        return torch.cat(cos_parts, dim=0), torch.cat(sin_parts, dim=0)

    def unpatchify(self, x: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        c = self.out_channels
        pt, ph, pw = self.patch_size
        if t * h * w != x.shape[1]:
            raise ValueError(f"Expected t*h*w ({t * h * w}) to equal x.shape[1] ({x.shape[1]})")
        x = x.reshape(x.shape[0], t, h, w, pt, ph, pw, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_latents=None,
        transformer_options={},
        **kwargs,
    ) -> torch.Tensor:
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(hidden_states, timestep, encoder_hidden_states, ref_latents, transformer_options, **kwargs)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        ref_latents=None,
        transformer_options={},
        **kwargs,
    ) -> torch.Tensor:
        # The target noise latent and each reference latent are independently patchified by img_in
        # (Conv3d) and concatenated along the sequence dim, in the order [target, ref0, ref1, ...].
        # RoPE is built per component so references may differ in resolution. Only the leading
        # target segment (tt*th*tw tokens) is projected back out; reference tokens are dropped.
        # A single reference is simply the len(ref_latents) == 1 case.
        if hidden_states.ndim != 5:
            raise ValueError(f"JoyImage transformer expects 5D (B,C,T,H,W) hidden_states; got shape {tuple(hidden_states.shape)}")

        _, _, ot, oh, ow = hidden_states.shape
        pt, ph, pw = self.patch_size
        if ot % pt != 0 or oh % ph != 0 or ow % pw != 0:
            raise ValueError(
                f"JoyImage: target latent spatial/temporal shape {(ot, oh, ow)} must be divisible by patch_size {tuple(self.patch_size)}"
            )
        tt = ot // pt
        th = oh // ph
        tw = ow // pw

        components = [hidden_states]
        if ref_latents is not None:
            for r in ref_latents:
                if r.ndim != 5:
                    raise ValueError(f"JoyImage: each reference latent must be 5D (B,C,T,H,W); got shape {tuple(r.shape)}")
                components.append(r)

        component_sizes = []
        img_tokens = []
        for comp in components:
            _, _, ct, ch, cw = comp.shape
            if ct % pt != 0 or ch % ph != 0 or cw % pw != 0:
                raise ValueError(
                    f"JoyImage: component shape {(ct, ch, cw)} must be divisible by patch_size {tuple(self.patch_size)}"
                )
            component_sizes.append((ct // pt, ch // ph, cw // pw))
            tokens = self.img_in(comp).flatten(2).transpose(1, 2)  # (B, n_i, D)
            img_tokens.append(tokens)

        img = torch.cat(img_tokens, dim=1)

        _, vec, txt = self.condition_embedder(timestep, encoder_hidden_states)
        vec = vec.unflatten(1, (6, -1))

        vis_cos, vis_sin = self.get_rotary_pos_embed_for_components(
            component_sizes,
            device=hidden_states.device,
        )
        vis_freqs = (vis_cos, vis_sin)
        txt_freqs = None
        image_rotary_emb = (vis_freqs, txt_freqs)

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        transformer_options["total_blocks"] = len(self.double_blocks)
        transformer_options["block_type"] = "double"
        for i, block in enumerate(self.double_blocks):
            transformer_options["block_index"] = i
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(
                        hidden_states=args["img"],
                        encoder_hidden_states=args["txt"],
                        temb=args["vec"],
                        image_rotary_emb=args["pe"],
                        transformer_options=args.get("transformer_options"),
                    )
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                           "txt": txt,
                                                           "vec": vec,
                                                           "pe": image_rotary_emb,
                                                           "transformer_options": transformer_options},
                                                          {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    temb=vec,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

        img = self.proj_out(self.norm_out(img))
        target_tokens = tt * th * tw
        img = img[:, :target_tokens, :]
        img = self.unpatchify(img, tt, th, tw)
        return img

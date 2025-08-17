# https://github.com/QwenLM/Qwen-Image (Apache 2.0)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import repeat

from comfy.ldm.lightricks.model import TimestepEmbedding, Timesteps
from comfy.ldm.modules.attention import optimized_attention_masked
from comfy.ldm.flux.layers import EmbedND
import comfy.ldm.common_dit

class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, bias=bias, dtype=dtype, device=device)
        self.approximate = approximate

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        inner_dim=None,
        bias: bool = True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.net = nn.ModuleList([])
        self.net.append(GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype, device=device, operations=operations))
        self.net.append(nn.Dropout(dropout))
        self.net.append(operations.Linear(inner_dim, dim_out, bias=bias, dtype=dtype, device=device))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def apply_rotary_emb(x, freqs_cis):
    if x.shape[1] == 0:
        return x

    t_ = x.reshape(*x.shape[:-1], -1, 1, 2)
    t_out = freqs_cis[..., 0] * t_[..., 0] + freqs_cis[..., 1] * t_[..., 1]
    return t_out.reshape(*x.shape)


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, dtype=None, device=None, operations=None):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        return timesteps_emb


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
        eps: float = 1e-5,
        out_bias: bool = True,
        out_dim: int = None,
        out_context_dim: int = None,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim
        self.heads = heads
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.dropout = dropout

        # Q/K normalization
        self.norm_q = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_k = operations.RMSNorm(dim_head, eps=eps, elementwise_affine=True, dtype=dtype, device=device)
        self.norm_added_q = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)
        self.norm_added_k = operations.RMSNorm(dim_head, eps=eps, dtype=dtype, device=device)

        # Image stream projections
        self.to_q = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.to_k = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.to_v = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Text stream projections
        self.add_q_proj = operations.Linear(query_dim, self.inner_dim, bias=bias, dtype=dtype, device=device)
        self.add_k_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)
        self.add_v_proj = operations.Linear(query_dim, self.inner_kv_dim, bias=bias, dtype=dtype, device=device)

        # Output projections
        self.to_out = nn.ModuleList([
            operations.Linear(self.inner_dim, self.out_dim, bias=out_bias, dtype=dtype, device=device),
            nn.Dropout(dropout)
        ])
        self.to_add_out = operations.Linear(self.inner_dim, self.out_context_dim, bias=out_bias, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_txt = encoder_hidden_states.shape[1]

        img_query = self.to_q(hidden_states).unflatten(-1, (self.heads, -1))
        img_key = self.to_k(hidden_states).unflatten(-1, (self.heads, -1))
        img_value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        txt_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))
        txt_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, -1))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        joint_query = apply_rotary_emb(joint_query, image_rotary_emb)
        joint_key = apply_rotary_emb(joint_key, image_rotary_emb)

        joint_query = joint_query.flatten(start_dim=2)
        joint_key = joint_key.flatten(start_dim=2)
        joint_value = joint_value.flatten(start_dim=2)

        joint_hidden_states = optimized_attention_masked(joint_query, joint_key, joint_value, self.heads, attention_mask)

        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output)
        img_attn_output = self.to_out[1](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype=None,
        device=None,
        operations=None
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.img_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.img_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            operations.Linear(dim, 6 * dim, bias=True, dtype=dtype, device=device),
        )
        self.txt_norm1 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_norm2 = operations.LayerNorm(dim, elementwise_affine=False, eps=eps, dtype=dtype, device=device)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, dtype=dtype, device=device, operations=operations)

        self.attn = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            eps=eps,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states


class LastLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=False,
        eps=1e-6,
        bias=True,
        dtype=None, device=None, operations=None
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = operations.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias, dtype=dtype, device=device)
        self.norm = operations.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=bias, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class QwenImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        image_model=None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.patch_size = patch_size
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pe_embedder = EmbedND(dim=attention_head_dim, theta=10000, axes_dim=list(axes_dims_rope))

        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            dtype=dtype,
            device=device,
            operations=operations
        )

        self.txt_norm = operations.RMSNorm(joint_attention_dim, eps=1e-6, dtype=dtype, device=device)
        self.img_in = operations.Linear(in_channels, self.inner_dim, dtype=dtype, device=device)
        self.txt_in = operations.Linear(joint_attention_dim, self.inner_dim, dtype=dtype, device=device)

        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dtype=dtype,
                device=device,
                operations=operations
            )
            for _ in range(num_layers)
        ])

        self.norm_out = LastLayer(self.inner_dim, self.inner_dim, dtype=dtype, device=device, operations=operations)
        self.proj_out = operations.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True, dtype=dtype, device=device)
        self.gradient_checkpointing = False

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        return hidden_states, repeat(img_ids, "h w c -> b (h w) c", b=bs), orig_shape

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        **kwargs
    ):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size), ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size)))
        txt_ids = torch.linspace(txt_start, txt_start + context.shape[1], steps=context.shape[1], device=x.device, dtype=x.dtype).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).squeeze(1).unsqueeze(2).to(x.dtype)

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

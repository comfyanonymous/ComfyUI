import torch
from torch import nn
import comfy.patcher_extension
import comfy.ldm.modules.attention
import comfy.ldm.common_dit
import math
from typing import Dict, Optional, Tuple

from .symmetric_patchifier import SymmetricPatchifier, latent_to_pixel_coords
from comfy.ldm.flux.math import apply_rope1

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
        dtype=None, device=None, operations=None,
    ):
        super().__init__()

        self.linear_1 = operations.Linear(in_channels, time_embed_dim, sample_proj_bias, dtype=dtype, device=device)

        if cond_proj_dim is not None:
            self.cond_proj = operations.Linear(cond_proj_dim, in_channels, bias=False, dtype=dtype, device=device)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = operations.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias, dtype=dtype, device=device)

        if post_act_fn is None:
            self.post_act = None
        # else:
        #     self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False, dtype=None, device=None, operations=None):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, dtype=dtype, device=device, operations=operations)

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)
        return timesteps_emb


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, dtype=None, device=None, operations=None):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions, dtype=dtype, device=device, operations=operations
        )

        self.silu = nn.SiLU()
        self.linear = operations.Linear(embedding_dim, 6 * embedding_dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep

class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh", dtype=None, device=None, operations=None):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = operations.Linear(in_features=in_features, out_features=hidden_size, bias=True, dtype=dtype, device=device)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = operations.Linear(in_features=hidden_size, out_features=out_features, bias=True, dtype=dtype, device=device)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GELU_approx(nn.Module):
    def __init__(self, dim_in, dim_out, dtype=None, device=None, operations=None):
        super().__init__()
        self.proj = operations.Linear(dim_in, dim_out, dtype=dtype, device=device)

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out, mult=4, glu=False, dropout=0., dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELU_approx(dim, inner_dim, dtype=dtype, device=device, operations=operations)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            operations.Linear(inner_dim, dim_out, dtype=dtype, device=device)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., attn_precision=None, dtype=None, device=None, operations=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.attn_precision = attn_precision

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = operations.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)

        self.to_q = operations.Linear(query_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_k = operations.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_v = operations.Linear(context_dim, inner_dim, bias=True, dtype=dtype, device=device)

        self.to_out = nn.Sequential(operations.Linear(inner_dim, query_dim, dtype=dtype, device=device), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, pe=None, transformer_options={}):
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rope1(q.unsqueeze(1), pe).squeeze(1)
            k = apply_rope1(k.unsqueeze(1), pe).squeeze(1)

        if mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        else:
            out = comfy.ldm.modules.attention.optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, attn_precision=None, dtype=None, device=None, operations=None):
        super().__init__()

        self.attn_precision = attn_precision
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, context_dim=None, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)
        self.ff = FeedForward(dim, dim_out=dim, glu=True, dtype=dtype, device=device, operations=operations)

        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, attn_precision=self.attn_precision, dtype=dtype, device=device, operations=operations)

        self.scale_shift_table = nn.Parameter(torch.empty(6, dim, device=device, dtype=dtype))

    def forward(self, x, context=None, attention_mask=None, timestep=None, pe=None, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)).unbind(dim=2)

        attn1_input = comfy.ldm.common_dit.rms_norm(x)
        attn1_input = torch.addcmul(attn1_input, attn1_input, scale_msa).add_(shift_msa)
        attn1_input = self.attn1(attn1_input, pe=pe, transformer_options=transformer_options)
        x.addcmul_(attn1_input, gate_msa)
        del attn1_input

        x += self.attn2(x, context=context, mask=attention_mask, transformer_options=transformer_options)

        y = comfy.ldm.common_dit.rms_norm(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        x.addcmul_(self.ff(y), gate_mlp)

        return x

def get_fractional_positions(indices_grid, max_pos):
    fractional_positions = torch.stack(
        [
            indices_grid[:, i] / max_pos[i]
            for i in range(3)
        ],
        dim=-1,
    )
    return fractional_positions


def precompute_freqs_cis(indices_grid, dim, out_dtype, theta=10000.0, max_pos=[20, 2048, 2048]):
    dtype = torch.float32
    device = indices_grid.device

    # Get fractional positions and compute frequency indices
    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = theta ** torch.linspace(0, 1, dim // 6, device=device, dtype=dtype) * math.pi / 2

    # Compute frequencies and apply cos/sin
    freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    cos_vals = freqs.cos().repeat_interleave(2, dim=-1)
    sin_vals = freqs.sin().repeat_interleave(2, dim=-1)

    # Pad if dim is not divisible by 6
    if dim % 6 != 0:
        padding_size = dim % 6
        cos_vals = torch.cat([torch.ones_like(cos_vals[:, :, :padding_size]), cos_vals], dim=-1)
        sin_vals = torch.cat([torch.zeros_like(sin_vals[:, :, :padding_size]), sin_vals], dim=-1)

    # Reshape and extract one value per pair (since repeat_interleave duplicates each value)
    cos_vals = cos_vals.reshape(*cos_vals.shape[:2], -1, 2)[..., 0].to(out_dtype)  # [B, N, dim//2]
    sin_vals = sin_vals.reshape(*sin_vals.shape[:2], -1, 2)[..., 0].to(out_dtype)  # [B, N, dim//2]

    # Build rotation matrix [[cos, -sin], [sin, cos]] and add heads dimension
    freqs_cis = torch.stack([
        torch.stack([cos_vals, -sin_vals], dim=-1),
        torch.stack([sin_vals, cos_vals], dim=-1)
    ], dim=-2).unsqueeze(1)  # [B, 1, N, dim//2, 2, 2]

    return freqs_cis


class LTXVModel(torch.nn.Module):
    def __init__(self,
                 in_channels=128,
                 cross_attention_dim=2048,
                 attention_head_dim=64,
                 num_attention_heads=32,

                 caption_channels=4096,
                 num_layers=28,


                 positional_embedding_theta=10000.0,
                 positional_embedding_max_pos=[20, 2048, 2048],
                 causal_temporal_positioning=False,
                 vae_scale_factors=(8, 32, 32),
                 dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.generator = None
        self.vae_scale_factors = vae_scale_factors
        self.dtype = dtype
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning

        self.patchify_proj = operations.Linear(in_channels, self.inner_dim, bias=True, dtype=dtype, device=device)

        self.adaln_single = AdaLayerNormSingle(
            self.inner_dim, use_additional_conditions=False, dtype=dtype, device=device, operations=operations
        )

        # self.adaln_single.linear = operations.Linear(self.inner_dim, 4 * self.inner_dim, bias=True, dtype=dtype, device=device)

        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=self.inner_dim, dtype=dtype, device=device, operations=operations
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    context_dim=cross_attention_dim,
                    # attn_precision=attn_precision,
                    dtype=dtype, device=device, operations=operations
                )
                for d in range(num_layers)
            ]
        )

        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim, dtype=dtype, device=device))
        self.norm_out = operations.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.proj_out = operations.Linear(self.inner_dim, self.out_channels, dtype=dtype, device=device)

        self.patchifier = SymmetricPatchifier(1)

    def forward(self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, **kwargs):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, attention_mask, frame_rate, transformer_options, keyframe_idxs, **kwargs)

    def _forward(self, x, timestep, context, attention_mask, frame_rate=25, transformer_options={}, keyframe_idxs=None, **kwargs):
        patches_replace = transformer_options.get("patches_replace", {})

        orig_shape = list(x.shape)

        x, latent_coords = self.patchifier.patchify(x)
        pixel_coords = latent_to_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.vae_scale_factors,
            causal_fix=self.causal_temporal_positioning,
        )

        if keyframe_idxs is not None:
            pixel_coords[:, :, -keyframe_idxs.shape[2]:] = keyframe_idxs

        fractional_coords = pixel_coords.to(torch.float32)
        fractional_coords[:, 0] = fractional_coords[:, 0] * (1.0 / frame_rate)

        x = self.patchify_proj(x)
        timestep = timestep * 1000.0

        if attention_mask is not None and not torch.is_floating_point(attention_mask):
            attention_mask = (attention_mask - 1).to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(x.dtype).max

        pe = precompute_freqs_cis(fractional_coords, dim=self.inner_dim, out_dtype=x.dtype)

        batch_size = x.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = x.shape[0]
            context = self.caption_projection(context)
            context = context.view(
                batch_size, -1, x.shape[-1]
            )

        blocks_replace = patches_replace.get("dit", {})
        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], context=args["txt"], attention_mask=args["attention_mask"], timestep=args["vec"], pe=args["pe"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "attention_mask": attention_mask, "vec": timestep, "pe": pe, "transformer_options": transformer_options}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(
                    x,
                    context=context,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    pe=pe,
                    transformer_options=transformer_options,
                )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = self.norm_out(x)
        # Modulation
        x = torch.addcmul(x, x, scale).add_(shift)
        x = self.proj_out(x)

        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        return x

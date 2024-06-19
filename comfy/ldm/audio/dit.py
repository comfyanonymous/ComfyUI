# code adapted from: https://github.com/Stability-AI/stable-audio-tools

from comfy.ldm.modules.attention import optimized_attention
import typing as tp

import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F
import math

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1., dtype=None, device=None):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.empty(
            [out_features // 2, in_features], dtype=dtype, device=device))

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T.to(dtype=input.dtype, device=input.device)
        return torch.cat([f.cos(), f.sin()], dim=-1)

# norms
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False, dtype=None, device=None):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        self.gamma = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

        if bias:
            self.beta = nn.Parameter(torch.empty(dim, dtype=dtype, device=device))
        else:
            self.beta = None

    def forward(self, x):
        beta = self.beta
        if self.beta is not None:
            beta = beta.to(dtype=x.dtype, device=x.device)
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma.to(dtype=x.dtype, device=x.device), bias=beta)

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation,
        use_conv = False,
        conv_kernel_size = 3,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.act = activation
        self.proj = operations.Linear(dim_in, dim_out * 2, dtype=dtype, device=device) if not use_conv else operations.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2), dtype=dtype, device=device)
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = torch.einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len, device, dtype):
        # device = self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=dtype)
        return self.forward(t)

    def forward(self, t):
        # device = self.inv_freq.device
        device = t.device
        dtype = t.dtype

        # t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq.to(dtype=dtype, device=device))
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale.to(dtype=dtype, device=device) ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = t.dtype #reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation, dtype=dtype, device=device, operations=operations)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                operations.Linear(dim, inner_dim, bias = not no_bias, dtype=dtype, device=device) if not use_conv else operations.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias, dtype=dtype, device=device),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = operations.Linear(inner_dim, dim_out, bias = not no_bias, dtype=dtype, device=device) if not use_conv else operations.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias, dtype=dtype, device=device)

        # # init last linear layer to 0
        # if zero_init_output:
        #     nn.init.zeros_(linear_out.weight)
        #     if not no_bias:
        #         nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm = False,
        natten_kernel_size = None,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)
            self.to_kv = operations.Linear(dim_kv, dim_kv * 2, bias=False, dtype=dtype, device=device)
        else:
            self.to_qkv = operations.Linear(dim, dim * 3, bias=False, dtype=dtype, device=device)

        self.to_out = operations.Linear(dim, dim, bias=False, dtype=dtype, device=device)

        # if zero_init_output:
        #     nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm


    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Normalize q and k for cosine sim attention
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if h != kv_h:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = h // kv_h
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        out = optimized_attention(q, k, v, h, skip_reshape=True)
        out = self.to_out(out)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):

        super().__init__()

        self.dim = dim

        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {},
            dtype=None,
            device=None,
            operations=None,
    ):

        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal

        self.pre_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()

        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            dtype=dtype,
            device=device,
            operations=operations,
            **attn_kwargs
        )

        if cross_attend:
            self.cross_attend_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                dtype=dtype,
                device=device,
                operations=operations,
                **attn_kwargs
            )

        self.ff_norm = LayerNorm(dim, dtype=dtype, device=device, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, dtype=dtype, device=device, operations=operations,**ff_kwargs)

        self.layer_ix = layer_ix

        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * 6, bias=False)
            )

            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            #nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None
    ):
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:

            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim = -1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual

            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                x = x + self.conformer(x)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual

        else:
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)

            if context is not None:
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None:
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))

        return x

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        dtype=None,
        device=None,
        operations=None,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = operations.Linear(dim_in, dim, bias=False, dtype=dtype, device=device) if dim_in is not None else nn.Identity()
        self.project_out = operations.Linear(dim, dim_out, bias=False, dtype=dtype, device=device) if dim_out is not None else nn.Identity()

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                    **kwargs
                )
            )

    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1], dtype=x.dtype, device=x.device)
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        for layer in self.layers:
            x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            # x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

        x = self.project_out(x)

        if return_info:
            return x, info

        return x

class AudioDiffusionTransformer(nn.Module):
    def __init__(self,
        io_channels=64,
        patch_size=1,
        embed_dim=1536,
        cond_token_dim=768,
        project_cond_tokens=False,
        global_cond_dim=1536,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=24,
        num_heads=24,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        audio_model="",
        dtype=None,
        device=None,
        operations=None,
        **kwargs):

        super().__init__()

        self.dtype = dtype
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim, dtype=dtype, device=device)

        self.to_timestep_embed = nn.Sequential(
            operations.Linear(timestep_features_dim, embed_dim, bias=True, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Linear(embed_dim, embed_dim, bias=True, dtype=dtype, device=device),
        )

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                operations.Linear(cond_token_dim, cond_embed_dim, bias=False, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(cond_embed_dim, cond_embed_dim, bias=False, dtype=dtype, device=device)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                operations.Linear(global_cond_dim, global_embed_dim, bias=False, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(global_embed_dim, global_embed_dim, bias=False, dtype=dtype, device=device)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                operations.Linear(prepend_cond_dim, embed_dim, bias=False, dtype=dtype, device=device),
                nn.SiLU(),
                operations.Linear(embed_dim, embed_dim, bias=False, dtype=dtype, device=device)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                dtype=dtype,
                device=device,
                operations=operations,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = operations.Conv1d(dim_in, dim_in, 1, bias=False, dtype=dtype, device=device)
        self.postprocess_conv = operations.Conv1d(io_channels, io_channels, 1, bias=False, dtype=dtype, device=device)

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        **kwargs):

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:

            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]).to(x.dtype)) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists
        if global_embed is not None:
            global_embed = global_embed + timestep_embed
        else:
            global_embed = timestep_embed

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend":
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        if self.transformer_type == "x-transformers":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, **extra_args, **kwargs)
        elif self.transformer_type == "continuous_transformer":
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_cond_mask, mask=mask, prepend_mask=prepend_mask, return_info=return_info, **extra_args, **kwargs)

            if return_info:
                output, info = output
        elif self.transformer_type == "mm_transformer":
            output = self.transformer(x, context=cross_attn_cond, mask=mask, context_mask=cross_attn_cond_mask, **extra_args, **kwargs)

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self,
        x,
        timestep,
        context=None,
        context_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        mask=None,
        return_info=False,
        control=None,
        transformer_options={},
        **kwargs):
            return self._forward(
                x,
                timestep,
                cross_attn_cond=context,
                cross_attn_cond_mask=context_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )

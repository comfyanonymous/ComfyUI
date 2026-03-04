import math

import torch
import torch.nn as nn

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.wan.model import sinusoidal_embedding_1d, repeat_e
import comfy.ldm.common_dit
import comfy.patcher_extension


def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")


class OutputNorm(nn.Module):

    def __init__(self, dim, eps=1e-6, operation_settings={}):
        super().__init__()
        self.scale_shift_table = nn.Parameter(torch.randn(
            1,
            2,
            dim,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        ) / dim**0.5)
        self.norm = operation_settings.get("operations").LayerNorm(
            dim,
            eps,
            elementwise_affine=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        original_context_length: int,
    ):
        temb = temb[:, -original_context_length:, :]
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2).to(hidden_states.device)
        scale = scale.squeeze(2).to(hidden_states.device)
        hidden_states = hidden_states[:, -original_context_length:, :]
        hidden_states = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states


class HeliosSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        qk_norm=True,
        eps=1e-6,
        is_cross_attention=False,
        is_amplify_history=False,
        history_scale_mode="per_head",
        operation_settings={},
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.is_cross_attention = is_cross_attention
        self.is_amplify_history = is_amplify_history

        self.to_q = operation_settings.get("operations").Linear(
            dim,
            dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.to_k = operation_settings.get("operations").Linear(
            dim,
            dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.to_v = operation_settings.get("operations").Linear(
            dim,
            dim,
            bias=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.to_out = nn.ModuleList([
            operation_settings.get("operations").Linear(
                dim,
                dim,
                bias=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
            nn.Dropout(0.0),
        ])

        if qk_norm:
            self.norm_q = operation_settings.get("operations").RMSNorm(
                dim,
                eps=eps,
                elementwise_affine=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            )
            self.norm_k = operation_settings.get("operations").RMSNorm(
                dim,
                eps=eps,
                elementwise_affine=True,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            )
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

        if is_amplify_history:
            if history_scale_mode == "scalar":
                self.history_key_scale = nn.Parameter(torch.ones(
                    1,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ))
            else:
                self.history_key_scale = nn.Parameter(torch.ones(
                    num_heads,
                    device=operation_settings.get("device"),
                    dtype=operation_settings.get("dtype"),
                ))
            self.history_scale_mode = history_scale_mode
            self.max_scale = 10.0

    def forward(
        self,
        x,
        context=None,
        freqs=None,
        original_context_length=None,
        transformer_options={},
    ):
        if context is None:
            context = x

        b, sq, _ = x.shape
        sk = context.shape[1]

        q = self.norm_q(self.to_q(x)).view(b, sq, self.num_heads, self.head_dim)
        k = self.norm_k(self.to_k(context)).view(b, sk, self.num_heads, self.head_dim)
        v = self.to_v(context).view(b, sk, self.num_heads, self.head_dim)

        if freqs is not None:
            q = apply_rope1(q, freqs)
            k = apply_rope1(k, freqs)

        if q.dtype != v.dtype:
            q = q.to(v.dtype)
        if k.dtype != v.dtype:
            k = k.to(v.dtype)

        if (not self.is_cross_attention and self.is_amplify_history and original_context_length is not None):
            history_seq_len = sq - original_context_length
            if history_seq_len > 0:
                scale_key = 1.0 + torch.sigmoid(self.history_key_scale) * (self.max_scale - 1.0)
                if self.history_scale_mode == "per_head":
                    scale_key = scale_key.view(1, 1, -1, 1)
                k = torch.cat([k[:, :history_seq_len] * scale_key, k[:, history_seq_len:]], dim=1)

        y = optimized_attention(
            q.view(b, sq, -1),
            k.view(b, sk, -1),
            v.view(b, sk, -1),
            heads=self.num_heads,
            transformer_options=transformer_options,
        )
        y = self.to_out[0](y)
        y = self.to_out[1](y)
        return y


class HeliosAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        guidance_cross_attn=False,
        is_amplify_history=False,
        history_scale_mode="per_head",
        operation_settings={},
    ):
        super().__init__()

        self.norm1 = operation_settings.get("operations").LayerNorm(
            dim,
            eps,
            elementwise_affine=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.attn1 = HeliosSelfAttention(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            is_cross_attention=False,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
            operation_settings=operation_settings,
        )

        self.norm2 = (operation_settings.get("operations").LayerNorm(
            dim,
            eps,
            elementwise_affine=True,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        ) if cross_attn_norm else nn.Identity())
        self.attn2 = HeliosSelfAttention(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            is_cross_attention=True,
            operation_settings=operation_settings,
        )

        self.norm3 = operation_settings.get("operations").LayerNorm(
            dim,
            eps,
            elementwise_affine=False,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(
                dim,
                ffn_dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
            nn.GELU(approximate="tanh"),
            operation_settings.get("operations").Linear(
                ffn_dim,
                dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

        self.scale_shift_table = nn.Parameter(torch.randn(
            1,
            6,
            dim,
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        ) / dim**0.5)
        self.guidance_cross_attn = guidance_cross_attn

    def forward(self, x, context, e, freqs, original_context_length=None, transformer_options={}):
        if e.ndim == 4:
            e = (self.scale_shift_table.unsqueeze(0) + e.float()).chunk(6, dim=2)
            e = [v.squeeze(2) for v in e]
        else:
            e = (self.scale_shift_table + e.float()).chunk(6, dim=1)

        # self-attn
        y = self.attn1(
            torch.addcmul(repeat_e(e[0], x), self.norm1(x), 1 + repeat_e(e[1], x)),
            freqs=freqs,
            original_context_length=original_context_length,
            transformer_options=transformer_options,
        )
        x = torch.addcmul(x, y, repeat_e(e[2], x))

        # cross-attn
        if self.guidance_cross_attn and original_context_length is not None:
            history_seq_len = x.shape[1] - original_context_length
            history_x, x_main = torch.split(x, [history_seq_len, original_context_length], dim=1)
            x_main = x_main + self.attn2(
                self.norm2(x_main),
                context=context,
                transformer_options=transformer_options,
            )
            x = torch.cat([history_x, x_main], dim=1)
        else:
            x = x + self.attn2(self.norm2(x), context=context, transformer_options=transformer_options)

        # ffn
        y = self.ffn(torch.addcmul(repeat_e(e[3], x), self.norm3(x), 1 + repeat_e(e[4], x)))
        x = torch.addcmul(x, y, repeat_e(e[5], x))
        return x


class HeliosModel(torch.nn.Module):

    def __init__(
            self,
            model_type="t2v",
            patch_size=(1, 2, 2),
            num_attention_heads=40,
            attention_head_dim=128,
            in_channels=16,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=13824,
            num_layers=40,
            cross_attn_norm=True,
            qk_norm=True,
            eps=1e-6,
            rope_dim=(44, 42, 42),
            rope_theta=10000.0,
            guidance_cross_attn=True,
            zero_history_timestep=True,
            has_multi_term_memory_patch=True,
            is_amplify_history=False,
            history_scale_mode="per_head",
            image_model=None,
            device=None,
            dtype=None,
            operations=None,
            **kwargs,
    ):
        del model_type, image_model, kwargs
        super().__init__()
        self.dtype = dtype
        operation_settings = {
            "operations": operations,
            "device": device,
            "dtype": dtype,
        }

        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_dim = out_channels or in_channels
        self.dim = dim
        self.freq_dim = freq_dim
        self.zero_history_timestep = zero_history_timestep

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            device=operation_settings.get("device"),
            dtype=torch.float32,
        )
        self.text_embedding = nn.Sequential(
            operations.Linear(
                text_dim,
                dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
            nn.GELU(approximate="tanh"),
            operations.Linear(
                dim,
                dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )
        self.time_embedding = nn.Sequential(
            operations.Linear(
                freq_dim,
                dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
            nn.SiLU(),
            operations.Linear(
                dim,
                dim,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            operations.Linear(
                dim,
                dim * 6,
                device=operation_settings.get("device"),
                dtype=operation_settings.get("dtype"),
            ),
        )

        d = dim // num_attention_heads
        self.rope_embedder = EmbedND(dim=d, theta=rope_theta, axes_dim=list(rope_dim))

        # pyramidal embedding
        if has_multi_term_memory_patch:
            self.patch_short = operations.Conv3d(
                in_channels,
                dim,
                kernel_size=patch_size,
                stride=patch_size,
                device=operation_settings.get("device"),
                dtype=torch.float32,
            )
            self.patch_mid = operations.Conv3d(
                in_channels,
                dim,
                kernel_size=tuple(2 * p for p in patch_size),
                stride=tuple(2 * p for p in patch_size),
                device=operation_settings.get("device"),
                dtype=torch.float32,
            )
            self.patch_long = operations.Conv3d(
                in_channels,
                dim,
                kernel_size=tuple(4 * p for p in patch_size),
                stride=tuple(4 * p for p in patch_size),
                device=operation_settings.get("device"),
                dtype=torch.float32,
            )

        # blocks
        self.blocks = nn.ModuleList([HeliosAttentionBlock(
            dim,
            ffn_dim,
            num_attention_heads,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            guidance_cross_attn=guidance_cross_attn,
            is_amplify_history=is_amplify_history,
            history_scale_mode=history_scale_mode,
            operation_settings=operation_settings,
        ) for _ in range(num_layers)])

        # head
        self.norm_out = OutputNorm(dim, eps=eps, operation_settings=operation_settings)
        self.proj_out = operations.Linear(
            dim,
            self.out_dim * math.prod(patch_size),
            device=operation_settings.get("device"),
            dtype=operation_settings.get("dtype"),
        )

    def rope_encode(
        self,
        t,
        h,
        w,
        t_start=0,
        steps_t=None,
        steps_h=None,
        steps_w=None,
        device=None,
        dtype=None,
        transformer_options={},
        frame_indices=None,
    ):
        patch_size = self.patch_size
        t_len = (t + (patch_size[0] // 2)) // patch_size[0]
        h_len = (h + (patch_size[1] // 2)) // patch_size[1]
        w_len = (w + (patch_size[2] // 2)) // patch_size[2]

        if steps_t is None:
            steps_t = t_len
        if steps_h is None:
            steps_h = h_len
        if steps_w is None:
            steps_w = w_len

        h_start = 0
        w_start = 0
        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            t_len = (t_len - 1.0) * rope_options.get("scale_t", 1.0) + 1.0
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            t_start += rope_options.get("shift_t", 0.0)
            h_start += rope_options.get("shift_y", 0.0)
            w_start += rope_options.get("shift_x", 0.0)

        if frame_indices is None:
            t_coords = torch.linspace(
                t_start,
                t_start + (t_len - 1),
                steps=steps_t,
                device=device,
                dtype=dtype,
            ).reshape(1, -1, 1, 1)
            batch_size = 1
        else:
            batch_size = frame_indices.shape[0]
            t_coords = frame_indices.to(device=device, dtype=dtype)
            if t_coords.shape[1] != steps_t:
                t_coords = torch.nn.functional.interpolate(
                    t_coords.unsqueeze(1),
                    size=steps_t,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            t_coords = (t_coords + t_start)[:, :, None, None]

        img_ids = torch.zeros((batch_size, steps_t, steps_h, steps_w, 3), device=device, dtype=dtype)
        img_ids[:, :, :, :, 0] = img_ids[:, :, :, :, 0] + t_coords.expand(batch_size, steps_t, steps_h, steps_w)
        img_ids[:, :, :, :, 1] = img_ids[:, :, :, :, 1] + torch.linspace(h_start, h_start + (h_len - 1), steps=steps_h, device=device, dtype=dtype).reshape(1, 1, -1, 1)
        img_ids[:, :, :, :, 2] = img_ids[:, :, :, :, 2] + torch.linspace(w_start, w_start + (w_len - 1), steps=steps_w, device=device, dtype=dtype).reshape(1, 1, 1, -1)
        img_ids = img_ids.reshape(batch_size, -1, img_ids.shape[-1])
        return self.rope_embedder(img_ids).movedim(1, 2)

    def forward(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options={},
        **kwargs,
    ):
        return comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options),
        ).execute(
            x,
            timestep,
            context,
            clip_fea,
            time_dim_concat,
            transformer_options,
            **kwargs,
        )

    def _forward(
        self,
        x,
        timestep,
        context,
        clip_fea=None,
        time_dim_concat=None,
        transformer_options={},
        **kwargs,
    ):
        del clip_fea, time_dim_concat

        _, _, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

        out = self.forward_orig(
            hidden_states=x,
            timestep=timestep,
            context=context,
            indices_hidden_states=kwargs.get("indices_hidden_states", None),
            indices_latents_history_short=kwargs.get("indices_latents_history_short", None),
            indices_latents_history_mid=kwargs.get("indices_latents_history_mid", None),
            indices_latents_history_long=kwargs.get("indices_latents_history_long", None),
            latents_history_short=kwargs.get("latents_history_short", None),
            latents_history_mid=kwargs.get("latents_history_mid", None),
            latents_history_long=kwargs.get("latents_history_long", None),
            transformer_options=transformer_options,
        )

        return out[:, :, :t, :h, :w]

    def forward_orig(
        self,
        hidden_states,
        timestep,
        context,
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        transformer_options={},
    ):
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size

        # embeddings
        hidden_states = self.patch_embedding(hidden_states.float()).to(hidden_states.dtype)
        _, _, post_t, post_h, post_w = hidden_states.shape
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if indices_hidden_states is None:
            indices_hidden_states = (torch.arange(0, post_t, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1))

        freqs = self.rope_encode(
            t=post_t * self.patch_size[0],
            h=post_h * self.patch_size[1],
            w=post_w * self.patch_size[2],
            steps_t=post_t,
            steps_h=post_h,
            steps_w=post_w,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            transformer_options=transformer_options,
            frame_indices=indices_hidden_states,
        )
        original_context_length = hidden_states.shape[1]

        if (latents_history_short is not None and indices_latents_history_short is not None and hasattr(self, "patch_short")):
            x_short = self.patch_short(latents_history_short.float()).to(hidden_states.dtype)
            _, _, ts, hs, ws = x_short.shape
            x_short = x_short.flatten(2).transpose(1, 2)
            f_short = self.rope_encode(
                t=ts * self.patch_size[0],
                h=hs * self.patch_size[1],
                w=ws * self.patch_size[2],
                steps_t=ts,
                steps_h=hs,
                steps_w=ws,
                device=x_short.device,
                dtype=x_short.dtype,
                transformer_options=transformer_options,
                frame_indices=indices_latents_history_short,
            )
            hidden_states = torch.cat([x_short, hidden_states], dim=1)
            freqs = torch.cat([f_short, freqs], dim=1)

        if (latents_history_mid is not None and indices_latents_history_mid is not None and hasattr(self, "patch_mid")):
            x_mid = self.patch_mid(pad_for_3d_conv(latents_history_mid, (2, 4, 4)).float()).to(hidden_states.dtype)
            _, _, tm, hm, wm = x_mid.shape
            x_mid = x_mid.flatten(2).transpose(1, 2)
            f_mid = self.rope_encode(
                t=tm * self.patch_size[0],
                h=hm * self.patch_size[1],
                w=wm * self.patch_size[2],
                steps_t=tm,
                steps_h=hm,
                steps_w=wm,
                device=x_mid.device,
                dtype=x_mid.dtype,
                transformer_options=transformer_options,
                frame_indices=indices_latents_history_mid,
            )
            hidden_states = torch.cat([x_mid, hidden_states], dim=1)
            freqs = torch.cat([f_mid, freqs], dim=1)

        if (latents_history_long is not None and indices_latents_history_long is not None and hasattr(self, "patch_long")):
            x_long = self.patch_long(pad_for_3d_conv(latents_history_long, (4, 8, 8)).float()).to(hidden_states.dtype)
            _, _, tl, hl, wl = x_long.shape
            x_long = x_long.flatten(2).transpose(1, 2)
            f_long = self.rope_encode(
                t=tl * self.patch_size[0],
                h=hl * self.patch_size[1],
                w=wl * self.patch_size[2],
                steps_t=tl,
                steps_h=hl,
                steps_w=wl,
                device=x_long.device,
                dtype=x_long.dtype,
                transformer_options=transformer_options,
                frame_indices=indices_latents_history_long,
            )
            hidden_states = torch.cat([x_long, hidden_states], dim=1)
            freqs = torch.cat([f_long, freqs], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)
        timestep = timestep.to(hidden_states.device)
        if timestep.shape[0] != batch_size:
            timestep = timestep[:1].expand(batch_size)

        # time embeddings
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).to(dtype=hidden_states.dtype))
        e = e.reshape(batch_size, -1, e.shape[-1])
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        context = self.text_embedding(context)

        if self.zero_history_timestep and history_context_length > 0:
            timestep_t0 = torch.zeros((1, ), dtype=timestep.dtype, device=timestep.device)
            e_t0 = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_t0.flatten()).to(dtype=hidden_states.dtype))
            e_t0 = e_t0.reshape(1, -1, e_t0.shape[-1]).expand(batch_size, history_context_length, -1)
            e0_t0 = self.time_projection(e_t0[:, :1]).unflatten(2, (6, self.dim))
            e0_t0 = (e0_t0.view(batch_size, 1, 6, self.dim).permute(0, 2, 1, 3).expand(batch_size, 6, history_context_length, self.dim))

            e = e.expand(batch_size, original_context_length, -1)
            e0 = (e0.view(batch_size, 1, 6, self.dim).permute(0, 2, 1, 3).expand(batch_size, 6, original_context_length, self.dim))
            e = torch.cat([e_t0, e], dim=1)
            e0 = torch.cat([e0_t0, e0], dim=2)
        else:
            e = e.expand(batch_size, hidden_states.shape[1], -1)
            e0 = (e0.view(batch_size, 1, 6, self.dim).permute(0, 2, 1, 3).expand(batch_size, 6, hidden_states.shape[1], self.dim))

        e0 = e0.permute(0, 2, 1, 3)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                context,
                e0,
                freqs,
                original_context_length=original_context_length,
                transformer_options=transformer_options,
            )

        hidden_states = self.norm_out(hidden_states, e, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        return self.unpatchify(hidden_states, (post_t, post_h, post_w))

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        b = x.shape[0]
        u = x[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum("bfhwpqrc->bcfphqwr", u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Keep compatibility with reference diffusers key names.
        remapped = {}
        for k, v in state_dict.items():
            nk = k
            nk = nk.replace("condition_embedder.time_embedder.linear_1.", "time_embedding.0.")
            nk = nk.replace("condition_embedder.time_embedder.linear_2.", "time_embedding.2.")
            nk = nk.replace("condition_embedder.time_proj.", "time_projection.1.")
            nk = nk.replace("condition_embedder.text_embedder.linear_1.", "text_embedding.0.")
            nk = nk.replace("condition_embedder.text_embedder.linear_2.", "text_embedding.2.")
            nk = nk.replace("blocks.", "blocks.")
            remapped[nk] = v

        return super().load_state_dict(remapped, strict=strict, assign=assign)


# Backward-compatible alias for existing integration points.
HeliosTransformer3DModel = HeliosModel

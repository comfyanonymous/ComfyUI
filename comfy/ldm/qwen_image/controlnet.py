import torch
import math

from .model import QwenImageTransformer2DModel
from .model import QwenImageTransformerBlock


class QwenImageFunControlBlock(QwenImageTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, has_before_proj=False, dtype=None, device=None, operations=None):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.has_before_proj = has_before_proj
        if has_before_proj:
            self.before_proj = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.after_proj = operations.Linear(dim, dim, device=device, dtype=dtype)


class QwenImageFunControlNetModel(torch.nn.Module):
    def __init__(
        self,
        control_in_features=132,
        inner_dim=3072,
        num_attention_heads=24,
        attention_head_dim=128,
        num_control_blocks=5,
        main_model_double=60,
        injection_layers=(0, 12, 24, 36, 48),
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()
        self.dtype = dtype
        self.main_model_double = main_model_double
        self.injection_layers = tuple(injection_layers)
        # Keep base hint scaling at 1.0 so user-facing strength behaves similarly
        # to the reference Gen2/VideoX implementation around strength=1.
        self.hint_scale = 1.0
        self.control_img_in = operations.Linear(control_in_features, inner_dim, device=device, dtype=dtype)

        self.control_blocks = torch.nn.ModuleList([])
        for i in range(num_control_blocks):
            self.control_blocks.append(
                QwenImageFunControlBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    has_before_proj=(i == 0),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )

    def _process_hint_tokens(self, hint):
        if hint is None:
            return None
        if hint.ndim == 4:
            hint = hint.unsqueeze(2)

        # Fun checkpoints are trained with 33 latent channels before 2x2 packing:
        # [control_latent(16), mask(1), inpaint_latent(16)] -> 132 features.
        # Default behavior (no inpaint input in stock Apply ControlNet) should use
        # zeros for mask/inpaint branches, matching VideoX fallback semantics.
        expected_c = self.control_img_in.weight.shape[1] // 4
        if hint.shape[1] == 16 and expected_c == 33:
            zeros_mask = torch.zeros_like(hint[:, :1])
            zeros_inpaint = torch.zeros_like(hint)
            hint = torch.cat([hint, zeros_mask, zeros_inpaint], dim=1)

        bs, c, t, h, w = hint.shape
        hidden_states = torch.nn.functional.pad(hint, (0, w % 2, 0, h % 2))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(
            orig_shape[0],
            orig_shape[1],
            orig_shape[-3],
            orig_shape[-2] // 2,
            2,
            orig_shape[-1] // 2,
            2,
        )
        hidden_states = hidden_states.permute(0, 2, 3, 5, 1, 4, 6)
        hidden_states = hidden_states.reshape(
            bs,
            t * ((h + 1) // 2) * ((w + 1) // 2),
            c * 4,
        )

        expected_in = self.control_img_in.weight.shape[1]
        cur_in = hidden_states.shape[-1]
        if cur_in < expected_in:
            pad = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1], expected_in - cur_in),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            hidden_states = torch.cat([hidden_states, pad], dim=-1)
        elif cur_in > expected_in:
            hidden_states = hidden_states[:, :, :expected_in]

        return hidden_states

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        hint=None,
        transformer_options={},
        base_model=None,
        **kwargs,
    ):
        if base_model is None:
            raise RuntimeError("Qwen Fun ControlNet requires a QwenImage base model at runtime.")

        encoder_hidden_states_mask = attention_mask
        # Keep attention mask disabled inside Fun control blocks to mirror
        # VideoX behavior (they rely on seq lengths for RoPE, not masked attention).
        encoder_hidden_states_mask = None

        hidden_states, img_ids, _ = base_model.process_img(x)
        hint_tokens = self._process_hint_tokens(hint)
        if hint_tokens is None:
            raise RuntimeError("Qwen Fun ControlNet requires a control hint image.")

        if hint_tokens.shape[1] != hidden_states.shape[1]:
            max_tokens = min(hint_tokens.shape[1], hidden_states.shape[1])
            hint_tokens = hint_tokens[:, :max_tokens]
            hidden_states = hidden_states[:, :max_tokens]
            img_ids = img_ids[:, :max_tokens]

        txt_start = round(
            max(
                ((x.shape[-1] + (base_model.patch_size // 2)) // base_model.patch_size) // 2,
                ((x.shape[-2] + (base_model.patch_size // 2)) // base_model.patch_size) // 2,
            )
        )
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = base_model.pe_embedder(ids).to(x.dtype).contiguous()

        hidden_states = base_model.img_in(hidden_states)
        encoder_hidden_states = base_model.txt_norm(context)
        encoder_hidden_states = base_model.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            base_model.time_text_embed(timesteps, hidden_states)
            if guidance is None
            else base_model.time_text_embed(timesteps, guidance, hidden_states)
        )

        c = self.control_img_in(hint_tokens)

        for i, block in enumerate(self.control_blocks):
            if i == 0:
                c_in = block.before_proj(c) + hidden_states
                all_c = []
            else:
                all_c = list(torch.unbind(c, dim=0))
                c_in = all_c.pop(-1)

            encoder_hidden_states, c_out = block(
                hidden_states=c_in,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                transformer_options=transformer_options,
            )

            c_skip = block.after_proj(c_out) * self.hint_scale
            all_c += [c_skip, c_out]
            c = torch.stack(all_c, dim=0)

        hints = torch.unbind(c, dim=0)[:-1]

        controlnet_block_samples = [None] * self.main_model_double
        for local_idx, base_idx in enumerate(self.injection_layers):
            if local_idx < len(hints) and base_idx < len(controlnet_block_samples):
                controlnet_block_samples[base_idx] = hints[local_idx]

        return {"input": controlnet_block_samples}


class QwenImageControlNetModel(QwenImageTransformer2DModel):
    def __init__(
        self,
        extra_condition_channels=0,
        dtype=None,
        device=None,
        operations=None,
        **kwargs
    ):
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)
        self.main_model_double = 60

        # controlnet_blocks
        self.controlnet_blocks = torch.nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            self.controlnet_blocks.append(operations.Linear(self.inner_dim, self.inner_dim, device=device, dtype=dtype))
        self.controlnet_x_embedder = operations.Linear(self.in_channels + extra_condition_channels, self.inner_dim, device=device, dtype=dtype)

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        hint=None,
        transformer_options={},
        **kwargs
    ):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        hint, _, _ = self.process_img(hint)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states) + self.controlnet_x_embedder(hint)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        repeat = math.ceil(self.main_model_double / len(self.controlnet_blocks))

        controlnet_block_samples = ()
        for i, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

            controlnet_block_samples = controlnet_block_samples + (self.controlnet_blocks[i](hidden_states),) * repeat

        return {"input": controlnet_block_samples[:self.main_model_double]}

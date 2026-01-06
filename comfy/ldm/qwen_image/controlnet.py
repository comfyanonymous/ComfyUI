import torch
import math

from .model import QwenImageTransformer2DModel


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

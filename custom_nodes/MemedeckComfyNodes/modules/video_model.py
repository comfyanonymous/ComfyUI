import torch
from torch import nn
import comfy.ldm.modules.attention
import comfy.ldm.common_dit
import math

from comfy.ldm.lightricks.model import apply_rotary_emb, precompute_freqs_cis, LTXVModel, BasicTransformerBlock


class GenesisModifiedCrossAttention(nn.Module):
    def forward(self, x, context=None, mask=None, pe=None, transformer_options={}):
        context = x if context is None else context
        context_v = x if context is None else context

        step = transformer_options.get('step', -1)
        total_steps = transformer_options.get('total_steps', 0)
        attn_bank = transformer_options.get('attn_bank', None)
        sample_mode = transformer_options.get('sample_mode', None)

        if attn_bank is not None and self.idx in attn_bank['block_map']:
            len_conds = len(transformer_options['cond_or_uncond'])
            pred_order = transformer_options['pred_order']
            block_map_entry = attn_bank['block_map'][self.idx]  # Pre-compute lookup

            if sample_mode == 'forward' and total_steps - step - 1 < attn_bank['save_steps']:
                step_idx = f'{pred_order}_{total_steps - step - 1}'
                block_map_entry[step_idx] = x.cpu()  
            elif sample_mode == 'reverse' and step < attn_bank['inject_steps']:
                step_idx = f'{pred_order}_{step}'
                inject_settings = attn_bank.get('inject_settings', {})
                if inject_settings:
                    inj = block_map_entry[step_idx].to(x.device).repeat(len_conds, 1, 1)
                    # Use a dictionary or function to map settings to actions
                    if 'q' in inject_settings:
                        x = inj
                    if 'k' in inject_settings:
                        context = inj
                    if 'v' in inject_settings:
                        context_v = inj
    # def forward(self, x, context=None, mask=None, pe=None, transformer_options={}):
    #     context = x if context is None else context
    #     context_v = x if context is None else context

    #     step = transformer_options.get('step', -1)
    #     total_steps = transformer_options.get('total_steps', 0)
    #     attn_bank = transformer_options.get('attn_bank', None)
    #     sample_mode = transformer_options.get('sample_mode', None)
    #     if attn_bank is not None and self.idx in attn_bank['block_map']:
    #         len_conds = len(transformer_options['cond_or_uncond'])
    #         pred_order = transformer_options['pred_order']
    #         if sample_mode == 'forward' and total_steps-step-1 < attn_bank['save_steps']:
    #             step_idx = f'{pred_order}_{total_steps-step-1}'
    #             attn_bank['block_map'][self.idx][step_idx] = x.cpu()
    #         elif sample_mode == 'reverse' and step < attn_bank['inject_steps']:
    #             step_idx = f'{pred_order}_{step}'
    #             inject_settings = attn_bank.get('inject_settings', {})
    #             if len(inject_settings) > 0:
    #                 inj = attn_bank['block_map'][self.idx][step_idx].to(x.device).repeat(len_conds, 1, 1)
    #             if 'q' in inject_settings:
    #                 x = inj
    #             if 'k' in inject_settings:
    #                 context = inj
    #             if 'v' in inject_settings:
    #                 context_v = inj

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context_v)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe)
        
        alt_attn_fn = transformer_options.get('patches_replace', {}).get(f'layer', {}).get(('self_attn', self.idx), None) 
        if alt_attn_fn is not None:
            out = alt_attn_fn(q,k,v, self.heads, attn_precision=self.attn_precision, transformer_options=transformer_options)
        elif mask is None:
            out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, attn_precision=self.attn_precision)
        else:
            out = comfy.ldm.modules.attention.optimized_attention_masked(q, k, v, self.heads, mask, attn_precision=self.attn_precision)
        return self.to_out(out)


class GenesisModifiedBasicTransformerBlock(BasicTransformerBlock):
    def forward(self, x, context=None, attention_mask=None, timestep=None, pe=None, transformer_options={}):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + timestep.reshape(x.shape[0], timestep.shape[1], self.scale_shift_table.shape[0], -1)).unbind(dim=2)
        x += self.attn1(comfy.ldm.common_dit.rms_norm(x) * (1 + scale_msa) + shift_msa, pe=pe, transformer_options=transformer_options) * gate_msa

        x += self.attn2(x, context=context, mask=attention_mask)

        y = comfy.ldm.common_dit.rms_norm(x) * (1 + scale_mlp) + shift_mlp
        x += self.ff(y) * gate_mlp

        return x


class GenesisModelModified(LTXVModel):
   
    def forward(self, x, timestep, context, attention_mask, frame_rate=25, guiding_latent=None, guiding_latents={}, transformer_options={}, **kwargs):
        patches_replace = transformer_options.get("patches_replace", {})

        guiding_latents = transformer_options.get('patches', {}).get('guiding_latents', None)

        indices_grid = self.patchifier.get_grid(
            orig_num_frames=x.shape[2],
            orig_height=x.shape[3],
            orig_width=x.shape[4],
            batch_size=x.shape[0],
            scale_grid=((1 / frame_rate) * 8, 32, 32),
            device=x.device,
        )

        ts = None
        input_x = None

        if guiding_latents is not None:
            input_x = x.clone()
            ts = torch.ones([x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]], device=x.device, dtype=x.dtype)
            input_ts = timestep.view([timestep.shape[0]] + [1] * (x.ndim - 1))
            ts *= input_ts
            for guide in guiding_latents:
                ts[:, :, guide.index] = 0.0
                x[:,:,guide.index] = guide.latent[:,:,0]
            timestep = self.patchifier.patchify(ts)

        orig_shape = list(x.shape)

        x = self.patchifier.patchify(x)

        x = self.patchify_proj(x)
        timestep = timestep * 1000.0

        attention_mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1]))
        attention_mask = attention_mask.masked_fill(attention_mask.to(torch.bool), float("-inf"))  # not sure about this
        # attention_mask = (context != 0).any(dim=2).to(dtype=x.dtype)

        pe = precompute_freqs_cis(indices_grid, dim=self.inner_dim, out_dtype=x.dtype)

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
                    out["img"] = block(args["img"], context=args["txt"], attention_mask=args["attention_mask"], timestep=args["vec"], pe=args["pe"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "attention_mask": attention_mask, "vec": timestep, "pe": pe}, {"original_block": block_wrap})
                x = out["img"]
            else:
                x = block(
                    x,
                    context=context,
                    attention_mask=attention_mask,
                    timestep=timestep,
                    pe=pe,
                    transformer_options=transformer_options
                )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        x = self.norm_out(x)
        # Modulation
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        x = self.patchifier.unpatchify(
            latents=x,
            output_height=orig_shape[3],
            output_width=orig_shape[4],
            output_num_frames=orig_shape[2],
            out_channels=orig_shape[1] // math.prod(self.patchifier.patch_size),
        )

        if guiding_latents is not None:
            for guide in guiding_latents:
                x[:, :, guide.index] = (input_x[:, :, guide.index] - guide.latent[:, :, 0]) / input_ts[:, :, 0]

        return x


def inject_model(diffusion_model):
    diffusion_model.__class__ = GenesisModelModified
    for idx, transformer_block in enumerate(diffusion_model.transformer_blocks):
        transformer_block.__class__ = GenesisModifiedBasicTransformerBlock
        transformer_block.idx = idx
        transformer_block.attn1.__class__ = GenesisModifiedCrossAttention
        transformer_block.attn1.idx = idx
    return diffusion_model
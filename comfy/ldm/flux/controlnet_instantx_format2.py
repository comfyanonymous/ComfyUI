# Original code can be found on: https://github.com/XLabs-AI/x-flux/blob/main/src/flux/controlnet.py

import numbers

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.utils.import_utils import is_torch_version
from einops import rearrange, repeat
from torch import Tensor, nn

from .layers import timestep_embedding
from .model import Flux
from ..common_dit import pad_to_patch_size

if is_torch_version(">=", "2.1.0"):
    LayerNorm = nn.LayerNorm
else:
    # Has optional bias parameter compared to torch layer norm
    # TODO: replace with torch layernorm once min required torch version >= 2.1
    class LayerNorm(nn.Module):
        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = torch.Size(dim)

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
            else:
                self.weight = None
                self.bias = None

        def forward(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# YiYi to-do: refactor rope related functions/classes
def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class InstantXControlNetFluxFormat2(Flux):
    def __init__(self, image_model=None, dtype=None, device=None, operations=None, joint_attention_dim=4096, **kwargs):
        kwargs["depth"] = 0
        kwargs["depth_single_blocks"] = 0
        depth_single_blocks_controlnet = kwargs.pop("depth_single_blocks_controlnet", 2)
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=24,
                    attention_head_dim=128,
                ).to(dtype=dtype)
                for i in range(5)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.hidden_size,
                    num_attention_heads=24,
                    attention_head_dim=128,
                ).to(dtype=dtype)
                for i in range(10)
            ]
        )

        self.require_vae = True
        # add ControlNet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(len(self.single_transformer_blocks)):
            controlnet_block = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_single_blocks.append(controlnet_block)

        # TODO support both union and unimodal
        self.union = True  # num_mode is not None
        num_mode = 10
        if self.union:
            self.controlnet_mode_embedder = nn.Embedding(num_mode, self.hidden_size)
        self.controlnet_x_embedder = zero_module(operations.Linear(self.in_channels, self.hidden_size).to(device=device, dtype=dtype))
        self.gradient_checkpointing = False

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    def forward_orig(
            self,
            img: Tensor,
            img_ids: Tensor,
            controlnet_cond: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            timesteps: Tensor,
            y: Tensor,
            guidance: Tensor = None,
            controlnet_mode=None
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        batch_size = img.shape[0]

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(self.dtype))
        if self.params.guidance_embed:
            vec.add_(self.guidance_in(timestep_embedding(guidance, 256).to(self.dtype)))
        vec.add_(self.vector_in(y))

        txt = self.txt_in(txt)

        if self.union:
            if controlnet_mode is None:
                raise ValueError('using union-controlnet, but controlnet_mode is not a list or is empty')
            controlnet_mode = torch.tensor(controlnet_mode).to(self.device, dtype=torch.long)
            controlnet_mode = controlnet_mode.reshape([-1, 1])
            emb_controlnet_mode = self.controlnet_mode_embedder(controlnet_mode).to(self.dtype)
            txt = torch.cat([emb_controlnet_mode, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:, :1], txt_ids], dim=1)

        img = img + self.controlnet_x_embedder(controlnet_cond)

        txt_ids = txt_ids.expand(img_ids.size(0), -1, -1)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples = ()
        for block in self.transformer_blocks:
            txt, img = block(hidden_states=img, encoder_hidden_states=txt, temb=vec, image_rotary_emb=pe)
            block_res_samples = block_res_samples + (img,)

        img = torch.cat([txt, img], dim=1)

        single_block_res_samples = ()
        for block in self.single_transformer_blocks:
            img = block(hidden_states=img, temb=vec, image_rotary_emb=pe)
            single_block_res_samples = single_block_res_samples + (img[:, txt.shape[1]:],)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        controlnet_single_block_res_samples = ()
        for single_block_res_sample, single_controlnet_block in zip(single_block_res_samples, self.controlnet_single_blocks):
            single_block_res_sample = single_controlnet_block(single_block_res_sample)
            controlnet_single_block_res_samples = controlnet_single_block_res_samples + (single_block_res_sample,)

        n_single_blocks = 38
        n_double_blocks = 19

        # Expand controlnet_block_res_samples to match n_double_blocks
        expanded_controlnet_block_res_samples = []
        interval_control_double = int(np.ceil(n_double_blocks / len(controlnet_block_res_samples)))
        for i in range(n_double_blocks):
            index = i // interval_control_double
            expanded_controlnet_block_res_samples.append(controlnet_block_res_samples[index])

        # Expand controlnet_single_block_res_samples to match n_single_blocks
        expanded_controlnet_single_block_res_samples = []
        interval_control_single = int(np.ceil(n_single_blocks / len(controlnet_single_block_res_samples)))
        for i in range(n_single_blocks):
            index = i // interval_control_single
            expanded_controlnet_single_block_res_samples.append(controlnet_single_block_res_samples[index])

        return {
            "input": expanded_controlnet_block_res_samples,
            "output": expanded_controlnet_single_block_res_samples
        }

    def forward(self, x, timesteps, context, y, guidance=None, hint=None, control_type=None, **kwargs):
        bs, c, h, w = x.shape
        patch_size = 2
        x = pad_to_patch_size(x, (patch_size, patch_size))

        height_control_image, width_control_image = hint.shape[2:]
        num_channels_latents = self.in_channels // 4
        hint = self._pack_latents(
            hint,
            hint.shape[0],
            num_channels_latents,
            height_control_image,
            width_control_image,
        )
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance, control_type)

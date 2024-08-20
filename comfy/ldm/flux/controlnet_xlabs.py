#Original code can be found on: https://github.com/XLabs-AI/x-flux/blob/main/src/flux/controlnet.py

import torch
from torch import Tensor, nn
from einops import rearrange, repeat

from .layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)

from .model import Flux
import comfy.ldm.common_dit


class ControlNetFlux(Flux):
    def __init__(self, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        # add ControlNet blocks
        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(self.params.depth):
            controlnet_block = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)
            # controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)
        self.pos_embed_input = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.gradient_checkpointing = False
        self.input_hint_block = nn.Sequential(
            operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
            nn.SiLU(),
            operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
        )

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
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        controlnet_cond = self.input_hint_block(controlnet_cond)
        controlnet_cond = rearrange(controlnet_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        block_res_samples = ()

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            block_res_samples = block_res_samples + (img,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)

        return {"input": (controlnet_block_res_samples * 10)[:19]}

    def forward(self, x, timesteps, context, y, guidance=None, hint=None, **kwargs):
        hint = hint * 2.0 - 1.0

        bs, c, h, w = x.shape
        patch_size = 2
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance)

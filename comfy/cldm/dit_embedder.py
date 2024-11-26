import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from comfy.ldm.modules.diffusionmodules.mmdit import DismantledBlock, PatchEmbed, VectorEmbedder, TimestepEmbedder, get_2d_sincos_pos_embed_torch


class ControlNetEmbedder(nn.Module):

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        attention_head_dim: int,
        num_attention_heads: int,
        adm_in_channels: int,
        num_layers: int,
        main_model_double: int,
        double_y_emb: bool,
        device: torch.device,
        dtype: torch.dtype,
        pos_embed_max_size: Optional[int] = None,
        operations = None,
    ):
        super().__init__()
        self.main_model_double = main_model_double
        self.dtype = dtype
        self.hidden_size = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.x_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.hidden_size,
            strict_img_size=pos_embed_max_size is None,
            device=device,
            dtype=dtype,
            operations=operations,
        )

        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype, device=device, operations=operations)

        self.double_y_emb = double_y_emb
        if self.double_y_emb:
            self.orig_y_embedder = VectorEmbedder(
                adm_in_channels, self.hidden_size, dtype, device, operations=operations
            )
            self.y_embedder = VectorEmbedder(
                self.hidden_size, self.hidden_size, dtype, device, operations=operations
            )
        else:
            self.y_embedder = VectorEmbedder(
                adm_in_channels, self.hidden_size, dtype, device, operations=operations
            )

        self.transformer_blocks = nn.ModuleList(
            DismantledBlock(
                hidden_size=self.hidden_size, num_heads=num_attention_heads, qkv_bias=True,
                dtype=dtype, device=device, operations=operations
            )
            for _ in range(num_layers)
        )

        # self.use_y_embedder = pooled_projection_dim != self.time_text_embed.text_embedder.linear_1.in_features
        # TODO double check this logic when 8b
        self.use_y_embedder = True

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(len(self.transformer_blocks)):
            controlnet_block = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)
            self.controlnet_blocks.append(controlnet_block)

        self.pos_embed_input = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.hidden_size,
            strict_img_size=False,
            device=device,
            dtype=dtype,
            operations=operations,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hint = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        x_shape = list(x.shape)
        x = self.x_embedder(x)
        if not self.double_y_emb:
            h = (x_shape[-2] + 1) // self.patch_size
            w = (x_shape[-1] + 1) // self.patch_size
            x += get_2d_sincos_pos_embed_torch(self.hidden_size, w, h, device=x.device)
        c = self.t_embedder(timesteps, dtype=x.dtype)
        if y is not None and self.y_embedder is not None:
            if self.double_y_emb:
                y = self.orig_y_embedder(y)
            y = self.y_embedder(y)
            c = c + y

        x = x + self.pos_embed_input(hint)

        block_out = ()

        repeat = math.ceil(self.main_model_double / len(self.transformer_blocks))
        for i in range(len(self.transformer_blocks)):
            out = self.transformer_blocks[i](x, c)
            if not self.double_y_emb:
                x = out
            block_out += (self.controlnet_blocks[i](out),) * repeat

        return {"output": block_out}

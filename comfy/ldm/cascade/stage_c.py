"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import torch
from torch import nn
import math
from .common import AttnBlock, LayerNorm2d_op, ResBlock, FeedForwardBlock, TimestepBlock
# from .controlnet import ControlNetDeliverer

class UpDownBlock2d(nn.Module):
    def __init__(self, c_in, c_out, mode, enabled=True, dtype=None, device=None, operations=None):
        super().__init__()
        assert mode in ['up', 'down']
        interpolation = nn.Upsample(scale_factor=2 if mode == 'up' else 0.5, mode='bilinear',
                                    align_corners=True) if enabled else nn.Identity()
        mapping = operations.Conv2d(c_in, c_out, kernel_size=1, dtype=dtype, device=device)
        self.blocks = nn.ModuleList([interpolation, mapping] if mode == 'up' else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class StageC(nn.Module):
    def __init__(self, c_in=16, c_out=16, c_r=64, patch_size=1, c_cond=2048, c_hidden=[2048, 2048], nhead=[32, 32],
                 blocks=[[8, 24], [24, 8]], block_repeat=[[1, 1], [1, 1]], level_config=['CTA', 'CTA'],
                 c_clip_text=1280, c_clip_text_pooled=1280, c_clip_img=768, c_clip_seq=4, kernel_size=3,
                 dropout=[0.0, 0.0], self_attn=True, t_conds=['sca', 'crp'], switch_level=[False], stable_cascade_stage=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)

        # CONDITIONING
        self.clip_txt_mapper = operations.Linear(c_clip_text, c_cond, dtype=dtype, device=device)
        self.clip_txt_pooled_mapper = operations.Linear(c_clip_text_pooled, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_img_mapper = operations.Linear(c_clip_img, c_cond * c_clip_seq, dtype=dtype, device=device)
        self.clip_norm = operations.LayerNorm(c_cond, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            operations.Conv2d(c_in * (patch_size ** 2), c_hidden[0], kernel_size=1, dtype=dtype, device=device),
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6)
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == 'C':
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'A':
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'F':
                return FeedForwardBlock(c_hidden, dropout=dropout, dtype=dtype, device=device, operations=operations)
            elif block_type == 'T':
                return TimestepBlock(c_hidden, c_r, conds=t_conds, dtype=dtype, device=device, operations=operations)
            else:
                raise Exception(f'Block type {block_type} not supported')

        # BLOCKS
        # -- down blocks
        self.down_blocks = nn.ModuleList()
        self.down_downscalers = nn.ModuleList()
        self.down_repeat_mappers = nn.ModuleList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(nn.Sequential(
                    LayerNorm2d_op(operations)(c_hidden[i - 1], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i - 1], c_hidden[i], mode='down', enabled=switch_level[i - 1], dtype=dtype, device=device, operations=operations)
                ))
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.ModuleList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(block_type, c_hidden[i], nhead[i], dropout=dropout[i], self_attn=self_attn[i])
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.down_repeat_mappers.append(block_repeat_mappers)

        # -- up blocks
        self.up_blocks = nn.ModuleList()
        self.up_upscalers = nn.ModuleList()
        self.up_repeat_mappers = nn.ModuleList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(nn.Sequential(
                    LayerNorm2d_op(operations)(c_hidden[i], elementwise_affine=False, eps=1e-6),
                    UpDownBlock2d(c_hidden[i], c_hidden[i - 1], mode='up', enabled=switch_level[i - 1], dtype=dtype, device=device, operations=operations)
                ))
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.ModuleList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(block_type, c_hidden[i], nhead[i], c_skip=c_skip, dropout=dropout[i],
                                      self_attn=self_attn[i])
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.ModuleList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(operations.Conv2d(c_hidden[i], c_hidden[i], kernel_size=1, dtype=dtype, device=device))
                self.up_repeat_mappers.append(block_repeat_mappers)

        # OUTPUT
        self.clf = nn.Sequential(
            LayerNorm2d_op(operations)(c_hidden[0], elementwise_affine=False, eps=1e-6, dtype=dtype, device=device),
            operations.Conv2d(c_hidden[0], c_out * (patch_size ** 2), kernel_size=1, dtype=dtype, device=device),
            nn.PixelShuffle(patch_size),
        )

        # --- WEIGHT INIT ---
    #     self.apply(self._init_weights)  # General init
    #     nn.init.normal_(self.clip_txt_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.clip_txt_pooled_mapper.weight, std=0.02)  # conditionings
    #     nn.init.normal_(self.clip_img_mapper.weight, std=0.02)  # conditionings
    #     torch.nn.init.xavier_uniform_(self.embedding[1].weight, 0.02)  # inputs
    #     nn.init.constant_(self.clf[1].weight, 0)  # outputs
    #
    #     # blocks
    #     for level_block in self.down_blocks + self.up_blocks:
    #         for block in level_block:
    #             if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
    #                 block.channelwise[-1].weight.data *= np.sqrt(1 / sum(blocks[0]))
    #             elif isinstance(block, TimestepBlock):
    #                 for layer in block.modules():
    #                     if isinstance(layer, nn.Linear):
    #                         nn.init.constant_(layer.weight, 0)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pooled = clip_txt_pooled.unsqueeze(1)
        if len(clip_img.shape) == 2:
            clip_img = clip_img.unsqueeze(1)
        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).view(clip_txt_pooled.size(0), clip_txt_pooled.size(1) * self.c_clip_seq, -1)
        clip_img = self.clip_img_mapper(clip_img).view(clip_img.size(0), clip_img.size(1) * self.c_clip_seq, -1)
        clip = torch.cat([clip_txt, clip_txt_pool, clip_img], dim=1)
        clip = self.clip_norm(clip)
        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None, transformer_options={}):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True).to(x.dtype)
                        x = block(x)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        x = block(x, clip, transformer_options=transformer_options)
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None, transformer_options={}):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            for j in range(len(repmap) + 1):
                for k, block in enumerate(up_block):
                    if isinstance(block, ResBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  ResBlock)):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):
                            x = torch.nn.functional.interpolate(x, skip.shape[-2:], mode='bilinear',
                                                                align_corners=True)
                        if cnet is not None:
                            next_cnet = cnet.pop()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(next_cnet, size=x.shape[-2:], mode='bilinear',
                                                                  align_corners=True).to(x.dtype)
                        x = block(x, skip)
                    elif isinstance(block, AttnBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  AttnBlock)):
                        x = block(x, clip, transformer_options=transformer_options)
                    elif isinstance(block, TimestepBlock) or (
                            hasattr(block, '_fsdp_wrapped_module') and isinstance(block._fsdp_wrapped_module,
                                                                                  TimestepBlock)):
                        x = block(x, r_embed)
                    else:
                        x = block(x)
                if j < len(repmap):
                    x = repmap[j](x)
            x = upscaler(x)
        return x

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, control=None, transformer_options={}, **kwargs):
        # Process the conditioning embeddings
        r_embed = self.gen_r_embedding(r).to(dtype=x.dtype)
        for c in self.t_conds:
            t_cond = kwargs.get(c, torch.zeros_like(r))
            r_embed = torch.cat([r_embed, self.gen_r_embedding(t_cond).to(dtype=x.dtype)], dim=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        if control is not None:
            cnet = control.get("input")
        else:
            cnet = None

        # Model Blocks
        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip, cnet, transformer_options=transformer_options)
        x = self._up_decode(level_outputs, r_embed, clip, cnet, transformer_options=transformer_options)
        return self.clf(x)

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone().to(self_params.device) * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone().to(self_buffers.device) * (1 - beta)

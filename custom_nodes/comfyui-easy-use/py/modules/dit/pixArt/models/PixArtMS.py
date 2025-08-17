# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
from tqdm import tqdm
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from .utils import auto_grad_checkpoint, to_2tuple
from .PixArt_blocks import t2i_modulate, CaptionEmbedder, AttentionKVCompress, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, SizeEmbedder
from .PixArt import PixArt, get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PixArtMSBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., input_size=None,
                 sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


### Core PixArt Model ###
class PixArtMS(PixArt):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            learn_sigma=True,
            pred_sigma=True,
            drop_path: float = 0.,
            caption_channels=4096,
            pe_interpolation=None,
            pe_precision=None,
            config=None,
            model_max_length=120,
            micro_condition=True,
            qk_norm=False,
            kv_compress_config=None,
            **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            pe_interpolation=pe_interpolation,
            config=config,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            kv_compress_config=kv_compress_config,
            **kwargs,
        )
        self.dtype = torch.get_default_dtype()
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        self.micro_conditioning = micro_condition
        if self.micro_conditioning:
            self.csize_embedder = SizeEmbedder(hidden_size//3)  # c_size embed
            self.ar_embedder = SizeEmbedder(hidden_size//3)     # aspect ratio embed
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        if kv_compress_config is None:
            kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }
        self.blocks = nn.ModuleList([
            PixArtMSBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=kv_compress_config['sampling'],
                sr_ratio=int(kv_compress_config['scale_factor']) if i in kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

    def forward_raw(self, x, t, y, mask=None, data_info=None, **kwargs):
        """
        Original forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = t.to(self.dtype)
        y = y.to(self.dtype)
        
        pe_interpolation = self.pe_interpolation
        if pe_interpolation is None or self.pe_precision is not None:
            # calculate pe_interpolation on-the-fly
            pe_interpolation = round((x.shape[-1]+x.shape[-2])/2.0 / (512/8.0), self.pe_precision or 0)

        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=pe_interpolation,
                base_size=self.base_size
            )
        ).unsqueeze(0).to(device=x.device, dtype=self.dtype)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            csize = self.csize_embedder(c_size, bs)  # (N, D)
            ar = self.ar_embedder(ar, bs)  # (N, D)
            t = t + torch.cat([csize, ar], dim=1)

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, (self.h, self.w), **kwargs)  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        
        return x

    def forward(self, x, timesteps, context, img_hw=None, aspect_ratio=None, **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 120, C) conditioning
        img_hw: height|width conditioning
        aspect_ratio: aspect ratio conditioning
        """
        ## size/ar from cond with fallback based on the latent image shape.
        bs = x.shape[0]
        data_info = {}
        if img_hw is None:
            data_info["img_hw"] = torch.tensor(
                [[x.shape[2]*8, x.shape[3]*8]],
                dtype=self.dtype,
                device=x.device
            ).repeat(bs, 1)
        else:
            data_info["img_hw"] = img_hw.to(dtype=x.dtype, device=x.device)
        if aspect_ratio is None or True:
            data_info["aspect_ratio"] = torch.tensor(
                [[x.shape[2]/x.shape[3]]],
                dtype=self.dtype,
                device=x.device
            ).repeat(bs, 1)
        else:
            data_info["aspect_ratio"] = aspect_ratio.to(dtype=x.dtype, device=x.device)

        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_raw(
            x = x.to(self.dtype),
            t = timesteps.to(self.dtype),
            y = context.to(self.dtype),
            data_info=data_info,
        )

        ## only return EPS
        out = out.to(torch.float)
        eps, rest = out[:, :self.in_channels], out[:, self.in_channels:]
        return eps

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

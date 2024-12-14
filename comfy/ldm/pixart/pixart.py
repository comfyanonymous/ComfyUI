# Based on:
# https://github.com/PixArt-alpha/PixArt-alpha [Apache 2.0 license]
# https://github.com/PixArt-alpha/PixArt-sigma [Apache 2.0 license]
import torch
import torch.nn as nn
import numpy as np

from .blocks import (
    t2i_modulate,
    CaptionEmbedder,
    AttentionKVCompress,
    MultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
)
from comfy.ldm.modules.diffusionmodules.mmdit import PatchEmbed, TimestepEmbedder, Mlp, get_1d_sincos_pos_embed_from_grid_torch


class PixArtBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm (adaLN-single) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0, input_size=None, sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
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
        self.drop_path = nn.Identity() #DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.sampling = sampling
        self.sr_ratio = sr_ratio

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


### Core PixArt Model ###
class PixArt(nn.Module):
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
            pred_sigma=True,
            drop_path: float = 0.,
            caption_channels=4096,
            pe_interpolation=1.0,
            pe_precision=None,
            config=None,
            model_max_length=120,
            qk_norm=False,
            kv_compress_config=None,
            **kwargs,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.pe_precision = pe_precision
        self.depth = depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob,
            act_layer=approx_gelu, token_num=model_max_length
		)
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.kv_compress_config = kv_compress_config
        if kv_compress_config is None:
            self.kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }
        self.blocks = nn.ModuleList([
            PixArtBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=self.kv_compress_config['sampling'],
                sr_ratio=int(
                    self.kv_compress_config['scale_factor']
                ) if i in self.kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

    def forward_raw(self, x, t, y, mask=None, data_info=None):
        """
        Original forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = t.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
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
            x = block(x, y, t0, y_lens)  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward(self, x, timesteps, context, y=None, **kwargs):
        """
        Forward pass that adapts comfy input to original forward function
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        timesteps: (N,) tensor of diffusion timesteps
        context: (N, 1, 120, C) conditioning
        y: extra conditioning.
        """
        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_raw(
            x = x.to(self.dtype),
            t = timesteps.to(self.dtype),
            y = context.to(self.dtype),
        )

        ## only return EPS
        out = out.to(torch.float)
        eps, _ = out[:, :self.in_channels], out[:, self.in_channels:]
        return eps

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

def get_2d_sincos_pos_embed_torch(embed_dim, w, h, pe_interpolation=1.0, base_size=16, device=None, dtype=torch.float32):
    grid_h, grid_w = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype) / (h/base_size) / pe_interpolation,
        torch.arange(w, device=device, dtype=dtype) / (w/base_size) / pe_interpolation,
        indexing='ij'
    )
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_h, device=device, dtype=dtype)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid_w, device=device, dtype=dtype)
    emb = torch.cat([emb_w, emb_h], dim=1)  # (H*W, D)
    return emb

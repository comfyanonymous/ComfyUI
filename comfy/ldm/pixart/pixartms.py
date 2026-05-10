# Based on:
# https://github.com/PixArt-alpha/PixArt-alpha [Apache 2.0 license]
# https://github.com/PixArt-alpha/PixArt-sigma [Apache 2.0 license]
import torch
import torch.nn as nn

from .blocks import (
    t2i_modulate,
    CaptionEmbedder,
    AttentionKVCompress,
    MultiHeadCrossAttention,
    T2IFinalLayer,
    SizeEmbedder,
)
from comfy.ldm.modules.diffusionmodules.mmdit import TimestepEmbedder, PatchEmbed, Mlp, get_1d_sincos_pos_embed_from_grid_torch


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

class PixArtMSBlock(nn.Module):
    """
    A PixArt block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., input_size=None,
                 sampling=None, sr_ratio=1, qk_norm=False, dtype=None, device=None, operations=None, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, dtype=dtype, device=device, operations=operations, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(
            hidden_size, num_heads, dtype=dtype, device=device, operations=operations, **block_kwargs
        )
        self.norm2 = operations.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, dtype=dtype, device=device)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu,
            dtype=dtype, device=device, operations=operations
        )
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask=None, HW=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None].to(dtype=x.dtype, device=x.device) + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x = x + (gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        x = x + self.cross_attn(x, y, mask)
        x = x + (gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


### Core PixArt Model ###
class PixArtMS(nn.Module):
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
            dtype=None,
            device=None,
            operations=None,
            **kwargs,
    ):
        nn.Module.__init__(self)
        self.dtype = dtype
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.pe_precision = pe_precision
        self.hidden_size = hidden_size
        self.depth = depth

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            operations.Linear(hidden_size, 6 * hidden_size, bias=True, dtype=dtype, device=device)
        )
        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            operations=operations
        )
        self.t_embedder = TimestepEmbedder(
            hidden_size, dtype=dtype, device=device, operations=operations,
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob,
            act_layer=approx_gelu, token_num=model_max_length,
            dtype=dtype, device=device, operations=operations,
        )

        self.micro_conditioning = micro_condition
        if self.micro_conditioning:
            self.csize_embedder = SizeEmbedder(hidden_size//3, dtype=dtype, device=device, operations=operations)
            self.ar_embedder = SizeEmbedder(hidden_size//3, dtype=dtype, device=device, operations=operations)

        # For fixed sin-cos embedding:
        # num_patches = (input_size // patch_size) * (input_size // patch_size)
        # self.base_size = input_size // self.patch_size
        # self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

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
                sampling=kv_compress_config['sampling'],
                sr_ratio=int(kv_compress_config['scale_factor']) if i in kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
                dtype=dtype,
                device=device,
                operations=operations,
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(
            hidden_size, patch_size, self.out_channels, dtype=dtype, device=device, operations=operations
        )

    def forward_orig(self, x, timestep, y, mask=None, c_size=None, c_ar=None, **kwargs):
        """
        Original forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) conditioning
        ar: (N, 1): aspect ratio
        cs: (N ,2) size conditioning for height/width
        """
        B, C, H, W = x.shape
        c_res = (H + W) // 2
        pe_interpolation = self.pe_interpolation
        if pe_interpolation is None or self.pe_precision is not None:
            # calculate pe_interpolation on-the-fly
            pe_interpolation = round(c_res / (512/8.0), self.pe_precision or 0)

        pos_embed = get_2d_sincos_pos_embed_torch(
            self.hidden_size,
            h=(H // self.patch_size),
            w=(W // self.patch_size),
            pe_interpolation=pe_interpolation,
            base_size=((round(c_res / 64) * 64) // self.patch_size),
            device=x.device,
            dtype=x.dtype,
        ).unsqueeze(0)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep, x.dtype)  # (N, D)

        if self.micro_conditioning and (c_size is not None and c_ar is not None):
            bs = x.shape[0]
            c_size = self.csize_embedder(c_size, bs)  # (N, D)
            c_ar = self.ar_embedder(c_ar, bs)  # (N, D)
            t = t + torch.cat([c_size, c_ar], dim=1)

        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = None
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = block(x, y, t0, y_lens, (H, W), **kwargs)  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, H, W)  # (N, out_channels, H, W)

        return x

    def forward(self, x, timesteps, context, c_size=None, c_ar=None, **kwargs):
        B, C, H, W = x.shape

        # Fallback for missing microconds
        if self.micro_conditioning:
            if c_size is None:
                c_size = torch.tensor([H*8, W*8], dtype=x.dtype, device=x.device).repeat(B, 1)

            if c_ar is None:
                c_ar = torch.tensor([H/W], dtype=x.dtype, device=x.device).repeat(B, 1)

        ## Still accepts the input w/o that dim but returns garbage
        if len(context.shape) == 3:
            context = context.unsqueeze(1)

        ## run original forward pass
        out = self.forward_orig(x, timesteps, context, c_size=c_size, c_ar=c_ar)

        ## only return EPS
        if self.pred_sigma:
            return out[:, :self.in_channels]
        return out

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = h // self.patch_size
        w = w // self.patch_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

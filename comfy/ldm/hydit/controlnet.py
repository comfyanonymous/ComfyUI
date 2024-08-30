from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import checkpoint

from comfy.ldm.modules.diffusionmodules.mmdit import (
    Mlp,
    TimestepEmbedder,
    PatchEmbed,
    RMSNorm,
)
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from .poolers import AttentionPool

import comfy.latent_formats
from .models import HunYuanDiTBlock, calc_rope

from .posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop


class HunYuanControlNet(nn.Module):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Inherit PeftAdapterMixin to be compatible with the PEFT training pipeline.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    def __init__(
        self,
        input_size: tuple = 128,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1408,
        depth: int = 40,
        num_heads: int = 16,
        mlp_ratio: float = 4.3637,
        text_states_dim=1024,
        text_states_dim_t5=2048,
        text_len=77,
        text_len_t5=256,
        qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
        size_cond=False,
        use_style_cond=False,
        learn_sigma=True,
        norm="layer",
        log_fn: callable = print,
        attn_precision=None,
        dtype=None,
        device=None,
        operations=None,
        **kwargs,
    ):
        super().__init__()
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = text_states_dim
        self.text_states_dim_t5 = text_states_dim_t5
        self.text_len = text_len
        self.text_len_t5 = text_len_t5
        self.size_cond = size_cond
        self.use_style_cond = use_style_cond
        self.norm = norm
        self.dtype = dtype
        self.latent_format = comfy.latent_formats.SDXL

        self.mlp_t5 = nn.Sequential(
            nn.Linear(
                self.text_states_dim_t5,
                self.text_states_dim_t5 * 4,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            nn.Linear(
                self.text_states_dim_t5 * 4,
                self.text_states_dim,
                bias=True,
                dtype=dtype,
                device=device,
            ),
        )
        # learnable replace
        self.text_embedding_padding = nn.Parameter(
            torch.randn(
                self.text_len + self.text_len_t5,
                self.text_states_dim,
                dtype=dtype,
                device=device,
            )
        )

        # Attention pooling
        pooler_out_dim = 1024
        self.pooler = AttentionPool(
            self.text_len_t5,
            self.text_states_dim_t5,
            num_heads=8,
            output_dim=pooler_out_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        if self.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256

        if self.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = nn.Embedding(
                1, hidden_size, dtype=dtype, device=device
            )
            self.extra_in_dim += hidden_size

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            hidden_size,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.t_embedder = TimestepEmbedder(
            hidden_size, dtype=dtype, device=device, operations=operations
        )
        self.extra_embedder = nn.Sequential(
            operations.Linear(
                self.extra_in_dim, hidden_size * 4, dtype=dtype, device=device
            ),
            nn.SiLU(),
            operations.Linear(
                hidden_size * 4, hidden_size, bias=True, dtype=dtype, device=device
            ),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches

        # HUnYuanDiT Blocks
        self.blocks = nn.ModuleList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.text_states_dim,
                    qk_norm=qk_norm,
                    norm_type=self.norm,
                    skip=False,
                    attn_precision=attn_precision,
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for _ in range(19)
            ]
        )

        # Input zero linear for the first block
        self.before_proj = operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)


        # Output zero linear for the every block
        self.after_proj_list = nn.ModuleList(
            [

                    operations.Linear(
                        self.hidden_size, self.hidden_size, dtype=dtype, device=device
                    )
                for _ in range(len(self.blocks))
            ]
        )

    def forward(
        self,
        x,
        hint,
        timesteps,
        context,#encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        return_dict=False,
        **kwarg,
    ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: torch.Tensor
            (B, D, H, W)
        t: torch.Tensor
            (B)
        encoder_hidden_states: torch.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: torch.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: torch.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: torch.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: torch.Tensor
            (B, 6)
        style: torch.Tensor
            (B)
        cos_cis_img: torch.Tensor
        sin_cis_img: torch.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        condition = hint
        if condition.shape[0] == 1:
            condition = torch.repeat_interleave(condition, x.shape[0], dim=0)

        text_states = context  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.bool()  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()  # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5)).view(b_t5, l_t5, -1)

        padding = comfy.ops.cast_to_input(self.text_embedding_padding, text_states)

        text_states[:, -self.text_len :] = torch.where(
            text_states_mask[:, -self.text_len :].unsqueeze(2),
            text_states[:, -self.text_len :],
            padding[: self.text_len],
        )
        text_states_t5[:, -self.text_len_t5 :] = torch.where(
            text_states_t5_mask[:, -self.text_len_t5 :].unsqueeze(2),
            text_states_t5[:, -self.text_len_t5 :],
            padding[self.text_len :],
        )

        text_states = torch.cat([text_states, text_states_t5], dim=1)  # 2,205，1024

        # _, _, oh, ow = x.shape
        # th, tw = oh // self.patch_size, ow // self.patch_size

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = calc_rope(
            x, self.patch_size, self.hidden_size // self.num_heads
        )  # (cos_cis_img, sin_cis_img)

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(timesteps, dtype=self.dtype)
        x = self.x_embedder(x)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        # Build image meta size tokens if applicable
        # if image_meta_size is not None:
        #     image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)   # [B * 6, 256]
        #     if image_meta_size.dtype != self.dtype:
        #         image_meta_size = image_meta_size.half()
        #     image_meta_size = image_meta_size.view(-1, 6 * 256)
        #     extra_vec = torch.cat([extra_vec, image_meta_size], dim=1)  # [B, D + 6 * 256]

        # Build style tokens
        if style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = torch.cat([extra_vec, style_embedding], dim=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Deal with Condition =========================
        condition = self.x_embedder(condition)

        # ========================= Forward pass through HunYuanDiT blocks =========================
        controls = []
        x = x + self.before_proj(condition)  # add condition
        for layer, block in enumerate(self.blocks):
            x = block(x, c, text_states, freqs_cis_img)
            controls.append(self.after_proj_list[layer](x))  # zero linear for output

        return {"output": controls}

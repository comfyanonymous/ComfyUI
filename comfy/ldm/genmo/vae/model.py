#original code from https://github.com/genmoai/models under apache 2.0 license
#adapted to ComfyUI

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import comfy.ops
ops = comfy.ops.disable_weight_init

# import mochi_preview.dit.joint_model.context_parallel as cp
# from mochi_preview.vae.cp_conv import cp_pass_frames, gather_all_frames


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class GroupNormSpatial(ops.GroupNorm):
    """
    GroupNorm applied per-frame.
    """

    def forward(self, x: torch.Tensor, *, chunk_size: int = 8):
        B, C, T, H, W = x.shape
        x = rearrange(x, "B C T H W -> (B T) C H W")
        # Run group norm in chunks.
        output = torch.empty_like(x)
        for b in range(0, B * T, chunk_size):
            output[b : b + chunk_size] = super().forward(x[b : b + chunk_size])
        return rearrange(output, "(B T) C H W -> B C T H W", B=B, T=T)

class PConv3d(ops.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        causal: bool = True,
        context_parallel: bool = True,
        **kwargs,
    ):
        self.causal = causal
        self.context_parallel = context_parallel
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, height_pad, width_pad),
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        # Compute padding amounts.
        context_size = self.kernel_size[0] - 1
        if self.causal:
            pad_front = context_size
            pad_back = 0
        else:
            pad_front = context_size // 2
            pad_back = context_size - pad_front

        # Apply padding.
        assert self.padding_mode == "replicate"  # DEBUG
        mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
        x = F.pad(x, (0, 0, 0, 0, pad_front, pad_back), mode=mode)
        return super().forward(x)


class Conv1x1(ops.Linear):
    """*1x1 Conv implemented with a linear layer."""

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, *] or [B, *, C].

        Returns:
            x: Output tensor. Shape: [B, C', *] or [B, *, C'].
        """
        x = x.movedim(1, -1)
        x = super().forward(x)
        x = x.movedim(-1, 1)
        return x


class DepthToSpaceTime(nn.Module):
    def __init__(
        self,
        temporal_expansion: int,
        spatial_expansion: int,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

    # When printed, this module should show the temporal and spatial expansion factors.
    def extra_repr(self):
        return f"texp={self.temporal_expansion}, sexp={self.spatial_expansion}"

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].

        Returns:
            x: Rearranged tensor. Shape: [B, C/(st*s*s), T*st, H*s, W*s].
        """
        x = rearrange(
            x,
            "B (C st sh sw) T H W -> B C (T st) (H sh) (W sw)",
            st=self.temporal_expansion,
            sh=self.spatial_expansion,
            sw=self.spatial_expansion,
        )

        # cp_rank, _ = cp.get_cp_rank_size()
        if self.temporal_expansion > 1: # and cp_rank == 0:
            # Drop the first self.temporal_expansion - 1 frames.
            # This is because we always want the 3x3x3 conv filter to only apply
            # to the first frame, and the first frame doesn't need to be repeated.
            assert all(x.shape)
            x = x[:, :, self.temporal_expansion - 1 :]
            assert all(x.shape)

        return x


def norm_fn(
    in_channels: int,
    affine: bool = True,
):
    return GroupNormSpatial(affine=affine, num_groups=32, num_channels=in_channels)


class ResBlock(nn.Module):
    """Residual block that preserves the spatial dimensions."""

    def __init__(
        self,
        channels: int,
        *,
        affine: bool = True,
        attn_block: Optional[nn.Module] = None,
        padding_mode: str = "replicate",
        causal: bool = True,
    ):
        super().__init__()
        self.channels = channels

        assert causal
        self.stack = nn.Sequential(
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            PConv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=True,
                # causal=causal,
            ),
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            PConv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=True,
                # causal=causal,
            ),
        )

        self.attn_block = attn_block if attn_block else nn.Identity()

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
        """
        residual = x
        x = self.stack(x)
        x = x + residual
        del residual

        return self.attn_block(x)


class CausalUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        *,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        **block_kwargs,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(block_fn(in_channels, **block_kwargs))
        self.blocks = nn.Sequential(*blocks)

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        # Change channels in the final convolution layer.
        self.proj = Conv1x1(
            in_channels,
            out_channels * temporal_expansion * (spatial_expansion**2),
        )

        self.d2st = DepthToSpaceTime(
            temporal_expansion=temporal_expansion, spatial_expansion=spatial_expansion
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.proj(x)
        x = self.d2st(x)
        return x


def block_fn(channels, *, has_attention: bool = False, **block_kwargs):
    assert has_attention is False #NOTE: if this is ever true add back the attention code.

    attn_block = None #AttentionBlock(channels) if has_attention else None

    return ResBlock(
        channels, affine=True, attn_block=attn_block, **block_kwargs
    )


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks,
        *,
        temporal_reduction=2,
        spatial_reduction=2,
        **block_kwargs,
    ):
        """
        Downsample block for the VAE encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks.
            temporal_reduction: Temporal reduction factor.
            spatial_reduction: Spatial reduction factor.
        """
        super().__init__()
        layers = []

        # Change the channel count in the strided convolution.
        # This lets the ResBlock have uniform channel count,
        # as in ConvNeXt.
        assert in_channels != out_channels
        layers.append(
            PConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(temporal_reduction, spatial_reduction, spatial_reduction),
                stride=(temporal_reduction, spatial_reduction, spatial_reduction),
                padding_mode="replicate",
                bias=True,
            )
        )

        for _ in range(num_res_blocks):
            layers.append(block_fn(out_channels, **block_kwargs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def add_fourier_features(inputs: torch.Tensor, start=6, stop=8, step=1):
    num_freqs = (stop - start) // step
    assert inputs.ndim == 5
    C = inputs.size(1)

    # Create Base 2 Fourier features.
    freqs = torch.arange(start, stop, step, dtype=inputs.dtype, device=inputs.device)
    assert num_freqs == len(freqs)
    w = torch.pow(2.0, freqs) * (2 * torch.pi)  # [num_freqs]
    C = inputs.shape[1]
    w = w.repeat(C)[None, :, None, None, None]  # [1, C * num_freqs, 1, 1, 1]

    # Interleaved repeat of input channels to match w.
    h = inputs.repeat_interleave(num_freqs, dim=1)  # [B, C * num_freqs, T, H, W]
    # Scale channels by frequency.
    h = w * h

    return torch.cat(
        [
            inputs,
            torch.sin(h),
            torch.cos(h),
        ],
        dim=1,
    )


class FourierFeatures(nn.Module):
    def __init__(self, start: int = 6, stop: int = 8, step: int = 1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs):
        """Add Fourier features to inputs.

        Args:
            inputs: Input tensor. Shape: [B, C, T, H, W]

        Returns:
            h: Output tensor. Shape: [B, (1 + 2 * num_freqs) * C, T, H, W]
        """
        return add_fourier_features(inputs, self.start, self.stop, self.step)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int = 3,
        latent_dim: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: List[int],
        temporal_expansions: Optional[List[int]] = None,
        spatial_expansions: Optional[List[int]] = None,
        has_attention: List[bool],
        output_norm: bool = True,
        nonlinearity: str = "silu",
        output_nonlinearity: str = "silu",
        causal: bool = True,
        **block_kwargs,
    ):
        super().__init__()
        self.input_channels = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.output_nonlinearity = output_nonlinearity
        assert nonlinearity == "silu"
        assert causal

        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        assert len(num_res_blocks) == self.num_up_blocks + 2

        blocks = []

        first_block = [
            nn.Conv3d(latent_dim, ch[-1], kernel_size=(1, 1, 1))
        ]  # Input layer.
        # First set of blocks preserve channel count.
        for _ in range(num_res_blocks[-1]):
            first_block.append(
                block_fn(
                    ch[-1],
                    has_attention=has_attention[-1],
                    causal=causal,
                    **block_kwargs,
                )
            )
        blocks.append(nn.Sequential(*first_block))

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        upsample_block_fn = CausalUpsampleBlock

        for i in range(self.num_up_blocks):
            block = upsample_block_fn(
                ch[-i - 1],
                ch[-i - 2],
                num_res_blocks=num_res_blocks[-i - 2],
                has_attention=has_attention[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
                causal=causal,
                **block_kwargs,
            )
            blocks.append(block)

        assert not output_norm

        # Last block. Preserve channel count.
        last_block = []
        for _ in range(num_res_blocks[0]):
            last_block.append(
                block_fn(
                    ch[0], has_attention=has_attention[0], causal=causal, **block_kwargs
                )
            )
        blocks.append(nn.Sequential(*last_block))

        self.blocks = nn.ModuleList(blocks)
        self.output_proj = Conv1x1(ch[0], out_channels)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Latent tensor. Shape: [B, input_channels, t, h, w]. Scaled [-1, 1].

        Returns:
            x: Reconstructed video tensor. Shape: [B, C, T, H, W]. Scaled to [-1, 1].
               T + 1 = (t - 1) * 4.
               H = h * 16, W = w * 16.
        """
        for block in self.blocks:
            x = block(x)

        if self.output_nonlinearity == "silu":
            x = F.silu(x, inplace=not self.training)
        else:
            assert (
                not self.output_nonlinearity
            )  # StyleGAN3 omits the to-RGB nonlinearity.

        return self.output_proj(x).contiguous()


class VideoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None #TODO once the model releases
        self.decoder = Decoder(
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 4, 6],
            temporal_expansions=[1, 2, 3],
            spatial_expansions=[2, 2, 2],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            has_attention=[False, False, False, False, False],
            padding_mode="replicate",
            output_norm=False,
            nonlinearity="silu",
            output_nonlinearity="silu",
            causal=True,
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

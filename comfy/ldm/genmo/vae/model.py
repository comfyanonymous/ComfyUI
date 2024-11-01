#original code from https://github.com/genmoai/models under apache 2.0 license
#adapted to ComfyUI

from typing import Callable, List, Optional, Tuple, Union
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from comfy.ldm.modules.attention import optimized_attention

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
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels

        assert causal
        self.stack = nn.Sequential(
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            PConv3d(
                in_channels=channels,
                out_channels=channels // 2 if prune_bottleneck else channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=bias,
                causal=causal,
            ),
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            PConv3d(
                in_channels=channels // 2 if prune_bottleneck else channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=bias,
                causal=causal,
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


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        out_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out = nn.Linear(dim, dim, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute temporal self-attention.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
            chunk_size: Chunk size for large tensors.

        Returns:
            x: Output tensor. Shape: [B, C, T, H, W].
        """
        B, _, T, H, W = x.shape

        if T == 1:
            # No attention for single frame.
            x = x.movedim(1, -1)  # [B, C, T, H, W] -> [B, T, H, W, C]
            qkv = self.qkv(x)
            _, _, x = qkv.chunk(3, dim=-1)  # Throw away queries and keys.
            x = self.out(x)
            return x.movedim(-1, 1)  # [B, T, H, W, C] -> [B, C, T, H, W]

        # 1D temporal attention.
        x = rearrange(x, "B C t h w -> (B h w) t C")
        qkv = self.qkv(x)

        # Input: qkv with shape [B, t, 3 * num_heads * head_dim]
        # Output: x with shape [B, num_heads, t, head_dim]
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(2)

        if self.qk_norm:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)

        x = optimized_attention(q, k, v, self.num_heads, skip_reshape=True)

        assert x.size(0) == q.size(0)

        x = self.out(x)
        x = rearrange(x, "(B h w) t C -> B C t h w", B=B, h=H, w=W)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        **attn_kwargs,
    ) -> None:
        super().__init__()
        self.norm = norm_fn(dim)
        self.attn = Attention(dim, **attn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


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


def block_fn(channels, *, affine: bool = True, has_attention: bool = False, **block_kwargs):
    attn_block = AttentionBlock(channels) if has_attention else None
    return ResBlock(channels, affine=affine, attn_block=attn_block, **block_kwargs)


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
                # First layer in each block always uses replicate padding
                padding_mode="replicate",
                bias=block_kwargs["bias"],
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
            ops.Conv3d(latent_dim, ch[-1], kernel_size=(1, 1, 1))
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

class LatentDistribution:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        """Initialize latent distribution.

        Args:
            mean: Mean of the distribution. Shape: [B, C, T, H, W].
            logvar: Logarithm of variance of the distribution. Shape: [B, C, T, H, W].
        """
        assert mean.shape == logvar.shape
        self.mean = mean
        self.logvar = logvar

    def sample(self, temperature=1.0, generator: torch.Generator = None, noise=None):
        if temperature == 0.0:
            return self.mean

        if noise is None:
            noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        else:
            assert noise.device == self.mean.device
            noise = noise.to(self.mean.dtype)

        if temperature != 1.0:
            raise NotImplementedError(f"Temperature {temperature} is not supported.")

        # Just Gaussian sample with no scaling of variance.
        return noise * torch.exp(self.logvar * 0.5) + self.mean

    def mode(self):
        return self.mean

class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: List[int],
        latent_dim: int,
        temporal_reductions: List[int],
        spatial_reductions: List[int],
        prune_bottlenecks: List[bool],
        has_attentions: List[bool],
        affine: bool = True,
        bias: bool = True,
        input_is_conv_1x1: bool = False,
        padding_mode: str,
    ):
        super().__init__()
        self.temporal_reductions = temporal_reductions
        self.spatial_reductions = spatial_reductions
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.latent_dim = latent_dim

        self.fourier_features = FourierFeatures()
        ch = [mult * base_channels for mult in channel_multipliers]
        num_down_blocks = len(ch) - 1
        assert len(num_res_blocks) == num_down_blocks + 2

        layers = (
            [ops.Conv3d(in_channels, ch[0], kernel_size=(1, 1, 1), bias=True)]
            if not input_is_conv_1x1
            else [Conv1x1(in_channels, ch[0])]
        )

        assert len(prune_bottlenecks) == num_down_blocks + 2
        assert len(has_attentions) == num_down_blocks + 2
        block = partial(block_fn, padding_mode=padding_mode, affine=affine, bias=bias)

        for _ in range(num_res_blocks[0]):
            layers.append(block(ch[0], has_attention=has_attentions[0], prune_bottleneck=prune_bottlenecks[0]))
        prune_bottlenecks = prune_bottlenecks[1:]
        has_attentions = has_attentions[1:]

        assert len(temporal_reductions) == len(spatial_reductions) == len(ch) - 1
        for i in range(num_down_blocks):
            layer = DownsampleBlock(
                ch[i],
                ch[i + 1],
                num_res_blocks=num_res_blocks[i + 1],
                temporal_reduction=temporal_reductions[i],
                spatial_reduction=spatial_reductions[i],
                prune_bottleneck=prune_bottlenecks[i],
                has_attention=has_attentions[i],
                affine=affine,
                bias=bias,
                padding_mode=padding_mode,
            )

            layers.append(layer)

        # Additional blocks.
        for _ in range(num_res_blocks[-1]):
            layers.append(block(ch[-1], has_attention=has_attentions[-1], prune_bottleneck=prune_bottlenecks[-1]))

        self.layers = nn.Sequential(*layers)

        # Output layers.
        self.output_norm = norm_fn(ch[-1])
        self.output_proj = Conv1x1(ch[-1], 2 * latent_dim, bias=False)

    @property
    def temporal_downsample(self):
        return math.prod(self.temporal_reductions)

    @property
    def spatial_downsample(self):
        return math.prod(self.spatial_reductions)

    def forward(self, x) -> LatentDistribution:
        """Forward pass.

        Args:
            x: Input video tensor. Shape: [B, C, T, H, W]. Scaled to [-1, 1]

        Returns:
            means: Latent tensor. Shape: [B, latent_dim, t, h, w]. Scaled [-1, 1].
                   h = H // 8, w = W // 8, t - 1 = (T - 1) // 6
            logvar: Shape: [B, latent_dim, t, h, w].
        """
        assert x.ndim == 5, f"Expected 5D input, got {x.shape}"
        x = self.fourier_features(x)

        x = self.layers(x)

        x = self.output_norm(x)
        x = F.silu(x, inplace=True)
        x = self.output_proj(x)

        means, logvar = torch.chunk(x, 2, dim=1)

        assert means.ndim == 5
        assert logvar.shape == means.shape
        assert means.size(1) == self.latent_dim

        return LatentDistribution(means, logvar)


class VideoVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            in_channels=15,
            base_channels=64,
            channel_multipliers=[1, 2, 4, 6],
            num_res_blocks=[3, 3, 4, 6, 3],
            latent_dim=12,
            temporal_reductions=[1, 2, 3],
            spatial_reductions=[2, 2, 2],
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate"
        )
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
        return self.encoder(x).mode()

    def decode(self, x):
        return self.decoder(x)

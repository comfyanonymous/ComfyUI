# Rewritten from diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

import comfy.model_management
import comfy.ops
ops = comfy.ops.disable_weight_init


class RMSNorm(ops.RMSNorm):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, bias=False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            self.bias = nn.Parameter(torch.empty(dim)) if bias else None

    def forward(self, x):
        x = super().forward(x)
        if self.elementwise_affine:
            if self.bias is not None:
                x = x + comfy.model_management.cast_to(self.bias, dtype=x.dtype, device=x.device)
        return x


def get_normalization(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "group_norm":
        return ops.GroupNorm(num_groups, num_features)
    elif norm_type == "layer_norm":
        return ops.LayerNorm(num_features)
    elif norm_type == "rms_norm":
        return RMSNorm(num_features, eps=eps, elementwise_affine=True, bias=True)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "silu":
        return nn.SiLU()
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(0.2)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
        act_fn: str = "relu6",
    ) -> None:
        super().__init__()

        self.norm_type = norm_type
        self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
        self.conv1 = ops.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = ops.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_normalization(norm_type, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual

class SanaMultiscaleAttentionProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        channels = 3 * in_channels
        self.proj_in = ops.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.proj_out = ops.Conv2d(channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

class SanaMultiscaleLinearAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: tuple = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):
        super().__init__()

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (
            int(in_channels // attention_head_dim * mult)
            if num_attention_heads is None
            else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = ops.Linear(in_channels, inner_dim, bias=False)
        self.to_k = ops.Linear(in_channels, inner_dim, bias=False)
        self.to_v = ops.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(
                SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size)
            )

        self.nonlinearity = nn.ReLU()
        self.to_out = ops.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, out_channels)

    def apply_linear_attention(self, query, key, value):
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
        return hidden_states

    def apply_quadratic_attention(self, query, key, value):
        scores = torch.matmul(key.transpose(-1, -2), query)
        scores = scores.to(dtype=torch.float32)
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        hidden_states = torch.matmul(value, scores.to(value.dtype))
        return hidden_states

    def forward(self, hidden_states):
        height, width = hidden_states.shape[-2:]
        if height * width > self.attention_head_dim:
            use_linear_attention = True
        else:
            use_linear_attention = False

        residual = hidden_states

        batch_size, _, height, width = list(hidden_states.size())
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in self.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))

        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            # for linear attention upcast hidden_states to float32
            hidden_states = hidden_states.to(dtype=torch.float32)

        hidden_states = hidden_states.reshape(batch_size, -1, 3 * self.attention_head_dim, height * width)

        query, key, value = hidden_states.chunk(3, dim=2)
        query = self.nonlinearity(query)
        key = self.nonlinearity(key)

        if use_linear_attention:
            hidden_states = self.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = self.apply_quadratic_attention(query, key, value)

        hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = self.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.norm_type == "rms_norm":
            hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm_out(hidden_states)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mult: float = 1.0,
        attention_head_dim: int = 32,
        qkv_multiscales: tuple = (5,),
        norm_type: str = "batch_norm",
    ) -> None:
        super().__init__()

        self.attn = SanaMultiscaleLinearAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            mult=mult,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
        )

        self.conv_out = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type="rms_norm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.conv_out(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: str = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = ops.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = ops.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = ops.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


def get_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: tuple = (),
):
    if block_type == "ResBlock":
        block = ResBlock(in_channels, out_channels, norm_type, act_fn)
    elif block_type == "EfficientViTBlock":
        block = EfficientViTBlock(
            in_channels,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            qkv_multiscales=qkv_mutliscales
        )
    else:
        raise ValueError(f"Block with {block_type=} is not supported.")

    return block


class DCDownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = ops.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2
        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = ops.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: str or tuple = "ResBlock",
        block_out_channels: tuple = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: tuple = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: tuple = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "pixel_unshuffle",
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks

        if layers_per_block[0] > 0:
            self.conv_in = ops.Conv2d(
                in_channels,
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                downsample=downsample_block_type == "pixel_unshuffle",
                shortcut=False,
            )

        down_blocks = []
        for i, (out_channel, num_layers) in enumerate(zip(block_out_channels, layers_per_block)):
            down_block_list = []

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type="rms_norm",
                    act_fn="silu",
                    qkv_mutliscales=qkv_multiscales[i],
                )
                down_block_list.append(block)

            if i < num_blocks - 1 and num_layers > 0:
                downsample_block = DCDownBlock2d(
                    in_channels=out_channel,
                    out_channels=block_out_channels[i + 1],
                    downsample=downsample_block_type == "pixel_unshuffle",
                    shortcut=True,
                )
                down_block_list.append(downsample_block)

            down_blocks.append(nn.Sequential(*down_block_list))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.conv_out = ops.Conv2d(block_out_channels[-1], latent_channels, 3, 1, 1)

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = block_out_channels[-1] // latent_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: str or tuple = "ResBlock",
        block_out_channels: tuple = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: tuple = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: tuple = ((), (), (), (5,), (5,), (5,)),
        norm_type: str or tuple = "rms_norm",
        act_fn: str or tuple = "silu",
        upsample_block_type: str = "pixel_shuffle",
        in_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks
        if isinstance(norm_type, str):
            norm_type = (norm_type,) * num_blocks
        if isinstance(act_fn, str):
            act_fn = (act_fn,) * num_blocks

        self.conv_in = ops.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)

        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        up_blocks = []
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            up_block_list = []

            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=upsample_block_type == "interpolate",
                    shortcut=True,
                )
                up_block_list.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type[i],
                    act_fn=act_fn[i],
                    qkv_mutliscales=qkv_multiscales[i],
                )
                up_block_list.append(block)

            up_blocks.insert(0, nn.Sequential(*up_block_list))

        self.up_blocks = nn.ModuleList(up_blocks)

        channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]

        self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = ops.Conv2d(channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = DCUpBlock2d(
                channels, in_channels, interpolate=upsample_block_type == "interpolate", shortcut=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(
                self.in_shortcut_repeats, dim=1, output_size=hidden_states.shape[1] * self.in_shortcut_repeats
            )
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderDC(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 8,
        attention_head_dim: int = 32,
        encoder_block_types: Union[str, Tuple[str]] = ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
        decoder_block_types: Union[str, Tuple[str]] = ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        encoder_layers_per_block: Tuple[int] = (2, 2, 3, 3),
        decoder_layers_per_block: Tuple[int] = (3, 3, 3, 3),
        encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,), (5,)),
        decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,), (5,)),
        upsample_block_type: str = "interpolate",
        downsample_block_type: str = "Conv",
        decoder_norm_types: Union[str, Tuple[str]] = "rms_norm",
        decoder_act_fns: Union[str, Tuple[str]] = "silu",
        scaling_factor: float = 0.41407,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
        )

        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
        )

        self.scaling_factor = scaling_factor
        self.spatial_compression_ratio = 2 ** (len(encoder_block_out_channels) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encoding function."""
        encoded = self.encoder(x)
        return encoded * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Scale the latents back
        z = z / self.scaling_factor
        decoded = self.decoder(z)
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


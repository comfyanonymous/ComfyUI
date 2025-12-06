from contextlib import nullcontext
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.models.upsampling import Upsample2D
from einops import rearrange

from model import safe_pad_operation
from comfy.ldm.hunyuan3d.vae import DiagonalGaussianDistribution

class SpatialNorm(nn.Module):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    if isinstance(norm_layer, (nn.LayerNorm, nn.RMSNorm)):
        if x.ndim == 4:
            x = rearrange(x, "b c h w -> b h w c")
            x = norm_layer(x)
            x = rearrange(x, "b h w c -> b c h w")
            return x.to(input_dtype)
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> b t h w c")
            x = norm_layer(x)
            x = rearrange(x, "b t h w c -> b c t h w")
            return x.to(input_dtype)
    if isinstance(norm_layer, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x).to(input_dtype)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, nn.GroupNorm) and memory_occupy > float("inf"): # TODO: this may be set dynamically from the vae
                num_chunks = min(4 if x.element_size() == 2 else 2, norm_layer.num_groups)
                assert norm_layer.num_groups % num_chunks == 0
                num_groups_per_chunk = norm_layer.num_groups // num_chunks

                x = list(x.chunk(num_chunks, dim=1))
                weights = norm_layer.weight.chunk(num_chunks, dim=0)
                biases = norm_layer.bias.chunk(num_chunks, dim=0)
                for i, (w, b) in enumerate(zip(weights, biases)):
                    x[i] = F.group_norm(x[i], num_groups_per_chunk, w, b, norm_layer.eps)
                    x[i] = x[i].to(input_dtype)
                x = torch.cat(x, dim=1)
            else:
                x = norm_layer(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x.to(input_dtype)
    raise NotImplementedError

def safe_interpolate_operation(x, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    """Safe interpolate operation that handles Half precision for problematic modes"""
    # Modes qui peuvent causer des problèmes avec Half precision
    problematic_modes = ['bilinear', 'bicubic', 'trilinear']
    
    if mode in problematic_modes:
        try:
            return F.interpolate(
                x, 
                size=size, 
                scale_factor=scale_factor, 
                mode=mode, 
                align_corners=align_corners,
                recompute_scale_factor=recompute_scale_factor
            )
        except RuntimeError as e:
            if ("not implemented for 'Half'" in str(e) or 
                "compute_indices_weights" in str(e)):
                original_dtype = x.dtype
                return F.interpolate(
                    x.float(), 
                    size=size, 
                    scale_factor=scale_factor, 
                    mode=mode, 
                    align_corners=align_corners,
                    recompute_scale_factor=recompute_scale_factor
                ).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'nearest' et autres modes compatibles, pas de fix nécessaire
        return F.interpolate(
            x, 
            size=size, 
            scale_factor=scale_factor, 
            mode=mode, 
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

_receptive_field_t = Literal["half", "full"]

class InflatedCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode,
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.padding = (0, *self.padding[1:])
        self.memory_limit = float("inf")

    def forward(
        self,
        input,
    ):
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # wirdly inflation_mode is pad, which would cause an assert error
        #if self.inflation_mode != "none":
        #    state_dict = modify_state_dict(
        #        self,
        #        state_dict,
        #        prefix,
        #        inflate_weight_fn=inflate_weight,
        #        inflate_bias_fn=inflate_bias,
        #    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class Upsample3D(nn.Module):

    def __init__(
        self,
        channels,
        out_channels = None,
        inflation_mode = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
        interpolate = True,
        name: str = "conv",
        use_conv_transpose = False,
        use_conv: bool = False,
        padding = 1,
        bias = True,
        kernel_size = None,
        **kwargs,
    ):
        super().__init__()
        self.interpolate = interpolate 
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv_transpose = use_conv_transpose
        self.use_conv = use_conv
        self.name = name

        self.conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        
        conv = self.conv if self.name == "conv" else self.Conv2d_0

        assert type(conv) is not nn.ConvTranspose2d
        # Note: lora_layer is not passed into constructor in the original implementation.
        # So we make a simplification.
        conv = InflatedCausalConv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        assert not self.interpolate
        # [Override] MAGViT v2 implementation
        if not self.interpolate:
            upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
            self.upscale_conv = nn.Conv3d(
                self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
            )
            identity = (
                torch.eye(self.channels)
                .repeat(upscale_ratio, 1)
                .reshape_as(self.upscale_conv.weight)
            )
            self.upscale_conv.weight.data.copy_(identity)

        if self.name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

        self.norm = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split([split_size, hidden_states.size(2) - split_size], dim=2)
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            hidden_states[i] = self.upscale_conv(hidden_states[i])
            hidden_states[i] = rearrange(
                hidden_states[i],
                "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                x=self.spatial_ratio,
                y=self.spatial_ratio,
                z=self.temporal_ratio,
            )

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        if not self.slicing:
            return hidden_states
        else:
            return torch.cat(hidden_states, dim=2)


class Downsample3D(nn.Module):
    """A 3D downsampling layer with an optional convolution."""

    def __init__(
        self,
        channels,
        out_channels = None,
        inflation_mode = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
        name: str = "conv",
        padding = 1,
        **kwargs,
    ):
        super().__init__()
        self.padding = padding
        self.name = name
        self.channels = channels
        self.out_channels = out_channels or channels
        conv = self.conv
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if type(conv) in [nn.Conv2d]:
            # Note: lora_layer is not passed into constructor in the original implementation.
            # So we make a simplification.
            conv = InflatedCausalConv3d(
                self.channels,
                self.out_channels,
                kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                padding=(
                    1 if self.temporal_down else 0,
                    self.padding if self.spatial_down else 0,
                    self.padding if self.spatial_down else 0,
                ),
                inflation_mode=inflation_mode,
            )
        elif type(conv) is nn.AvgPool2d:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv and self.padding == 0 and self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = safe_pad_operation(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        skip_time_act: bool = False,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.up = up
        self.down = down
        self.use_in_shortcut = use_in_shortcut
        self.output_scale_factor = output_scale_factor
        self.skip_time_act = skip_time_act
        self.nonlinearity = nn.SiLU()
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        self.conv1 = InflatedCausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = InflatedCausalConv3d(
            self.out_channels,
            self.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        if self.up:
            self.upsample = Upsample3D(
                self.in_channels,
                use_conv=False,
                inflation_mode=inflation_mode,
                slicing=slicing,
            )
        elif self.down:
            self.downsample = Downsample3D(
                self.in_channels,
                use_conv=False,
                padding=1,
                name="op",
                inflation_mode=inflation_mode,
            )

        if self.use_in_shortcut:
            self.conv_shortcut = InflatedCausalConv3d(
                self.in_channels,
                self.conv_shortcut.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )

    def forward(
        self, input_tensor, temb, **kwargs
    ):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None:
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DownEncoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__()
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # [Override] Replace module.
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        **kwargs,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__()
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                    slicing=slicing,
                )
            )

            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_upsample:
            # [Override] Replace module & use learnable upsample
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        interpolate=False,
                        inflation_mode=inflation_mode,
                        slicing=slicing,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            # [Override] Replace module.
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            print(
                f"It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups if resnet_time_scale_shift == "default" else None
                        ),
                        spatial_norm_dim=(
                            temb_channels if resnet_time_scale_shift == "spatial" else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        # [Override] add extra_cond_dim, temporal down num
        temporal_down_num: int = 2,
        extra_cond_dim: int = None,
        gradient_checkpoint: bool = False,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.extra_cond_dim = extra_cond_dim

        self.conv_extra_cond = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # [Override] to support temporal down block design
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last ones

            assert down_block_type == "DownEncoderBlock3D"

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                # Note: Don't know why set it as 0
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

            def zero_module(module):
                # Zero out the parameters of a module and return it.
                for p in module.parameters():
                    p.detach().zero_()
                return module

            self.conv_extra_cond.append(
                zero_module(
                    nn.Conv3d(extra_cond_dim, output_channel, kernel_size=1, stride=1, padding=0)
                )
                if self.extra_cond_dim is not None and self.extra_cond_dim > 0
                else None
            )

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    def forward(
        self,
        sample: torch.FloatTensor,
        extra_cond=None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), sample, use_reentrant=False
                )
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample)

        else:
            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = down_block(sample)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder3D(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        # [Override] add temporal up block
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = InflatedCausalConv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        print(f"slicing_up_num: {slicing_up_num}")
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            assert up_block_type == "UpDecoderBlock3D"
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedCausalConv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    # Note: Just copy from Decoder.
    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

class VideoAutoencoderKL(nn.Module):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode = "tail",
        time_receptive_field: _receptive_field_t = "full",
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        *args,
        **kwargs,
    ):
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            InflatedCausalConv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            InflatedCausalConv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        h = self.slicing_encode(x)
        posterior = DiagonalGaussianDistribution(h).sample()

        if not return_dict:
            return (posterior,)

        return posterior

    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ):
        decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return decoded

    def _encode(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        _x = x.to(self.device)
        h = self.encoder(_x,)
        if self.quant_conv is not None:
            output = self.quant_conv(h)
        else:
            output = h
        return output.to(x.device)

    def _decode(
        self, z: torch.Tensor
    ) -> torch.Tensor:
        _z = z.to(self.device)
        if self.post_quant_conv is not None:
            _z = self.post_quant_conv(_z)
        output = self.decoder(_z)
        return output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode(z)

    def tiled_encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def tiled_decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self,
        *args,
        spatial_downsample_factor = 8,
        temporal_downsample_factor = 4,
        freeze_encoder = True,
        **kwargs,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.FloatTensor):
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z).sample
        return x, z, p

    def encode(self, x: torch.FloatTensor):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x).latent_dist
        z = p.sample().squeeze(2)
        return z, p

    def decode(self, z: torch.FloatTensor):
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z).sample.squeeze(2)
        return x

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        # TODO
        #set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
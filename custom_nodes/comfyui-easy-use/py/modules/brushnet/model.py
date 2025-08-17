from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...libs.utils import install_package
try:
    install_package("diffusers", "0.27.2", True, "0.25.0")

    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.utils import BaseOutput, logging
    from diffusers.models.attention_processor import (
        ADDED_KV_ATTENTION_PROCESSORS,
        CROSS_ATTENTION_PROCESSORS,
        AttentionProcessor,
        AttnAddedKVProcessor,
        AttnProcessor,
    )
    from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

    from diffusers.models.transformers.dual_transformer_2d import DualTransformer2DModel
    from diffusers.models.transformers.transformer_2d import Transformer2DModel

    from .unet_2d_blocks import (
        CrossAttnDownBlock2D,
        DownBlock2D,
        get_down_block,
        get_mid_block,
        get_up_block,
    )

    from .unet_2d_condition import UNet2DConditionModel

    logger = logging.get_logger(__name__)

    def zero_module(module):
        for p in module.parameters():
            nn.init.zeros_(p)
        return module

    @dataclass
    class BrushNetOutput(BaseOutput):

        up_block_res_samples: Tuple[torch.Tensor]
        down_block_res_samples: Tuple[torch.Tensor]
        mid_block_res_sample: torch.Tensor

    # BrushNetModel
    class BrushNetModel(ModelMixin, ConfigMixin):
        """A BrushNet model."""
        _supports_gradient_checkpointing = True

        @register_to_config
        def __init__(
                self,
                in_channels: int = 4,
                conditioning_channels: int = 5,
                flip_sin_to_cos: bool = True,
                freq_shift: int = 0,
                down_block_types: Tuple[str, ...] = (
                        "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                        "DownBlock2D",
                ),
                mid_block_type: Optional[str] = "UNetMidBlock2D",
                up_block_types: Tuple[str, ...] = (
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                        "UpBlock2D",
                ),
                only_cross_attention: Union[bool, Tuple[bool]] = False,
                block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                layers_per_block: int = 2,
                downsample_padding: int = 1,
                mid_block_scale_factor: float = 1,
                act_fn: str = "silu",
                norm_num_groups: Optional[int] = 32,
                norm_eps: float = 1e-5,
                cross_attention_dim: int = 1280,
                transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
                encoder_hid_dim: Optional[int] = None,
                encoder_hid_dim_type: Optional[str] = None,
                attention_head_dim: Union[int, Tuple[int, ...]] = 8,
                num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
                use_linear_projection: bool = False,
                class_embed_type: Optional[str] = None,
                addition_embed_type: Optional[str] = None,
                addition_time_embed_dim: Optional[int] = None,
                num_class_embeds: Optional[int] = None,
                upcast_attention: bool = False,
                resnet_time_scale_shift: str = "default",
                projection_class_embeddings_input_dim: Optional[int] = None,
                brushnet_conditioning_channel_order: str = "rgb",
                conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
                global_pool_conditions: bool = False,
                addition_embed_type_num_heads: int = 64,
        ):
            super().__init__()

            # If `num_attention_heads` is not defined (which is the case for most models)
            # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
            # The reason for this behavior is to correct for incorrectly named variables that were introduced
            # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
            # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
            # which is why we correct for the naming here.
            num_attention_heads = num_attention_heads or attention_head_dim

            # Check inputs
            if len(down_block_types) != len(up_block_types):
                raise ValueError(
                    f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
                )

            if len(block_out_channels) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
                )

            if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
                )

            if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
                )

            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

            # input
            conv_in_kernel = 3
            conv_in_padding = (conv_in_kernel - 1) // 2
            self.conv_in_condition = nn.Conv2d(
                in_channels + conditioning_channels, block_out_channels[0], kernel_size=conv_in_kernel,
                padding=conv_in_padding
            )

            # time
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
            self.time_embedding = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim,
                act_fn=act_fn,
            )

            if encoder_hid_dim_type is None and encoder_hid_dim is not None:
                encoder_hid_dim_type = "text_proj"
                self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
                print("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

            if encoder_hid_dim is None and encoder_hid_dim_type is not None:
                raise ValueError(
                    f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
                )

            if encoder_hid_dim_type == "text_proj":
                self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
            elif encoder_hid_dim_type == "text_image_proj":
                # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
                # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
                # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
                self.encoder_hid_proj = TextImageProjection(
                    text_embed_dim=encoder_hid_dim,
                    image_embed_dim=cross_attention_dim,
                    cross_attention_dim=cross_attention_dim,
                )

            elif encoder_hid_dim_type is not None:
                raise ValueError(
                    f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
                )
            else:
                self.encoder_hid_proj = None

            # class embedding
            if class_embed_type is None and num_class_embeds is not None:
                self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
            elif class_embed_type == "timestep":
                self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            elif class_embed_type == "identity":
                self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
            elif class_embed_type == "projection":
                if projection_class_embeddings_input_dim is None:
                    raise ValueError(
                        "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                    )
                # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
                # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
                # 2. it projects from an arbitrary input dimension.
                #
                # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
                # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
                # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
                self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
            else:
                self.class_embedding = None

            if addition_embed_type == "text":
                if encoder_hid_dim is not None:
                    text_time_embedding_from_dim = encoder_hid_dim
                else:
                    text_time_embedding_from_dim = cross_attention_dim

                self.add_embedding = TextTimeEmbedding(
                    text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
                )
            elif addition_embed_type == "text_image":
                # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
                # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
                # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
                self.add_embedding = TextImageTimeEmbedding(
                    text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim,
                    time_embed_dim=time_embed_dim
                )
            elif addition_embed_type == "text_time":
                self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
                self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

            elif addition_embed_type is not None:
                raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

            self.down_blocks = nn.ModuleList([])
            self.brushnet_down_blocks = nn.ModuleList([])

            if isinstance(only_cross_attention, bool):
                only_cross_attention = [only_cross_attention] * len(down_block_types)

            if isinstance(attention_head_dim, int):
                attention_head_dim = (attention_head_dim,) * len(down_block_types)

            if isinstance(num_attention_heads, int):
                num_attention_heads = (num_attention_heads,) * len(down_block_types)

            # down
            output_channel = block_out_channels[0]

            brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
            brushnet_block = zero_module(brushnet_block)
            self.brushnet_down_blocks.append(brushnet_block)

            for i, down_block_type in enumerate(down_block_types):
                input_channel = output_channel
                output_channel = block_out_channels[i]
                is_final_block = i == len(block_out_channels) - 1

                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i],
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
                self.down_blocks.append(down_block)

                for _ in range(layers_per_block):
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_down_blocks.append(brushnet_block)

                if not is_final_block:
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_down_blocks.append(brushnet_block)

            # mid
            mid_block_channel = block_out_channels[-1]

            brushnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
            brushnet_block = zero_module(brushnet_block)
            self.brushnet_mid_block = brushnet_block

            self.mid_block = get_mid_block(
                mid_block_type,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )

            # count how many layers upsample the images
            self.num_upsamplers = 0

            # up
            reversed_block_out_channels = list(reversed(block_out_channels))
            reversed_num_attention_heads = list(reversed(num_attention_heads))
            reversed_transformer_layers_per_block = (list(reversed(transformer_layers_per_block)))
            only_cross_attention = list(reversed(only_cross_attention))

            output_channel = reversed_block_out_channels[0]

            self.up_blocks = nn.ModuleList([])
            self.brushnet_up_blocks = nn.ModuleList([])

            for i, up_block_type in enumerate(up_block_types):
                is_final_block = i == len(block_out_channels) - 1

                prev_output_channel = output_channel
                output_channel = reversed_block_out_channels[i]
                input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

                # add upsample block for all BUT final layer
                if not is_final_block:
                    add_upsample = True
                    self.num_upsamplers += 1
                else:
                    add_upsample = False

                up_block = get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resolution_idx=i,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=reversed_num_attention_heads[i],
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                )
                self.up_blocks.append(up_block)
                prev_output_channel = output_channel

                for _ in range(layers_per_block + 1):
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_up_blocks.append(brushnet_block)

                if not is_final_block:
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_up_blocks.append(brushnet_block)

        @classmethod
        def from_unet(
                cls,
                unet: UNet2DConditionModel,
                brushnet_conditioning_channel_order: str = "rgb",
                conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
                load_weights_from_unet: bool = True,
                conditioning_channels: int = 5,
        ):
            r"""
            Instantiate a [`BrushNetModel`] from [`UNet2DConditionModel`].

            Parameters:
                unet (`UNet2DConditionModel`):
                    The UNet model weights to copy to the [`BrushNetModel`]. All configuration options are also copied
                    where applicable.
            """
            transformer_layers_per_block = (
                unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
            )
            encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
            encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
            addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
            addition_time_embed_dim = (
                unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
            )

            down_block_types = ["DownBlock2D" for block_name in unet.config.down_block_types]
            mid_block_type = "MidBlock2D"
            up_block_types = ["UpBlock2D" for block_name in unet.config.down_block_types]

            brushnet = cls(
                in_channels=unet.config.in_channels,
                conditioning_channels=conditioning_channels,
                flip_sin_to_cos=unet.config.flip_sin_to_cos,
                freq_shift=unet.config.freq_shift,
                down_block_types=down_block_types,
                mid_block_type=mid_block_type,
                up_block_types=up_block_types,
                only_cross_attention=unet.config.only_cross_attention,
                block_out_channels=unet.config.block_out_channels,
                layers_per_block=unet.config.layers_per_block,
                downsample_padding=unet.config.downsample_padding,
                mid_block_scale_factor=unet.config.mid_block_scale_factor,
                act_fn=unet.config.act_fn,
                norm_num_groups=unet.config.norm_num_groups,
                norm_eps=unet.config.norm_eps,
                cross_attention_dim=unet.config.cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block,
                encoder_hid_dim=encoder_hid_dim,
                encoder_hid_dim_type=encoder_hid_dim_type,
                attention_head_dim=unet.config.attention_head_dim,
                num_attention_heads=unet.config.num_attention_heads,
                use_linear_projection=unet.config.use_linear_projection,
                class_embed_type=unet.config.class_embed_type,
                addition_embed_type=addition_embed_type,
                addition_time_embed_dim=addition_time_embed_dim,
                num_class_embeds=unet.config.num_class_embeds,
                upcast_attention=unet.config.upcast_attention,
                resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
                projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
                brushnet_conditioning_channel_order=brushnet_conditioning_channel_order,
                conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            )

            if load_weights_from_unet:
                conv_in_condition_weight = torch.zeros_like(brushnet.conv_in_condition.weight)
                conv_in_condition_weight[:, :4, ...] = unet.conv_in.weight
                conv_in_condition_weight[:, 4:8, ...] = unet.conv_in.weight
                brushnet.conv_in_condition.weight = torch.nn.Parameter(conv_in_condition_weight)
                brushnet.conv_in_condition.bias = unet.conv_in.bias

                brushnet.time_proj.load_state_dict(unet.time_proj.state_dict())
                brushnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

                if brushnet.class_embedding:
                    brushnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

                brushnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
                brushnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)
                brushnet.up_blocks.load_state_dict(unet.up_blocks.state_dict(), strict=False)

            return brushnet

        @property
        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            Returns:
                `dict` of attention processors: A dictionary containing all attention processors used in the model with
                indexed by its weight name.
            """
            # set recursively
            processors = {}

            def fn_recursive_add_processors(name: str, module: torch.nn.Module,
                                            processors: Dict[str, AttentionProcessor]):
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

                return processors

            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)

            return processors

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            Sets the attention processor to use to compute attention.

            Parameters:
                processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                    The instantiated processor class or a dictionary of processor classes that will be set as the processor
                    for **all** `Attention` layers.

                    If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                    processor. This is strongly recommended when setting trainable attention processors.

            """
            count = len(self.attn_processors.keys())

            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                    f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
                )

            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                if hasattr(module, "set_processor"):
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        module.set_processor(processor.pop(f"{name}.processor"))

                for sub_name, child in module.named_children():
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

            for name, module in self.named_children():
                fn_recursive_attn_processor(name, module, processor)

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
        def set_default_attn_processor(self):
            """
            Disables custom attention processors and sets the default attention implementation.
            """
            if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnAddedKVProcessor()
            elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnProcessor()
            else:
                raise ValueError(
                    f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
                )

            self.set_attn_processor(processor)

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
        def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
            r"""
            Enable sliced attention computation.

            When this option is enabled, the attention module splits the input tensor in slices to compute attention in
            several steps. This is useful for saving some memory in exchange for a small decrease in speed.

            Args:
                slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                    When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                    `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                    provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                    must be a multiple of `slice_size`.
            """
            sliceable_head_dims = []

            def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
                if hasattr(module, "set_attention_slice"):
                    sliceable_head_dims.append(module.sliceable_head_dim)

                for child in module.children():
                    fn_recursive_retrieve_sliceable_dims(child)

            # retrieve number of attention layers
            for module in self.children():
                fn_recursive_retrieve_sliceable_dims(module)

            num_sliceable_layers = len(sliceable_head_dims)

            if slice_size == "auto":
                # half the attention head size is usually a good trade-off between
                # speed and memory
                slice_size = [dim // 2 for dim in sliceable_head_dims]
            elif slice_size == "max":
                # make smallest slice possible
                slice_size = num_sliceable_layers * [1]

            slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

            if len(slice_size) != len(sliceable_head_dims):
                raise ValueError(
                    f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                    f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
                )

            for i in range(len(slice_size)):
                size = slice_size[i]
                dim = sliceable_head_dims[i]
                if size is not None and size > dim:
                    raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

            # Recursively walk through all the children.
            # Any children which exposes the set_attention_slice method
            # gets the message
            def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
                if hasattr(module, "set_attention_slice"):
                    module.set_attention_slice(slice_size.pop())

                for child in module.children():
                    fn_recursive_set_attention_slice(child, slice_size)

            reversed_slice_size = list(reversed(slice_size))
            for module in self.children():
                fn_recursive_set_attention_slice(module, reversed_slice_size)

        def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                module.gradient_checkpointing = value

        def forward(
                self,
                sample: torch.FloatTensor,
                encoder_hidden_states: torch.Tensor,
                brushnet_cond: torch.FloatTensor,
                timestep=None,
                time_emb=None,
                conditioning_scale: float = 1.0,
                class_labels: Optional[torch.Tensor] = None,
                timestep_cond: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
                cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                guess_mode: bool = False,
                return_dict: bool = True,
                debug=False,
        ) -> Union[BrushNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:

            # check channel order
            channel_order = self.config.brushnet_conditioning_channel_order

            if channel_order == "rgb":
                # in rgb order by default
                ...
            elif channel_order == "bgr":
                brushnet_cond = torch.flip(brushnet_cond, dims=[1])
            else:
                raise ValueError(f"unknown `brushnet_conditioning_channel_order`: {channel_order}")

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            if timestep is None and time_emb is None:
                raise ValueError(f"`timestep` and `emb` are both None")

            # print("BN: sample.device", sample.device)
            # print("BN: TE.device", self.time_embedding.linear_1.weight.device)

            if timestep is not None:
                # 1. time
                timesteps = timestep
                if not torch.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = sample.device.type == "mps"
                    if isinstance(timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None].to(sample.device)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(sample.shape[0])

                t_emb = self.time_proj(timesteps)

                # timesteps does not contain any weights and will always return f32 tensors
                # but time_embedding might actually be running in fp16. so we need to cast here.
                # there might be better ways to encapsulate this.
                t_emb = t_emb.to(dtype=sample.dtype)

                # print("t_emb.device =",t_emb.device)

                emb = self.time_embedding(t_emb, timestep_cond)
                aug_emb = None

                # print('emb.shape', emb.shape)

                if self.class_embedding is not None:
                    if class_labels is None:
                        raise ValueError("class_labels should be provided when num_class_embeds > 0")

                    if self.config.class_embed_type == "timestep":
                        class_labels = self.time_proj(class_labels)

                    class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                    emb = emb + class_emb

                if self.config.addition_embed_type is not None:
                    if self.config.addition_embed_type == "text":
                        aug_emb = self.add_embedding(encoder_hidden_states)

                    elif self.config.addition_embed_type == "text_time":
                        if "text_embeds" not in added_cond_kwargs:
                            raise ValueError(
                                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                            )
                        text_embeds = added_cond_kwargs.get("text_embeds")
                        if "time_ids" not in added_cond_kwargs:
                            raise ValueError(
                                f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                            )
                        time_ids = added_cond_kwargs.get("time_ids")
                        time_embeds = self.add_time_proj(time_ids.flatten())
                        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                        add_embeds = add_embeds.to(emb.dtype)
                        aug_emb = self.add_embedding(add_embeds)

                        # print('text_embeds', text_embeds.shape, 'time_ids', time_ids.shape, 'time_embeds', time_embeds.shape, 'add__embeds', add_embeds.shape, 'aug_emb', aug_emb.shape)

                emb = emb + aug_emb if aug_emb is not None else emb
            else:
                emb = time_emb

            # 2. pre-process

            brushnet_cond = torch.concat([sample, brushnet_cond], 1)
            sample = self.conv_in_condition(brushnet_cond)

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                down_block_res_samples += res_samples

            # 4. PaintingNet down blocks
            brushnet_down_block_res_samples = ()
            for down_block_res_sample, brushnet_down_block in zip(down_block_res_samples, self.brushnet_down_blocks):
                down_block_res_sample = brushnet_down_block(down_block_res_sample)
                brushnet_down_block_res_samples = brushnet_down_block_res_samples + (down_block_res_sample,)

            # 5. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample = self.mid_block(sample, emb)

            # 6. BrushNet mid blocks
            brushnet_mid_block_res_sample = self.brushnet_mid_block(sample)

            # 7. up
            up_block_res_samples = ()
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets):]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample, up_res_samples = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        return_res_samples=True
                    )
                else:
                    sample, up_res_samples = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        return_res_samples=True
                    )

                up_block_res_samples += up_res_samples

            # 8. BrushNet up blocks
            brushnet_up_block_res_samples = ()
            for up_block_res_sample, brushnet_up_block in zip(up_block_res_samples, self.brushnet_up_blocks):
                up_block_res_sample = brushnet_up_block(up_block_res_sample)
                brushnet_up_block_res_samples = brushnet_up_block_res_samples + (up_block_res_sample,)

            # 6. scaling
            if guess_mode and not self.config.global_pool_conditions:
                scales = torch.logspace(-1, 0,
                                        len(brushnet_down_block_res_samples) + 1 + len(brushnet_up_block_res_samples),
                                        device=sample.device)  # 0.1 to 1.0
                scales = scales * conditioning_scale

                brushnet_down_block_res_samples = [sample * scale for sample, scale in
                                                   zip(brushnet_down_block_res_samples,
                                                       scales[:len(brushnet_down_block_res_samples)])]
                brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * scales[
                    len(brushnet_down_block_res_samples)]
                brushnet_up_block_res_samples = [sample * scale for sample, scale in zip(brushnet_up_block_res_samples,
                                                                                         scales[
                                                                                         len(brushnet_down_block_res_samples) + 1:])]
            else:
                brushnet_down_block_res_samples = [sample * conditioning_scale for sample in
                                                   brushnet_down_block_res_samples]
                brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * conditioning_scale
                brushnet_up_block_res_samples = [sample * conditioning_scale for sample in
                                                 brushnet_up_block_res_samples]

            if self.config.global_pool_conditions:
                brushnet_down_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in brushnet_down_block_res_samples
                ]
                brushnet_mid_block_res_sample = torch.mean(brushnet_mid_block_res_sample, dim=(2, 3), keepdim=True)
                brushnet_up_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in brushnet_up_block_res_samples
                ]

            if not return_dict:
                return (brushnet_down_block_res_samples, brushnet_mid_block_res_sample, brushnet_up_block_res_samples)

            return BrushNetOutput(
                down_block_res_samples=brushnet_down_block_res_samples,
                mid_block_res_sample=brushnet_mid_block_res_sample,
                up_block_res_samples=brushnet_up_block_res_samples
            )

    # PowerPaintModel
    class PowerPaintModel(ModelMixin, ConfigMixin):
        _supports_gradient_checkpointing = True

        @register_to_config
        def __init__(
            self,
            in_channels: int = 4,
            conditioning_channels: int = 5,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str, ...] = (
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
            up_block_types: Tuple[str, ...] = (
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            only_cross_attention: Union[bool, Tuple[bool]] = False,
            block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: Optional[int] = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
            encoder_hid_dim: Optional[int] = None,
            encoder_hid_dim_type: Optional[str] = None,
            attention_head_dim: Union[int, Tuple[int, ...]] = 8,
            num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
            use_linear_projection: bool = False,
            class_embed_type: Optional[str] = None,
            addition_embed_type: Optional[str] = None,
            addition_time_embed_dim: Optional[int] = None,
            num_class_embeds: Optional[int] = None,
            upcast_attention: bool = False,
            resnet_time_scale_shift: str = "default",
            projection_class_embeddings_input_dim: Optional[int] = None,
            brushnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
            global_pool_conditions: bool = False,
            addition_embed_type_num_heads: int = 64,
        ):
            super().__init__()

            # If `num_attention_heads` is not defined (which is the case for most models)
            # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
            # The reason for this behavior is to correct for incorrectly named variables that were introduced
            # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
            # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
            # which is why we correct for the naming here.
            num_attention_heads = num_attention_heads or attention_head_dim

            # Check inputs
            if len(down_block_types) != len(up_block_types):
                raise ValueError(
                    f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
                )

            if len(block_out_channels) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
                )

            if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
                )

            if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
                raise ValueError(
                    f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
                )

            if isinstance(transformer_layers_per_block, int):
                transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

            # input
            conv_in_kernel = 3
            conv_in_padding = (conv_in_kernel - 1) // 2
            self.conv_in_condition = nn.Conv2d(
                in_channels + conditioning_channels,
                block_out_channels[0],
                kernel_size=conv_in_kernel,
                padding=conv_in_padding,
            )

            # time
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
            self.time_embedding = TimestepEmbedding(
                timestep_input_dim,
                time_embed_dim,
                act_fn=act_fn,
            )

            if encoder_hid_dim_type is None and encoder_hid_dim is not None:
                encoder_hid_dim_type = "text_proj"
                self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
                logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

            if encoder_hid_dim is None and encoder_hid_dim_type is not None:
                raise ValueError(
                    f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
                )

            if encoder_hid_dim_type == "text_proj":
                self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
            elif encoder_hid_dim_type == "text_image_proj":
                # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
                # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
                # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
                self.encoder_hid_proj = TextImageProjection(
                    text_embed_dim=encoder_hid_dim,
                    image_embed_dim=cross_attention_dim,
                    cross_attention_dim=cross_attention_dim,
                )

            elif encoder_hid_dim_type is not None:
                raise ValueError(
                    f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
                )
            else:
                self.encoder_hid_proj = None

            # class embedding
            if class_embed_type is None and num_class_embeds is not None:
                self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
            elif class_embed_type == "timestep":
                self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            elif class_embed_type == "identity":
                self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
            elif class_embed_type == "projection":
                if projection_class_embeddings_input_dim is None:
                    raise ValueError(
                        "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                    )
                # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
                # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
                # 2. it projects from an arbitrary input dimension.
                #
                # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
                # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
                # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
                self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
            else:
                self.class_embedding = None

            if addition_embed_type == "text":
                if encoder_hid_dim is not None:
                    text_time_embedding_from_dim = encoder_hid_dim
                else:
                    text_time_embedding_from_dim = cross_attention_dim

                self.add_embedding = TextTimeEmbedding(
                    text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
                )
            elif addition_embed_type == "text_image":
                # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
                # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
                # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
                self.add_embedding = TextImageTimeEmbedding(
                    text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
                )
            elif addition_embed_type == "text_time":
                self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
                self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

            elif addition_embed_type is not None:
                raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

            self.down_blocks = nn.ModuleList([])
            self.brushnet_down_blocks = nn.ModuleList([])

            if isinstance(only_cross_attention, bool):
                only_cross_attention = [only_cross_attention] * len(down_block_types)

            if isinstance(attention_head_dim, int):
                attention_head_dim = (attention_head_dim,) * len(down_block_types)

            if isinstance(num_attention_heads, int):
                num_attention_heads = (num_attention_heads,) * len(down_block_types)

            # down
            output_channel = block_out_channels[0]

            brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
            brushnet_block = zero_module(brushnet_block)
            self.brushnet_down_blocks.append(brushnet_block)

            for i, down_block_type in enumerate(down_block_types):
                input_channel = output_channel
                output_channel = block_out_channels[i]
                is_final_block = i == len(block_out_channels) - 1

                down_block = get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    transformer_layers_per_block=transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=time_embed_dim,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=num_attention_heads[i],
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                    downsample_padding=downsample_padding,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                )
                self.down_blocks.append(down_block)

                for _ in range(layers_per_block):
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_down_blocks.append(brushnet_block)

                if not is_final_block:
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_down_blocks.append(brushnet_block)

            # mid
            mid_block_channel = block_out_channels[-1]

            brushnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
            brushnet_block = zero_module(brushnet_block)
            self.brushnet_mid_block = brushnet_block

            self.mid_block = get_mid_block(
                mid_block_type,
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=mid_block_channel,
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )

            # count how many layers upsample the images
            self.num_upsamplers = 0

            # up
            reversed_block_out_channels = list(reversed(block_out_channels))
            reversed_num_attention_heads = list(reversed(num_attention_heads))
            reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))
            only_cross_attention = list(reversed(only_cross_attention))

            output_channel = reversed_block_out_channels[0]

            self.up_blocks = nn.ModuleList([])
            self.brushnet_up_blocks = nn.ModuleList([])

            for i, up_block_type in enumerate(up_block_types):
                is_final_block = i == len(block_out_channels) - 1

                prev_output_channel = output_channel
                output_channel = reversed_block_out_channels[i]
                input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

                # add upsample block for all BUT final layer
                if not is_final_block:
                    add_upsample = True
                    self.num_upsamplers += 1
                else:
                    add_upsample = False

                up_block = get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=time_embed_dim,
                    add_upsample=add_upsample,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resolution_idx=i,
                    resnet_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    num_attention_heads=reversed_num_attention_heads[i],
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention[i],
                    upcast_attention=upcast_attention,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                )
                self.up_blocks.append(up_block)
                prev_output_channel = output_channel

                for _ in range(layers_per_block + 1):
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_up_blocks.append(brushnet_block)

                if not is_final_block:
                    brushnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                    brushnet_block = zero_module(brushnet_block)
                    self.brushnet_up_blocks.append(brushnet_block)

        @classmethod
        def from_unet(
            cls,
            unet: UNet2DConditionModel,
            brushnet_conditioning_channel_order: str = "rgb",
            conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
            load_weights_from_unet: bool = True,
            conditioning_channels: int = 5,
        ):
            r"""
            Instantiate a [`BrushNetModel`] from [`UNet2DConditionModel`].

            Parameters:
                unet (`UNet2DConditionModel`):
                    The UNet model weights to copy to the [`BrushNetModel`]. All configuration options are also copied
                    where applicable.
            """
            transformer_layers_per_block = (
                unet.config.transformer_layers_per_block if "transformer_layers_per_block" in unet.config else 1
            )
            encoder_hid_dim = unet.config.encoder_hid_dim if "encoder_hid_dim" in unet.config else None
            encoder_hid_dim_type = unet.config.encoder_hid_dim_type if "encoder_hid_dim_type" in unet.config else None
            addition_embed_type = unet.config.addition_embed_type if "addition_embed_type" in unet.config else None
            addition_time_embed_dim = (
                unet.config.addition_time_embed_dim if "addition_time_embed_dim" in unet.config else None
            )

            brushnet = cls(
                in_channels=unet.config.in_channels,
                conditioning_channels=conditioning_channels,
                flip_sin_to_cos=unet.config.flip_sin_to_cos,
                freq_shift=unet.config.freq_shift,
                # down_block_types=['DownBlock2D','DownBlock2D','DownBlock2D','DownBlock2D'],
                down_block_types=[
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ],
                # mid_block_type='MidBlock2D',
                mid_block_type="UNetMidBlock2DCrossAttn",
                # up_block_types=['UpBlock2D','UpBlock2D','UpBlock2D','UpBlock2D'],
                up_block_types=["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
                only_cross_attention=unet.config.only_cross_attention,
                block_out_channels=unet.config.block_out_channels,
                layers_per_block=unet.config.layers_per_block,
                downsample_padding=unet.config.downsample_padding,
                mid_block_scale_factor=unet.config.mid_block_scale_factor,
                act_fn=unet.config.act_fn,
                norm_num_groups=unet.config.norm_num_groups,
                norm_eps=unet.config.norm_eps,
                cross_attention_dim=unet.config.cross_attention_dim,
                transformer_layers_per_block=transformer_layers_per_block,
                encoder_hid_dim=encoder_hid_dim,
                encoder_hid_dim_type=encoder_hid_dim_type,
                attention_head_dim=unet.config.attention_head_dim,
                num_attention_heads=unet.config.num_attention_heads,
                use_linear_projection=unet.config.use_linear_projection,
                class_embed_type=unet.config.class_embed_type,
                addition_embed_type=addition_embed_type,
                addition_time_embed_dim=addition_time_embed_dim,
                num_class_embeds=unet.config.num_class_embeds,
                upcast_attention=unet.config.upcast_attention,
                resnet_time_scale_shift=unet.config.resnet_time_scale_shift,
                projection_class_embeddings_input_dim=unet.config.projection_class_embeddings_input_dim,
                brushnet_conditioning_channel_order=brushnet_conditioning_channel_order,
                conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            )

            if load_weights_from_unet:
                conv_in_condition_weight = torch.zeros_like(brushnet.conv_in_condition.weight)
                conv_in_condition_weight[:, :4, ...] = unet.conv_in.weight
                conv_in_condition_weight[:, 4:8, ...] = unet.conv_in.weight
                brushnet.conv_in_condition.weight = torch.nn.Parameter(conv_in_condition_weight)
                brushnet.conv_in_condition.bias = unet.conv_in.bias

                brushnet.time_proj.load_state_dict(unet.time_proj.state_dict())
                brushnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

                if brushnet.class_embedding:
                    brushnet.class_embedding.load_state_dict(unet.class_embedding.state_dict())

                brushnet.down_blocks.load_state_dict(unet.down_blocks.state_dict(), strict=False)
                brushnet.mid_block.load_state_dict(unet.mid_block.state_dict(), strict=False)
                brushnet.up_blocks.load_state_dict(unet.up_blocks.state_dict(), strict=False)

            return brushnet.to(unet.dtype)

        @property
        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
        def attn_processors(self) -> Dict[str, AttentionProcessor]:
            r"""
            Returns:
                `dict` of attention processors: A dictionary containing all attention processors used in the model with
                indexed by its weight name.
            """
            # set recursively
            processors = {}

            def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
                if hasattr(module, "get_processor"):
                    processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

                for sub_name, child in module.named_children():
                    fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

                return processors

            for name, module in self.named_children():
                fn_recursive_add_processors(name, module, processors)

            return processors

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
        def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
            r"""
            Sets the attention processor to use to compute attention.

            Parameters:
                processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                    The instantiated processor class or a dictionary of processor classes that will be set as the processor
                    for **all** `Attention` layers.

                    If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                    processor. This is strongly recommended when setting trainable attention processors.

            """
            count = len(self.attn_processors.keys())

            if isinstance(processor, dict) and len(processor) != count:
                raise ValueError(
                    f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                    f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
                )

            def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
                if hasattr(module, "set_processor"):
                    if not isinstance(processor, dict):
                        module.set_processor(processor)
                    else:
                        module.set_processor(processor.pop(f"{name}.processor"))

                for sub_name, child in module.named_children():
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

            for name, module in self.named_children():
                fn_recursive_attn_processor(name, module, processor)

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
        def set_default_attn_processor(self):
            """
            Disables custom attention processors and sets the default attention implementation.
            """
            if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnAddedKVProcessor()
            elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
                processor = AttnProcessor()
            else:
                raise ValueError(
                    f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
                )

            self.set_attn_processor(processor)

        # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attention_slice
        def set_attention_slice(self, slice_size: Union[str, int, List[int]]) -> None:
            r"""
            Enable sliced attention computation.

            When this option is enabled, the attention module splits the input tensor in slices to compute attention in
            several steps. This is useful for saving some memory in exchange for a small decrease in speed.

            Args:
                slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                    When `"auto"`, input to the attention heads is halved, so attention is computed in two steps. If
                    `"max"`, maximum amount of memory is saved by running only one slice at a time. If a number is
                    provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                    must be a multiple of `slice_size`.
            """
            sliceable_head_dims = []

            def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
                if hasattr(module, "set_attention_slice"):
                    sliceable_head_dims.append(module.sliceable_head_dim)

                for child in module.children():
                    fn_recursive_retrieve_sliceable_dims(child)

            # retrieve number of attention layers
            for module in self.children():
                fn_recursive_retrieve_sliceable_dims(module)

            num_sliceable_layers = len(sliceable_head_dims)

            if slice_size == "auto":
                # half the attention head size is usually a good trade-off between
                # speed and memory
                slice_size = [dim // 2 for dim in sliceable_head_dims]
            elif slice_size == "max":
                # make smallest slice possible
                slice_size = num_sliceable_layers * [1]

            slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

            if len(slice_size) != len(sliceable_head_dims):
                raise ValueError(
                    f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                    f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
                )

            for i in range(len(slice_size)):
                size = slice_size[i]
                dim = sliceable_head_dims[i]
                if size is not None and size > dim:
                    raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

            # Recursively walk through all the children.
            # Any children which exposes the set_attention_slice method
            # gets the message
            def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
                if hasattr(module, "set_attention_slice"):
                    module.set_attention_slice(slice_size.pop())

                for child in module.children():
                    fn_recursive_set_attention_slice(child, slice_size)

            reversed_slice_size = list(reversed(slice_size))
            for module in self.children():
                fn_recursive_set_attention_slice(module, reversed_slice_size)

        def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
            if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D)):
                module.gradient_checkpointing = value

        def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            brushnet_cond: torch.FloatTensor,
            conditioning_scale: float = 1.0,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guess_mode: bool = False,
            return_dict: bool = True,
            debug=False,
        ) -> Union[BrushNetOutput, Tuple[Tuple[torch.FloatTensor, ...], torch.FloatTensor]]:
            """
            The [`BrushNetModel`] forward method.

            Args:
                sample (`torch.FloatTensor`):
                    The noisy input tensor.
                timestep (`Union[torch.Tensor, float, int]`):
                    The number of timesteps to denoise an input.
                encoder_hidden_states (`torch.Tensor`):
                    The encoder hidden states.
                brushnet_cond (`torch.FloatTensor`):
                    The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
                conditioning_scale (`float`, defaults to `1.0`):
                    The scale factor for BrushNet outputs.
                class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                    Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
                timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                    Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                    timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                    embeddings.
                attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                    An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                    is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                    negative values to the attention scores corresponding to "discard" tokens.
                added_cond_kwargs (`dict`):
                    Additional conditions for the Stable Diffusion XL UNet.
                cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                    A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
                guess_mode (`bool`, defaults to `False`):
                    In this mode, the BrushNet encoder tries its best to recognize the input content of the input even if
                    you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
                return_dict (`bool`, defaults to `True`):
                    Whether or not to return a [`~models.brushnet.BrushNetOutput`] instead of a plain tuple.

            Returns:
                [`~models.brushnet.BrushNetOutput`] **or** `tuple`:
                    If `return_dict` is `True`, a [`~models.brushnet.BrushNetOutput`] is returned, otherwise a tuple is
                    returned where the first element is the sample tensor.
            """
            # check channel order
            channel_order = self.config.brushnet_conditioning_channel_order

            if channel_order == "rgb":
                # in rgb order by default
                ...
            elif channel_order == "bgr":
                brushnet_cond = torch.flip(brushnet_cond, dims=[1])
            else:
                raise ValueError(f"unknown `brushnet_conditioning_channel_order`: {channel_order}")

            if debug: print('BrushNet CA: attn mask')

            # prepare attention_mask
            if attention_mask is not None:
                attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)

            if debug: print('BrushNet CA: time')

            # 1. time
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = sample.device.type == "mps"
                if isinstance(timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])

            t_emb = self.time_proj(timesteps)

            # timesteps does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=sample.dtype)

            emb = self.time_embedding(t_emb, timestep_cond)
            aug_emb = None

            if self.class_embedding is not None:
                if class_labels is None:
                    raise ValueError("class_labels should be provided when num_class_embeds > 0")

                if self.config.class_embed_type == "timestep":
                    class_labels = self.time_proj(class_labels)

                class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
                emb = emb + class_emb

            if self.config.addition_embed_type is not None:
                if self.config.addition_embed_type == "text":
                    aug_emb = self.add_embedding(encoder_hidden_states)

                elif self.config.addition_embed_type == "text_time":
                    if "text_embeds" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                        )
                    text_embeds = added_cond_kwargs.get("text_embeds")
                    if "time_ids" not in added_cond_kwargs:
                        raise ValueError(
                            f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                        )
                    time_ids = added_cond_kwargs.get("time_ids")
                    time_embeds = self.add_time_proj(time_ids.flatten())
                    time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                    add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                    add_embeds = add_embeds.to(emb.dtype)
                    aug_emb = self.add_embedding(add_embeds)

            emb = emb + aug_emb if aug_emb is not None else emb

            if debug: print('BrushNet CA: pre-process')


            # 2. pre-process
            brushnet_cond = torch.concat([sample, brushnet_cond], 1)
            sample = self.conv_in_condition(brushnet_cond)

            if debug: print('BrushNet CA: down')

            # 3. down
            down_block_res_samples = (sample,)
            for downsample_block in self.down_blocks:
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    if debug: print('BrushNet CA (down block with XA): ', type(downsample_block))
                    sample, res_samples = downsample_block(
                        hidden_states=sample,
                        temb=emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                        debug=debug,
                    )
                else:
                    if debug: print('BrushNet CA (down block): ', type(downsample_block))
                    sample, res_samples = downsample_block(hidden_states=sample, temb=emb, debug=debug)

                down_block_res_samples += res_samples

            if debug: print('BrushNet CA: PP down')

            # 4. PaintingNet down blocks
            brushnet_down_block_res_samples = ()
            for down_block_res_sample, brushnet_down_block in zip(down_block_res_samples, self.brushnet_down_blocks):
                down_block_res_sample = brushnet_down_block(down_block_res_sample)
                brushnet_down_block_res_samples = brushnet_down_block_res_samples + (down_block_res_sample,)

            if debug: print('BrushNet CA: PP mid')

            # 5. mid
            if self.mid_block is not None:
                if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                    sample = self.mid_block(
                        sample,
                        emb,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=attention_mask,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )
                else:
                    sample = self.mid_block(sample, emb)

            if debug: print('BrushNet CA: mid')

            # 6. BrushNet mid blocks
            brushnet_mid_block_res_sample = self.brushnet_mid_block(sample)

            if debug: print('BrushNet CA: PP up')

            # 7. up
            up_block_res_samples = ()
            for i, upsample_block in enumerate(self.up_blocks):
                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

                # if we have not reached the final block and need to forward the
                # upsample size, we do it here
                if not is_final_block:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                    sample, up_res_samples = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        upsample_size=upsample_size,
                        attention_mask=attention_mask,
                        return_res_samples=True,
                    )
                else:
                    sample, up_res_samples = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                        return_res_samples=True,
                    )

                up_block_res_samples += up_res_samples

            if debug: print('BrushNet CA: up')

            # 8. BrushNet up blocks
            brushnet_up_block_res_samples = ()
            for up_block_res_sample, brushnet_up_block in zip(up_block_res_samples, self.brushnet_up_blocks):
                up_block_res_sample = brushnet_up_block(up_block_res_sample)
                brushnet_up_block_res_samples = brushnet_up_block_res_samples + (up_block_res_sample,)

            if debug: print('BrushNet CA: scaling')

            # 6. scaling
            if guess_mode and not self.config.global_pool_conditions:
                scales = torch.logspace(
                    -1,
                    0,
                    len(brushnet_down_block_res_samples) + 1 + len(brushnet_up_block_res_samples),
                    device=sample.device,
                )  # 0.1 to 1.0
                scales = scales * conditioning_scale

                brushnet_down_block_res_samples = [
                    sample * scale
                    for sample, scale in zip(
                        brushnet_down_block_res_samples, scales[: len(brushnet_down_block_res_samples)]
                    )
                ]
                brushnet_mid_block_res_sample = (
                    brushnet_mid_block_res_sample * scales[len(brushnet_down_block_res_samples)]
                )
                brushnet_up_block_res_samples = [
                    sample * scale
                    for sample, scale in zip(
                        brushnet_up_block_res_samples, scales[len(brushnet_down_block_res_samples) + 1 :]
                    )
                ]
            else:
                brushnet_down_block_res_samples = [
                    sample * conditioning_scale for sample in brushnet_down_block_res_samples
                ]
                brushnet_mid_block_res_sample = brushnet_mid_block_res_sample * conditioning_scale
                brushnet_up_block_res_samples = [sample * conditioning_scale for sample in brushnet_up_block_res_samples]

            if self.config.global_pool_conditions:
                brushnet_down_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in brushnet_down_block_res_samples
                ]
                brushnet_mid_block_res_sample = torch.mean(brushnet_mid_block_res_sample, dim=(2, 3), keepdim=True)
                brushnet_up_block_res_samples = [
                    torch.mean(sample, dim=(2, 3), keepdim=True) for sample in brushnet_up_block_res_samples
                ]

            if debug: print('BrushNet CA: finish')

            if not return_dict:
                return (brushnet_down_block_res_samples, brushnet_mid_block_res_sample, brushnet_up_block_res_samples)

            return BrushNetOutput(
                down_block_res_samples=brushnet_down_block_res_samples,
                mid_block_res_sample=brushnet_mid_block_res_sample,
                up_block_res_samples=brushnet_up_block_res_samples,
            )

except ImportError:
    BrushNetModel = None
    PowerPaintModel = None
    # print("\33[33mModule 'diffusers' load failed. If you don't have it installed, do it:\033[0m")
    # print("\33[33mpip install diffusers\033[0m")
#taken from: https://github.com/lllyasviel/ControlNet
#and modified

import torch
import torch as th
import torch.nn as nn

from ..ldm.modules.diffusionmodules.util import (
    zero_module,
    timestep_embedding,
)

from ..ldm.modules.attention import SpatialTransformer
from ..ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample
from ..ldm.util import exists
from collections import OrderedDict
import comfy.ops
from comfy.ldm.modules.attention import optimized_attention

class OptimizedAttention(nn.Module):
    def __init__(self, c, nhead, dropout=0.0, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = nhead
        self.c = c

        self.in_proj = operations.Linear(c, c * 3, bias=True, dtype=dtype, device=device)
        self.out_proj = operations.Linear(c, c, bias=True, dtype=dtype, device=device)

    def forward(self, x):
        x = self.in_proj(x)
        q, k, v = x.split(self.c, dim=2)
        out = optimized_attention(q, k, v, self.heads)
        return self.out_proj(out)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResBlockUnionControlnet(nn.Module):
    def __init__(self, dim, nhead, dtype=None, device=None, operations=None):
        super().__init__()
        self.attn = OptimizedAttention(dim, nhead, dtype=dtype, device=device, operations=operations)
        self.ln_1 = operations.LayerNorm(dim, dtype=dtype, device=device)
        self.mlp = nn.Sequential(
            OrderedDict([("c_fc", operations.Linear(dim, dim * 4, dtype=dtype, device=device)), ("gelu", QuickGELU()),
                         ("c_proj", operations.Linear(dim * 4, dim, dtype=dtype, device=device))]))
        self.ln_2 = operations.LayerNorm(dim, dtype=dtype, device=device)

    def attention(self, x: torch.Tensor):
        return self.attn(x)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ControlledUnetModel(UNetModel):
    #implemented in the ldm unet
    pass

class ControlNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        dtype=torch.float32,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        attn_precision=None,
        union_controlnet_num_control_type=None,
        device=None,
        operations=comfy.ops.disable_weight_init,
        **kwargs,
    ):
        super().__init__()
        assert use_spatial_transformer == True, "use_spatial_transformer has to be true"
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            # from omegaconf.listconfig import ListConfig
            # if type(context_dim) == ListConfig:
            #     context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))

        transformer_depth = transformer_depth[:]

        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        operations.Linear(adm_in_channels, time_embed_dim, dtype=self.dtype, device=device),
                        nn.SiLU(),
                        operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels, operations=operations, dtype=self.dtype, device=device)])

        self.input_hint_block = TimestepEmbedSequential(
                    operations.conv_nd(dims, hint_channels, 16, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 16, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 16, 32, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 32, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 32, 96, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 96, 3, padding=1, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 96, 256, 3, padding=1, stride=2, dtype=self.dtype, device=device),
                    nn.SiLU(),
                    operations.conv_nd(dims, 256, model_channels, 3, padding=1, dtype=self.dtype, device=device)
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        dtype=self.dtype,
                        device=device,
                        operations=operations,
                    )
                ]
                ch = mult * model_channels
                num_transformers = transformer_depth.pop(0)
                if num_transformers > 0:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(
                                ch, num_heads, dim_head, depth=num_transformers, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=self.dtype, device=device, operations=operations
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            dtype=self.dtype,
                            device=device,
                            operations=operations
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        mid_block = [
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]
        if transformer_depth_middle >= 0:
            mid_block += [SpatialTransformer(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth_middle, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint, attn_precision=attn_precision, dtype=self.dtype, device=device, operations=operations
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                device=device,
                operations=operations
            )]
        self.middle_block = TimestepEmbedSequential(*mid_block)
        self.middle_block_out = self.make_zero_conv(ch, operations=operations, dtype=self.dtype, device=device)
        self._feature_size += ch

        if union_controlnet_num_control_type is not None:
            self.num_control_type = union_controlnet_num_control_type
            num_trans_channel = 320
            num_trans_head = 8
            num_trans_layer = 1
            num_proj_channel = 320
            # task_scale_factor = num_trans_channel ** 0.5
            self.task_embedding = nn.Parameter(torch.empty(self.num_control_type, num_trans_channel, dtype=self.dtype, device=device))

            self.transformer_layes = nn.Sequential(*[ResBlockUnionControlnet(num_trans_channel, num_trans_head, dtype=self.dtype, device=device, operations=operations) for _ in range(num_trans_layer)])
            self.spatial_ch_projs = operations.Linear(num_trans_channel, num_proj_channel, dtype=self.dtype, device=device)
            #-----------------------------------------------------------------------------------------------------

            control_add_embed_dim = 256
            class ControlAddEmbedding(nn.Module):
                def __init__(self, in_dim, out_dim, num_control_type, dtype=None, device=None, operations=None):
                    super().__init__()
                    self.num_control_type = num_control_type
                    self.in_dim = in_dim
                    self.linear_1 = operations.Linear(in_dim * num_control_type, out_dim, dtype=dtype, device=device)
                    self.linear_2 = operations.Linear(out_dim, out_dim, dtype=dtype, device=device)
                def forward(self, control_type, dtype, device):
                    c_type = torch.zeros((self.num_control_type,), device=device)
                    c_type[control_type] = 1.0
                    c_type = timestep_embedding(c_type.flatten(), self.in_dim, repeat_only=False).to(dtype).reshape((-1, self.num_control_type * self.in_dim))
                    return self.linear_2(torch.nn.functional.silu(self.linear_1(c_type)))

            self.control_add_embedding = ControlAddEmbedding(control_add_embed_dim, time_embed_dim, self.num_control_type, dtype=self.dtype, device=device, operations=operations)
        else:
            self.task_embedding = None
            self.control_add_embedding = None

    def union_controlnet_merge(self, hint, control_type, emb, context):
        # Equivalent to: https://github.com/xinsir6/ControlNetPlus/tree/main
        inputs = []
        condition_list = []

        for idx in range(min(1, len(control_type))):
            controlnet_cond = self.input_hint_block(hint[idx], emb, context)
            feat_seq = torch.mean(controlnet_cond, dim=(2, 3))
            if idx < len(control_type):
                feat_seq += self.task_embedding[control_type[idx]]

            inputs.append(feat_seq.unsqueeze(1))
            condition_list.append(controlnet_cond)

        x = torch.cat(inputs, dim=1)
        x = self.transformer_layes(x)
        controlnet_cond_fuser = None
        for idx in range(len(control_type)):
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            o = condition_list[idx] + alpha
            if controlnet_cond_fuser is None:
                controlnet_cond_fuser = o
            else:
                controlnet_cond_fuser += o
        return controlnet_cond_fuser

    def make_zero_conv(self, channels, operations=None, dtype=None, device=None):
        return TimestepEmbedSequential(operations.conv_nd(self.dims, channels, channels, 1, padding=0, dtype=dtype, device=device))

    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        emb = self.time_embed(t_emb)

        guided_hint = None
        if self.control_add_embedding is not None: #Union Controlnet
            control_type = kwargs.get("control_type", [])

            emb += self.control_add_embedding(control_type, emb.dtype, emb.device)
            if len(control_type) > 0:
                if len(hint.shape) < 5:
                    hint = hint.unsqueeze(dim=0)
                guided_hint = self.union_controlnet_merge(hint, control_type, emb, context)

        if guided_hint is None:
            guided_hint = self.input_hint_block(hint, emb, context)

        out_output = []
        out_middle = []

        hs = []
        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            out_output.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        out_middle.append(self.middle_block_out(h, emb, context))

        return {"middle": out_middle, "output": out_output}


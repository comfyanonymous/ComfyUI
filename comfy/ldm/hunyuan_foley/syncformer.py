#!/usr/bin/env python3

from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat, einops

from comfy.ldm.hunyuan3dv2_1.hunyuandit import MLP as Mlp
from transformers.modeling_outputs import BaseModelOutputWithPooling
from comfy.ldm.modules.attention import optimized_attention, TransformerEncoderComfyv
from transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer import ASTConfig

from typing import Optional, Union, Tuple

class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, device=None, dtype=None, operations=None):
        super().__init__()
        img_size = img_size if type(img_size) is tuple else (img_size, img_size)
        patch_size = img_size if type(patch_size) is tuple else (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = operations.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, device=device, dtype=dtype)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        patch_size=16,
        z_block_size=2,
        embed_dim=768,
        flatten=True,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        self.height = img_size // patch_size
        self.width = img_size // patch_size
        self.z_block_size = z_block_size
        self.proj = operations.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size),
            stride=(z_block_size, patch_size, patch_size),
            device=device, dtype=dtype
        )
        self.flatten = flatten

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x

def qkv_attn(q, k, v, heads):
    bh, seq_q, dim_head = q.shape
    b = bh // heads

    # (b*heads, seq, dim) -> (b, heads, seq, dim)
    q2 = q.view(b, heads, seq_q, dim_head)
    k2 = k.view(b, heads, k.shape[1], dim_head)
    v2 = v.view(b, heads, v.shape[1], dim_head)

    out = optimized_attention(q2, k2, v2, heads=heads, skip_reshape=True)

    out = out.permute(0, 2, 1, 3).contiguous().view(b * heads, seq_q, dim_head)

    return out


class DividedAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = operations.Linear(dim, dim * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, einops_from, einops_to, tok_mask: torch.Tensor = None, **einops_dims):
        h = self.num_heads

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        cls_out = qkv_attn(cls_q, k, v, self.num_heads)

        q_, k_, v_ = map(lambda t: rearrange(t, f"{einops_from} -> {einops_to}", **einops_dims), (q_, k_, v_))

        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, "b () d -> (b r) () d", r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        out = qkv_attn(q_, k_, v_, self.num_heads)
        out = rearrange(out, f"{einops_to} -> {einops_from}", **einops_dims)

        out = torch.cat((cls_out, out), dim=1)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        x = self.proj(out)
        return x

class DividedSpaceTimeBlock(nn.Module):

    def __init__(
        self,
        dim=768,
        num_heads=12,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        device = None, dtype = None, operations = None
    ):
        super().__init__()

        factory_kwargs = {"device":device, "dtype": dtype}

        self.einops_from_space = "b (f n) d"
        self.einops_to_space = "(b f) n d"
        self.einops_from_time = "b (f n) d"
        self.einops_to_time = "(b n) f d"

        self.norm1 = norm_layer(dim)

        self.attn = DividedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, operations = operations, **factory_kwargs)

        self.timeattn = DividedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, operations=operations, **factory_kwargs
        )

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(width = dim, operations = operations, device=device, dtype=dtype)
        self.norm3 = norm_layer(dim)

    def forward(self, x, seq_len=196, num_frames=8, tok_mask: torch.Tensor = None):
        time_output = self.timeattn(
            self.norm3(x), self.einops_from_time, self.einops_to_time, n=seq_len, tok_mask=tok_mask
        )
        time_residual = x + time_output

        space_output = self.attn(
            self.norm1(time_residual), self.einops_from_space, self.einops_to_space, f=num_frames, tok_mask=tok_mask
        )
        space_residual = time_residual + self.drop_path(space_output)

        x = space_residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MotionFormer(nn.Module):
    def __init__(self, device = None, dtype = None, operations = None):
        super().__init__()
        self.APPROX_ATTN_TYPE = "none"
        self.APPROX_ATTN_DIM = 64 
        self.img_size = 224
        self.patch_size = 16
        self.in_chans = 3
        self.num_classes = 174
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.drop_rate = 0.0
        self.drop_path_rate = 0.2
        self.temporal_resolution = 8
        self.use_mlp = True
        self.num_features = self.embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.attn_drop_rate = 0.0
        self.factorize_space_time = True

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim,
            device=device, dtype=dtype, operations=operations
        )

        # 3D Patch Embedding
        self.patch_embed_3d = PatchEmbed3D(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            z_block_size = 2,
            device=device, dtype=dtype, operations=operations
        )
        self.patch_embed_3d.proj.weight.data = torch.zeros_like(self.patch_embed_3d.proj.weight.data)

        # Number of patches
        self.num_patches = self.patch_embed.num_patches * self.temporal_resolution

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim, device=device, dtype=dtype))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim, device=device, dtype=dtype))
        self.pos_drop = nn.Dropout(p=0.0)

        self.temp_embed = nn.Parameter(torch.zeros(1, self.temporal_resolution, self.embed_dim, device=device, dtype=dtype))

        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    qkv_bias=self.qkv_bias,
                    norm_layer=norm_layer,
                    device=device, dtype=dtype, operations=operations
                )
                for _ in range(self.depth)
            ]
        )

        self.norm = norm_layer(self.embed_dim)

        self.pre_logits = nn.Identity()
        
        transf_enc_layer_kwargs = dict(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            activation=nn.GELU(),
            batch_first=True,
            dim_feedforward=self.mlp_ratio * self.embed_dim,
            dropout=self.drop_rate,
            layer_norm_eps=1e-6,
            norm_first=True,
        )
        self.spatial_attn_agg = SpatialTransformerEncoderLayer(device = device, dtype=dtype, operations=operations,**transf_enc_layer_kwargs)
        self.temp_attn_agg = nn.Identity()

    def forward_features(self, x):

        B = x.shape[0]

        # apply patching on input
        x = self.patch_embed_3d(x)
        tok_mask = None

        # Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        new_pos_embed = self.pos_embed
        npatch = self.patch_embed.num_patches

        cls_embed = self.pos_embed[:, 0, :].unsqueeze(1)
        tile_pos_embed = new_pos_embed[:, 1:, :].repeat(1, self.temporal_resolution, 1)
        tile_temporal_embed = self.temp_embed.repeat_interleave(npatch, 1)
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=1)
        x = x + total_pos_embed

        # Apply positional dropout
        x = self.pos_drop(x)

        # Encoding using transformer layers
        for i, blk in enumerate(self.blocks):
            x = blk(
                x,
                seq_len=npatch,
                num_frames=self.temporal_resolution,
                tok_mask=tok_mask,
            )

        return x, tok_mask
    
    def forward(self, x):
        B, S, C, T, H, W = x.shape

        orig_shape = (B, S, C, T, H, W)
        x = x.view(B * S, C, T, H, W)  # flatten batch and segments
        x = self.forward_segments(x, orig_shape=orig_shape)
        x = x.view(B, S, *x.shape[1:])

        return x

    def forward_segments(self, x, orig_shape: tuple) -> torch.Tensor:
        x, x_mask = self.forward_features(x)

        x = x[:, 1:, :]
        x = self.norm(x)
        x = self.pre_logits(x)
        if self.factorize_space_time:
            x = self.restore_spatio_temp_dims(x, orig_shape)

            x = self.spatial_attn_agg(x, x_mask)
            x = self.temp_attn_agg(x)

        return x

    def restore_spatio_temp_dims(self, feats: torch.Tensor, orig_shape: tuple) -> torch.Tensor:

        B, S, C, T, H, W = orig_shape
        D = self.embed_dim

        # num patches in each dimension
        t = T // self.patch_embed_3d.z_block_size
        h = self.patch_embed_3d.height
        w = self.patch_embed_3d.width

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(B * S, D, t, h, w)  # (B*S, D, t, h, w)

        return feats

class BaseEncoderLayer(TransformerEncoderComfyv):
    def __init__(
        self,
        add_pos_emb: bool = False,
        pos_emb_drop: float = None,
        pos_max_len: int = None,
        device = None,
        dtype = None, operations = None,
        *args, **kwargs
    ):  
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(operations = operations, *args, **kwargs, **factory_kwargs)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim, **factory_kwargs))

        self.add_pos_emb = add_pos_emb
        if add_pos_emb:
            self.pos_max_len = 1 + pos_max_len
            self.pos_emb = nn.Parameter(torch.zeros(1, self.pos_max_len, self.self_attn.embed_dim, **factory_kwargs))
            self.pos_drop = nn.Dropout(pos_emb_drop)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        batch_dim = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_dim, -1, -1)
        x = torch.cat((cls_tokens, x), dim=-2)
        if x_mask is not None:
            cls_mask = torch.ones((batch_dim, 1), dtype=torch.bool, device=x_mask.device)
            x_mask_w_cls = torch.cat((cls_mask, x_mask), dim=-1)
            B, N = x_mask_w_cls.shape
            x_mask_w_cls = (
                x_mask_w_cls.reshape(B, 1, 1, N)
                .expand(-1, self.self_attn.num_heads, N, -1)
                .reshape(B * self.self_attn.num_heads, N, N)
            )
            assert x_mask_w_cls.dtype == x_mask_w_cls.bool().dtype, "x_mask_w_cls.dtype != bool"
            x_mask_w_cls = ~x_mask_w_cls  # invert mask (1=mask)
        else:
            x_mask_w_cls = None

        # add positional embedding
        if self.add_pos_emb:
            seq_len = x.shape[1]
            assert seq_len <= self.pos_max_len, f"Seq len ({seq_len}) > pos_max_len ({self.pos_max_len})"
            x = x + self.pos_emb[:, :seq_len, :]
            x = self.pos_drop(x)

        x = super().forward(src=x, src_mask=x_mask_w_cls)

        x = x[:, 0, :]

        return x

class SpatialTransformerEncoderLayer(BaseEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        BS, D, t, h, w = x.shape

        x = rearrange(x, "BS D t h w -> (BS t) (h w) D")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "BS t h w -> (BS t) (h w)")

        x = super().forward(x=x, x_mask=x_mask)

        x = rearrange(x, "(BS t) D -> BS t D", BS=BS, t=t)

        return x
    
class AST(torch.nn.Module):
    def __init__(
        self,
        max_spec_t: int = None,
        factorize_freq_time: bool = None,
        max_segments: int = None,
        device = None, dtype = None, operations = None
    ) -> None:
        
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.extract_features = True
        self.max_spec_t = max_spec_t
        self.max_segments = max_segments

        self.config = ASTConfig()
        self.config.num_labels = 527

        self.ast = ASTModel(self.config, device=device, dtype=dtype, operations=operations)

        self.feat_type = "last_hidden_state"
        self.factorize_freq_time = factorize_freq_time

        transf_enc_layer_kwargs = dict(
            d_model=self.config.hidden_size,
            nhead=self.config.num_attention_heads,
            dim_feedforward=self.config.intermediate_size,
            activation=torch.nn.GELU(),
            batch_first=True,
            dropout=self.config.attention_probs_dropout_prob,
            layer_norm_eps=1e-6,
            norm_first=True,
        )
        if factorize_freq_time:
            self.feat_type = "last_hidden_state"
            self.freq_attn_agg = FrequencyTransformerEncoderLayer(operations = operations, **transf_enc_layer_kwargs, **factory_kwargs)
            self.temp_attn_agg = torch.nn.Identity()

        self.device = device

        self.patch_position_emb()

    def forward(
        self, x: torch.Tensor, for_loop: bool = False, cont_mask: torch.Tensor = None, **ast_kwargs
    ) -> torch.Tensor:

        B, S, T, F = x.shape

        if for_loop:
            assert cont_mask is None, "cont_mask is not supported with for_loop=True"
            orig_shape_s = (B, 1, T, F)
            x = torch.cat(
                [self.forward_segments(x[:, s], orig_shape_s, **ast_kwargs).unsqueeze(1) for s in range(S)], dim=1
            )
        else:
            orig_shape = (B, S, T, F)
            x = x.view(B * S, T, F)
            if cont_mask is not None:
                cont_mask = cont_mask.reshape(B * S, T, F)
            x = self.forward_segments(x, orig_shape=orig_shape, cont_mask=cont_mask, **ast_kwargs)
            x = x.view(B, S, *x.shape[1:])

        global_x = None

        return x, global_x

    def forward_segments(self, x, orig_shape: tuple, cont_mask: torch.Tensor = None, **ast_kwargs):

        x, x_mask = self.ast(x, cont_mask=cont_mask, **ast_kwargs)

        if self.extract_features:
            x = self.get_features_by_type(x)
            if self.factorize_freq_time:
                x = self.restore_freq_temp_dims(x, orig_shape)
                if cont_mask is not None:
                    x_mask = x_mask.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
                    x_mask = self.restore_freq_temp_dims(x_mask, orig_shape)
                    x_mask = x_mask[:, 0, :, :]
                else:
                    x_mask = None
                x = self.freq_attn_agg(x, x_mask)
                x = self.temp_attn_agg(x)
        else:
            x = x["pooler_output"]
            x = self.classifier(x)
        return x

    def get_features_by_type(self, x) -> torch.Tensor:
        return x["last_hidden_state"]  # (B, 2+T, D)

    def restore_freq_temp_dims(self, feats, orig_shape: tuple):
        B, S, T, F = orig_shape
        D = self.config.hidden_size

        # num patches in each dimension
        f, t = self.ast.embeddings.get_shape(self.config)

        if self.feat_type == "last_hidden_state":
            feats = feats[:, 2:, :]  # removing CLS and distill tokens

        feats = feats.permute(0, 2, 1)  # (B*S, D, T)
        feats = feats.view(B * S, D, f, t)  # (B*S, D, f, t)

        return feats

    def patch_position_emb(self):
        if self.max_spec_t is not None:
            self.config.max_length = self.max_spec_t
        f, t = self.ast.embeddings.get_shape(self.config)
        shortened = self.ast.embeddings.position_embeddings[:, : f * t + 2].clone()  # +2 for CLS and distill tokens
        self.ast.embeddings.position_embeddings = torch.nn.Parameter(shortened).to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        return super().to(device)


class FrequencyTransformerEncoderLayer(BaseEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        BS, D, f, t = x.shape

        x = x.permute(0, 3, 2, 1)
        x = x.reshape(BS * t, f, D)
        if x_mask is not None:
            x_mask = x_mask.permute(0, 2, 1)
            x_mask = x_mask.reshape(BS * t, f)

        x = super().forward(x=x, x_mask=x_mask)

        x = x.view(BS, t, D)

        return x
    
class ASTEmbeddings(nn.Module):

    def __init__(self, config: ASTConfig, device = None, dtype = None, operations = None) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size, device=device, dtype=dtype))
        self.patch_embeddings = ASTPatchEmbeddings(config, device, dtype, operations)

        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        num_patches = frequency_out_dimension * time_out_dimension
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size, device=device, dtype=dtype))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def get_shape(self, config):
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        batch_size = input_values.shape[0]
        embeddings = self.patch_embeddings(input_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class ASTPatchEmbeddings(nn.Module):
    def __init__(self, config, device = None, dtype = None, operations = None):
        super().__init__()


        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        self.projection = operations.Conv2d(
            1, config.hidden_size, kernel_size=(patch_size, patch_size), stride=(frequency_stride, time_stride), device = device, dtype = dtype
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        input_values = input_values.unsqueeze(1)
        input_values = input_values.transpose(2, 3)
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        return embeddings


class ASTSelfAttention(nn.Module):
    def __init__(self, config: ASTConfig, device = None, dtype = None, operations = None) -> None:
        super().__init__()
        factory_kwargs = { "device": device, "dtype": dtype }
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = operations.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias, **factory_kwargs)
        self.key = operations.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias, **factory_kwargs)
        self.value = operations.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias, **factory_kwargs)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        tok_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        if tok_mask is not None:
            attn_mask = (tok_mask == 0)
            attn_mask = attn_mask[:, None, None, :]
        else:
            attn_mask = None
        context_layer = optimized_attention(query_layer, key_layer, value_layer, self.num_attention_heads, mask = attn_mask, skip_output_reshape=True, skip_reshape=True)
        context_layer = context_layer.view(*query_layer.size())

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return (context_layer,)

class ASTSelfOutput(nn.Module):

    def __init__(self, config: ASTConfig, device=None, dtype=None, operations=None) -> None:
        super().__init__()
        self.dense = operations.Linear(config.hidden_size, config.hidden_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig, device=None, dtype=None, operations=None) -> None:
        super().__init__()
        self.attention = ASTSelfAttention(config, device=device, dtype=dtype, operations=operations)
        self.output = ASTSelfOutput(config, device=device, dtype=dtype, operations=operations)

    def forward(
        self,
        hidden_states: torch.Tensor,
        tok_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, tok_mask, head_mask)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig, device, dtype, operations) -> None:
        super().__init__()
        self.dense = operations.Linear(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig, device, dtype, operations) -> None:
        super().__init__()
        self.dense = operations.Linear(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

class ASTLayer(nn.Module):
    def __init__(self, config: ASTConfig, device=None, dtype=None, operations=None) -> None:
        super().__init__()
        factory_kwargs = {"device":device, "dtype":dtype}
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ASTAttention(config, operations = operations, **factory_kwargs)
        self.intermediate = ASTIntermediate(config, operations=operations, **factory_kwargs)
        self.output = ASTOutput(config, operations=operations, **factory_kwargs)
        self.layernorm_before = operations.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **factory_kwargs)
        self.layernorm_after = operations.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, **factory_kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        tok_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            tok_mask,
            head_mask,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig, device, dtype, operations) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ASTLayer(config, device, dtype, operations) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        tok_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, tok_mask, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]

        return hidden_states

class ASTModel(nn.Module):
    def __init__(self, config: ASTConfig, device, dtype, operations):
        super().__init__()
        self.config = config

        self.embeddings = ASTEmbeddings(config, device, dtype, operations)
        self.encoder = ASTEncoder(config, device, dtype, operations)

        self.layernorm = operations.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        cont_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_values)

        if cont_mask is not None:
            indicator = torch.ones_like(input_values).to(input_values.dtype)
            indicator[~cont_mask] = torch.inf
            with torch.no_grad():
                indicator = self.embeddings(indicator)
            tok_mask = ~torch.isnan(indicator)
            tok_mask = tok_mask[:, :, 0]
        else:
            tok_mask = None

        encoder_outputs = self.encoder(
            embedding_output,
            tok_mask=tok_mask,
            head_mask=head_mask,
        )
        sequence_output = encoder_outputs
        sequence_output = self.layernorm(sequence_output)

        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        return (
            BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
            ),
            tok_mask,
        )
    
class ASTMLPHead(nn.Module):
    def __init__(self, config: ASTConfig, device, dtype, operations):
        super().__init__()
        self.layernorm = operations.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device, dtype=dtype)
        self.dense = operations.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.layernorm(hidden_state)
        hidden_state = self.dense(hidden_state)
        return hidden_state

class RandInitPositionalEncoding(nn.Module):
    def __init__(self, block_shape: list, n_embd: int, device = None, dtype = None,):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.randn(1, *block_shape, n_embd, device=device, dtype=dtype))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb


class GlobalTransformer(torch.nn.Module):
    def __init__(
        self,
        tok_pdrop=0.0,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        n_layer=3,
        n_head=8,
        n_embd=768,
        pos_emb_block_shape=[
            198,
        ],
        n_off_head_out=21,
        device = None, dtype = None, operations = None
    ) -> None:
        super().__init__()

        factory_kwargs = {"device":device, "dtype": dtype}
        self.config = Config(
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        # input norm
        self.vis_in_lnorm = operations.LayerNorm(n_embd, **factory_kwargs)
        self.aud_in_lnorm = operations.LayerNorm(n_embd, **factory_kwargs)
        # aux tokens
        self.OFF_tok = nn.Parameter(torch.randn(1, 1, n_embd, **factory_kwargs))
        self.MOD_tok = nn.Parameter(torch.randn(1, 1, n_embd, **factory_kwargs))
        # whole token dropout
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        # maybe add pos emb
        self.pos_emb_cfg = RandInitPositionalEncoding(
            block_shape=pos_emb_block_shape,
            n_embd=n_embd,
        )
        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = nn.Sequential(*[Block(self.config, operations=operations, **factory_kwargs) for _ in range(n_layer)])
        # pre-output norm
        self.ln_f = operations.LayerNorm(n_embd)
        # maybe add a head
        self.off_head = operations.Linear(in_features=n_embd, out_features=n_off_head_out)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape
        B, Sa, D = a.shape

        off_tok = einops.repeat(self.OFF_tok, "1 1 d -> b 1 d", b=B)
        mod_tok = einops.repeat(self.MOD_tok, "1 1 d -> b 1 d", b=B)

        v, a = self.vis_in_lnorm(v), self.aud_in_lnorm(a)

        if self.tok_pdrop > 0:
            v, a = self.tok_drop_vis(v), self.tok_drop_aud(a)

        x = torch.cat((off_tok, v, mod_tok, a), dim=1)
        if hasattr(self, "pos_emb_cfg"):
            x = self.pos_emb_cfg(x)

        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        if attempt_to_apply_heads and hasattr(self, "off_head"):
            x = self.off_head(x[:, 0, :])
        return x


class SelfAttention(nn.Module):

    def __init__(self, config, device, dtype, operations):
        super().__init__()

        self.key = operations.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.query = operations.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.value = operations.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        self.proj = operations.Linear(config.n_embd, config.n_embd, device=device, dtype=dtype)
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = optimized_attention(q, k, v, self.n_head, skip_reshape=True)

        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    def __init__(self, config, device, dtype, operations):
        super().__init__()
        factory_kwargs = {"device":device, "dtype":dtype}
        self.ln1 = operations.LayerNorm(config.n_embd, **factory_kwargs)
        self.ln2 = operations.LayerNorm(config.n_embd, **factory_kwargs)
        self.attn = SelfAttention(config, device, dtype, operations)
        self.mlp = nn.Sequential(
            operations.Linear(config.n_embd, 4 * config.n_embd, **factory_kwargs),
            nn.GELU(),
            operations.Linear(4 * config.n_embd, config.n_embd, **factory_kwargs),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class Synchformer(nn.Module):

    def __init__(self, device, dtype, operations):
        super().__init__()

        factory_kwargs = {"device":device, "dtype":dtype}

        self.vfeat_extractor = MotionFormer(operations = operations, **factory_kwargs)
        self.afeat_extractor = AST(
            operations = operations,
            max_spec_t = 66,
            factorize_freq_time = True,
            **factory_kwargs
        )

        self.vproj = operations.Linear(in_features=768, out_features=768, **factory_kwargs)
        self.aproj = operations.Linear(in_features=768, out_features=768, **factory_kwargs)
        self.transformer = GlobalTransformer(
            tok_pdrop=0.0, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=3, n_head=8, n_embd=768, operations=operations, **factory_kwargs
        )

    def forward(self, vis):
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        vis = self.vfeat_extractor(vis)
        return vis

    def compare_v_a(self, vis: torch.Tensor, aud: torch.Tensor):
        vis = self.vproj(vis)
        aud = self.aproj(aud)

        B, S, tv, D = vis.shape
        B, S, ta, D = aud.shape
        vis = vis.view(B, S * tv, D)
        aud = aud.view(B, S * ta, D)

        logits = self.transformer(vis, aud)

        return logits

    def extract_vfeats(self, vis):
        return self.vfeat_extractor(vis.permute(0, 1, 3, 2, 4, 5))

    def extract_afeats(self, aud):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)
        aud, _ = self.afeat_extractor(aud)
        return aud

from contextlib import nullcontext
from typing import Literal, Optional, Tuple
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from contextlib import contextmanager
from comfy.utils import ProgressBar

from comfy.ldm.seedvr.model import safe_pad_operation
from comfy.ldm.seedvr.constants import (
    BYTEDANCE_BLOCK_OUT_CHANNELS,
    BYTEDANCE_CONTIGUOUS_BATCH_THRESHOLD,
    BYTEDANCE_GN_CHUNKS_FP16,
    BYTEDANCE_GN_CHUNKS_FP32,
    BYTEDANCE_LOGVAR_CLAMP_MAX,
    BYTEDANCE_LOGVAR_CLAMP_MIN,
    BYTEDANCE_SLICING_SAMPLE_MIN,
    BYTEDANCE_VAE_CONV_MEM_GIB,
    BYTEDANCE_VAE_NORM_MEM_GIB,
    BYTEDANCE_VAE_SCALING_FACTOR,
    BYTEDANCE_VAE_SHIFTING_FACTOR,
    BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE,
    BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE,
    SEEDVR2_LATENT_CHANNELS,
)
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.model import vae_attention

import math
from enum import Enum
from comfy.ops import NVIDIA_MEMORY_CONV_BUG_WORKAROUND

import logging
import comfy.model_management
import comfy.ops
ops = comfy.ops.disable_weight_init


def _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap, temporal_scale=1):
    if temporal_size is None:
        return None

    temporal_size = int(temporal_size)
    if temporal_size <= 0:
        return 0

    temporal_overlap = max(0, int(temporal_overlap or 0))
    temporal_overlap = min(temporal_overlap, temporal_size - 1)
    temporal_step = temporal_size - temporal_overlap
    temporal_scale = max(1, int(temporal_scale))
    return max(1, math.ceil(temporal_step / temporal_scale))


def _seedvr2_clamped_spatial_overlap(overlap, tile_size):
    overlap = max(0, int(overlap))
    tile_size = max(1, int(tile_size))
    return min(overlap, tile_size - 1)


def _seedvr2_clear_temporal_memory(model):
    for module in model.modules():
        if hasattr(module, "memory"):
            module.memory = None


@torch.inference_mode()
def tiled_vae(
    x,
    vae_model,
    tile_size=(512, 512),
    tile_overlap=(64, 64),
    temporal_size=16,
    temporal_overlap=0,
    encode=True,
    **kwargs,
):
    gc.collect()
    comfy.model_management.soft_empty_cache()

    x = x.to(next(vae_model.parameters()).dtype)
    if x.ndim != 5:
        x = x.unsqueeze(2)

    _, _, d, h, w = x.shape

    sf_s = getattr(vae_model, "spatial_downsample_factor", BYTEDANCE_VAE_SPATIAL_DOWNSAMPLE)
    sf_t = getattr(vae_model, "temporal_downsample_factor", BYTEDANCE_VAE_TEMPORAL_DOWNSAMPLE)
    if encode:
        slicing_attr = "slicing_sample_min_size"
        slicing_min_size = _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap)
    else:
        slicing_attr = "slicing_latent_min_size"
        slicing_min_size = _seedvr2_temporal_slicing_min_size(temporal_size, temporal_overlap, sf_t)
    if encode:
        ti_h, ti_w = tile_size
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0], ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1], ti_w)
        blend_ov_h = max(0, ov_h // sf_s)
        blend_ov_w = max(0, ov_w // sf_s)
        target_d = (d + sf_t - 1) // sf_t
        target_h = (h + sf_s - 1) // sf_s
        target_w = (w + sf_s - 1) // sf_s
    else:
        ti_h = max(1, tile_size[0] // sf_s)
        ti_w = max(1, tile_size[1] // sf_s)
        ov_h = _seedvr2_clamped_spatial_overlap(tile_overlap[0] // sf_s, ti_h)
        ov_w = _seedvr2_clamped_spatial_overlap(tile_overlap[1] // sf_s, ti_w)
        blend_ov_h = ov_h * sf_s
        blend_ov_w = ov_w * sf_s

        target_d = max(1, d * sf_t - (sf_t - 1))
        target_h = h * sf_s
        target_w = w * sf_s

    stride_h = max(1, ti_h - ov_h)
    stride_w = max(1, ti_w - ov_w)

    storage_device = vae_model.device
    result = None
    count = None
    def run_temporal_chunks(spatial_tile, model=vae_model, device=storage_device):
        device = torch.device(device)
        _seedvr2_clear_temporal_memory(model)
        t_chunk = spatial_tile.to(device=device, dtype=next(model.parameters()).dtype, non_blocking=True).contiguous()
        old_device = getattr(model, "device", None)
        model.device = device
        old_slicing_min_size = getattr(model, slicing_attr, None)
        if old_slicing_min_size is not None and slicing_min_size is not None:
            if slicing_min_size <= 0:
                setattr(model, slicing_attr, t_chunk.shape[2])
            else:
                setattr(model, slicing_attr, slicing_min_size)
        try:
            if encode:
                out = model.encode(t_chunk)[0]
            else:
                out = model.decode_(t_chunk)
        finally:
            if old_slicing_min_size is not None and slicing_min_size is not None:
                setattr(model, slicing_attr, old_slicing_min_size)
            if old_device is not None:
                model.device = old_device
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim == 4:
            out = out.unsqueeze(2)
        return out.to(storage_device)

    ramp_cache = {}
    def get_ramp(steps):
        if steps not in ramp_cache:
            t = torch.linspace(0, 1, steps=steps, device=storage_device, dtype=torch.float32)
            ramp_cache[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramp_cache[steps]

    tile_ranges = []
    for y_idx in range(0, h, stride_h):
        y_end = min(y_idx + ti_h, h)
        if y_idx > 0 and (y_end - y_idx) <= ov_h:
            continue
        for x_idx in range(0, w, stride_w):
            x_end = min(x_idx + ti_w, w)
            if x_idx > 0 and (x_end - x_idx) <= ov_w:
                continue
            tile_ranges.append((y_idx, y_end, x_idx, x_end))

    total_tiles = len(tile_ranges)
    bar = ProgressBar(total_tiles)
    single_spatial_tile = h <= ti_h and w <= ti_w

    _seedvr2_clear_temporal_memory(vae_model)

    def run_tile(tile_index, tile_range):
        y_idx, y_end, x_idx, x_end = tile_range
        tile_x = x[:, :, :, y_idx:y_end, x_idx:x_end]
        tile_out = run_temporal_chunks(tile_x)
        return tile_index, y_idx, y_end, x_idx, x_end, tile_out

    ordered_tile_outputs = (
        run_tile(tile_index, tile_range)
        for tile_index, tile_range in enumerate(tile_ranges)
    )

    for _, y_idx, y_end, x_idx, x_end, tile_out in ordered_tile_outputs:

        if single_spatial_tile:
            result = tile_out[:, :, :target_d, :target_h, :target_w]
            if result.device != x.device:
                result = result.to(x.device).to(x.dtype)
            if x.shape[2] == 1 and sf_t == 1:
                result = result.squeeze(2)
            bar.update(1)
            return result

        if result is None:
            b_out, c_out = tile_out.shape[0], tile_out.shape[1]
            result = torch.zeros((b_out, c_out, target_d, target_h, target_w), device=storage_device, dtype=torch.float32)
            count = torch.zeros((1, 1, 1, target_h, target_w), device=storage_device, dtype=torch.float32)

        if encode:
            ys, ye = y_idx // sf_s, (y_idx // sf_s) + tile_out.shape[3]
            xs, xe = x_idx // sf_s, (x_idx // sf_s) + tile_out.shape[4]
            cur_ov_h = max(0, min(blend_ov_h, tile_out.shape[3] // 2))
            cur_ov_w = max(0, min(blend_ov_w, tile_out.shape[4] // 2))
        else:
            ys, ye = y_idx * sf_s, (y_idx * sf_s) + tile_out.shape[3]
            xs, xe = x_idx * sf_s, (x_idx * sf_s) + tile_out.shape[4]
            cur_ov_h = max(0, min(blend_ov_h, tile_out.shape[3] // 2))
            cur_ov_w = max(0, min(blend_ov_w, tile_out.shape[4] // 2))

        w_h = torch.ones((tile_out.shape[3],), device=storage_device)
        w_w = torch.ones((tile_out.shape[4],), device=storage_device)

        if cur_ov_h > 0:
            r = get_ramp(cur_ov_h)
            if y_idx > 0:
                w_h[:cur_ov_h] = r
            if y_end < h:
                w_h[-cur_ov_h:] = 1.0 - r

        if cur_ov_w > 0:
            r = get_ramp(cur_ov_w)
            if x_idx > 0:
                w_w[:cur_ov_w] = r
            if x_end < w:
                w_w[-cur_ov_w:] = 1.0 - r

        final_weight = w_h.view(1,1,1,-1,1) * w_w.view(1,1,1,1,-1)

        valid_d = min(tile_out.shape[2], result.shape[2])
        tile_out = tile_out[:, :, :valid_d, :, :]

        tile_out.mul_(final_weight)

        result[:, :, :valid_d, ys:ye, xs:xe] += tile_out
        count[:, :, :, ys:ye, xs:xe] += final_weight

        del tile_out, final_weight, w_h, w_w
        bar.update(1)

    result.div_(count.clamp(min=1e-6))
    _seedvr2_clear_temporal_memory(vae_model)

    if result.device != x.device:
        result = result.to(x.device).to(x.dtype)

    if x.shape[2] == 1 and sf_t == 1:
        result = result.squeeze(2)

    return result

_NORM_LIMIT = float("inf")
def get_norm_limit():
    return _NORM_LIMIT


def set_norm_limit(value: Optional[float] = None):
    global _NORM_LIMIT
    if value is None:
        value = float("inf")
    _NORM_LIMIT = value

@contextmanager
def ignore_padding(model):
    orig_padding = model.padding
    model.padding = (0, 0, 0)
    try:
        yield
    finally:
        model.padding = orig_padding

class MemoryState(Enum):
    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2
    UNSET = 3

def get_cache_size(conv_module, input_len, pad_len, dim=0):
    dilated_kernerl_size = conv_module.dilation[dim] * (conv_module.kernel_size[dim] - 1) + 1
    output_len = (input_len + pad_len - dilated_kernerl_size) // conv_module.stride[dim] + 1
    remain_len = (
        input_len + pad_len - ((output_len - 1) * conv_module.stride[dim] + dilated_kernerl_size)
    )
    overlap_len = dilated_kernerl_size - conv_module.stride[dim]
    cache_len = overlap_len + remain_len  # >= 0

    assert output_len > 0
    return cache_len

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, BYTEDANCE_LOGVAR_CLAMP_MIN, BYTEDANCE_LOGVAR_CLAMP_MAX)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        sample = torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def mode(self):
        return self.mean

class SpatialNorm(nn.Module):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = ops.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = ops.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = ops.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f

# partial implementation of diffusers's Attention for comfyui
class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        is_causal: bool = False,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if norm_num_groups is not None:
            self.group_norm = ops.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        self.norm_q = None
        self.norm_k = None

        self.norm_cross = None
        self.to_q = ops.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = ops.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = ops.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = ops.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = ops.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = ops.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(ops.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = ops.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        self.norm_added_q = None
        self.norm_added_k = None
        self.optimized_vae_attention = vae_attention()

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        if self.spatial_norm is not None:
            hidden_states = self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif self.norm_cross:
            encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if input_ndim == 4 and encoder_hidden_states is hidden_states and attention_mask is None and self.heads == 1:
            query = query.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            key = key.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            value = value.squeeze(1).transpose(1, 2).reshape(batch_size, head_dim, height, width)
            hidden_states = self.optimized_vae_attention(query, key, value).reshape(batch_size, self.heads, head_dim, height * width).transpose(2, 3)
        else:
            hidden_states = optimized_attention(query, key, value, heads = self.heads, mask = attention_mask, skip_reshape=True, skip_output_reshape=True)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / self.rescale_output_factor

        return hidden_states


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor):
    with torch.no_grad():
        depth = weight_3d.size(2)
        weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
    return weight_3d

def inflate_bias(bias_2d: torch.Tensor, bias_3d: torch.Tensor):
    with torch.no_grad():
        bias_3d.copy_(bias_2d)
    return bias_3d


def modify_state_dict(layer, state_dict, prefix, inflate_weight_fn, inflate_bias_fn):
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            weight_3d = inflate_weight_fn(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        if bias_2d.dim() == 1:
            bias_3d = inflate_bias_fn(
                bias_2d=bias_2d,
                bias_3d=layer.bias,
            )
            state_dict[bias_name] = bias_3d
    return state_dict

def causal_norm_wrapper(norm_layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    input_dtype = x.dtype
    if isinstance(norm_layer, (ops.LayerNorm, ops.RMSNorm)):
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
    if isinstance(norm_layer, (ops.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
        if x.ndim <= 4:
            return norm_layer(x).to(input_dtype)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            memory_occupy = x.numel() * x.element_size() / 1024**3
            if isinstance(norm_layer, ops.GroupNorm) and memory_occupy > get_norm_limit():
                num_chunks = min(BYTEDANCE_GN_CHUNKS_FP16 if x.element_size() == 2 else BYTEDANCE_GN_CHUNKS_FP32, norm_layer.num_groups)
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

def extend_head(tensor, times: int = 2, memory = None):
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat(tensors=(torch.tile(tensor[:, :, :1], tile_repeat), tensor), dim=2)

def cache_send_recv(tensor, cache_size, times, memory=None):
    recv_buffer = None

    if memory is not None:
        recv_buffer = memory.to(tensor[0])
    elif times > 0:
        tile_repeat = [1] * tensor[0].ndim
        tile_repeat[2] = times
        recv_buffer = torch.tile(tensor[0][:, :, :1], tile_repeat)

    return recv_buffer

class InflatedCausalConv3d(ops.Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode,
        memory_device = "same",
        **kwargs,
    ):
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])
        self.memory_limit = float("inf")
        self.logged_once = False

    def set_memory_limit(self, value: float):
        self.memory_limit = value

    def set_memory_device(self, memory_device):
        self.memory_device = memory_device

    def _conv_forward(self, input, weight, bias, *args, **kwargs):
        if (NVIDIA_MEMORY_CONV_BUG_WORKAROUND and
            weight.dtype in (torch.float16, torch.bfloat16) and
            hasattr(torch.backends.cudnn, 'is_available') and
            torch.backends.cudnn.is_available() and
            getattr(torch.backends.cudnn, 'enabled', True)):
            try:
                out = torch.cudnn_convolution(
                    input, weight, self.padding, self.stride, self.dilation, self.groups,
                    benchmark=False, deterministic=False, allow_tf32=True
                )
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            except RuntimeError:
                pass
            except NotImplementedError:
                pass
        try:
            return super()._conv_forward(input, weight, bias, *args, **kwargs)
        except NotImplementedError:
            # for: Could not run 'aten::cudnn_convolution' with arguments from the 'CPU' backend
            if not self.logged_once:
                logging.warning("VAE is on CPU for decoding. This is most likely due to not enough memory")
                self.logged_once = True
            return F.conv3d(input, weight, bias, *args, **kwargs)

    def memory_limit_conv(
        self,
        x,
        *,
        split_dim=3,
        padding=(0, 0, 0, 0, 0, 0),
        prev_cache=None,
    ):
        # Compatible with no limit.
        if math.isinf(self.memory_limit):
            if prev_cache is not None:
                x = torch.cat([prev_cache, x], dim=split_dim - 1)
            return super().forward(x)

        # Compute tensor shape after concat & padding.
        shape = torch.tensor(x.size())
        if prev_cache is not None:
            shape[split_dim - 1] += prev_cache.size(split_dim - 1)
        shape[-3:] += torch.tensor(padding).view(3, 2).sum(-1).flip(0)
        memory_occupy = shape.prod() * x.element_size() / 1024**3  # GiB
        if memory_occupy < self.memory_limit or split_dim == x.ndim:
            x_concat = x
            if prev_cache is not None:
                x_concat = torch.cat([prev_cache, x], dim=split_dim - 1)

            def pad_and_forward():
                padded = safe_pad_operation(x_concat, padding, mode='constant', value=0.0)
                if not padded.is_contiguous():
                    padded = padded.contiguous()
                with ignore_padding(self):
                    return torch.nn.Conv3d.forward(self, padded)

            return pad_and_forward()

        num_splits = math.ceil(memory_occupy / self.memory_limit)
        size_per_split = x.size(split_dim) // num_splits
        split_sizes = [size_per_split] * (num_splits - 1)
        split_sizes += [x.size(split_dim) - sum(split_sizes)]

        x = list(x.split(split_sizes, dim=split_dim))
        if prev_cache is not None:
            prev_cache = list(prev_cache.split(split_sizes, dim=split_dim))
        cache = None
        for idx in range(len(x)):
            if prev_cache is not None:
                x[idx] = torch.cat([prev_cache[idx], x[idx]], dim=split_dim - 1)

            lpad_dim = (x[idx].ndim - split_dim - 1) * 2
            rpad_dim = lpad_dim + 1
            padding = list(padding)
            padding[lpad_dim] = self.padding[split_dim - 2] if idx == 0 else 0
            padding[rpad_dim] = self.padding[split_dim - 2] if idx == len(x) - 1 else 0
            pad_len = padding[lpad_dim] + padding[rpad_dim]
            padding = tuple(padding)

            next_cache = None
            cache_len = cache.size(split_dim) if cache is not None else 0
            next_catch_size = get_cache_size(
                conv_module=self,
                input_len=x[idx].size(split_dim) + cache_len,
                pad_len=pad_len,
                dim=split_dim - 2,
            )
            if next_catch_size != 0:
                assert next_catch_size <= x[idx].size(split_dim)
                next_cache = (
                    x[idx].transpose(0, split_dim)[-next_catch_size:].transpose(0, split_dim)
                )

            x[idx] = self.memory_limit_conv(
                x[idx],
                split_dim=split_dim + 1,
                padding=padding,
                prev_cache=cache
            )

            cache = next_cache

        output = torch.cat(x, dim=split_dim)
        return output

    def forward(
        self,
        input,
        memory_state: MemoryState = MemoryState.UNSET
    ) -> Tensor:
        assert memory_state != MemoryState.UNSET
        if memory_state != MemoryState.ACTIVE:
            self.memory = None
        if (
            math.isinf(self.memory_limit)
            and torch.is_tensor(input)
        ):
            return self.basic_forward(input, memory_state)
        return self.slicing_forward(input, memory_state)

    def basic_forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET):
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")
        return super().forward(input)

    def slicing_forward(
        self,
        input,
        memory_state: MemoryState = MemoryState.UNSET,
    ) -> Tensor:
        squeeze_out = False
        if torch.is_tensor(input):
            input = [input]
            squeeze_out = True

        cache_size = self.kernel_size[0] - self.stride[0]
        cache = cache_send_recv(
            input, cache_size=cache_size, memory=self.memory, times=self.temporal_padding * 2
        )

        # Single GPU inference - simplified memory management
        if (
            memory_state in [MemoryState.INITIALIZING, MemoryState.ACTIVE]  # use_slicing
            and not self.training
            and (self.memory_device is not None)
            and cache_size != 0
        ):
            if cache_size > input[-1].size(2) and cache is not None and len(input) == 1:
                input[0] = torch.cat([cache, input[0]], dim=2)
                cache = None
            if cache_size <= input[-1].size(2):
                self.memory = input[-1][:, :, -cache_size:].detach().contiguous()
                if self.memory_device == "cpu" and self.memory is not None:
                    self.memory = self.memory.to("cpu")

        padding = tuple(x for x in reversed(self.padding) for _ in range(2))
        for i in range(len(input)):
            # Prepare cache for next input slice.
            next_cache = None
            cache_size = 0
            if i < len(input) - 1:
                cache_len = cache.size(2) if cache is not None else 0
                cache_size = get_cache_size(self, input[i].size(2) + cache_len, pad_len=0)
            if cache_size != 0:
                if cache_size > input[i].size(2) and cache is not None:
                    input[i] = torch.cat([cache, input[i]], dim=2)
                    cache = None
                assert cache_size <= input[i].size(2), f"{cache_size} > {input[i].size(2)}"
                next_cache = input[i][:, :, -cache_size:]

            # Conv forward for this input slice.
            input[i] = self.memory_limit_conv(
                input[i],
                padding=padding,
                prev_cache=cache
            )

            # Update cache.
            cache = next_cache

        return input[0] if squeeze_out else input

def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)

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
            self.conv = ops.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = ops.Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

        conv = self.conv if self.name == "conv" else self.Conv2d_0

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
            self.upscale_conv = ops.Conv3d(
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

        self.norm = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state=None,
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

        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states, memory_state=memory_state)
            else:
                hidden_states = self.Conv2d_0(hidden_states, memory_state=memory_state)

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
        kernel_size=3,
        use_conv: bool = False,
        padding = 1,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        self.padding = padding
        self.name = name
        self.channels = channels
        self.out_channels = out_channels or channels
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down
        self.use_conv = use_conv
        self.padding = padding

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if use_conv:
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
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )

        self.conv = conv


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state = None,
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

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

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
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.use_in_shortcut = use_in_shortcut
        self.output_scale_factor = output_scale_factor
        self.skip_time_act = skip_time_act
        self.nonlinearity = nn.SiLU()
        if temb_channels is not None:
            self.time_emb_proj = ops.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
        self.norm1 = ops.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        if groups_out is None:
            groups_out = groups
        self.norm2 = ops.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.use_in_shortcut = self.in_channels != out_channels
        self.dropout = torch.nn.Dropout(dropout)
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
            conv_2d_out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.upsample = self.downsample = None
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

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedCausalConv3d(
                self.in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                inflation_mode=inflation_mode,
            )

    def forward(
        self, input_tensor, temb, memory_state = None, **kwargs
    ):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= BYTEDANCE_CONTIGUOUS_BATCH_THRESHOLD:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor, memory_state=memory_state)
            hidden_states = self.upsample(hidden_states, memory_state=memory_state)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, memory_state=memory_state)
            hidden_states = self.downsample(hidden_states, memory_state=memory_state)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None:
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

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
        memory_state = None,
        **kwargs,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

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
        memory_state=None
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state)

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

    def forward(self, hidden_states, temb=None, memory_state=None):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state)

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
                    ops.Conv3d(extra_cond_dim, output_channel, kernel_size=1, stride=1, padding=0)
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
        self.conv_norm_out = ops.GroupNorm(
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
        memory_state = None
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = sample.to(next(self.parameters()).device)
        sample = self.conv_in(sample, memory_state = memory_state)
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
                sample = down_block(sample, memory_state=memory_state)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state = memory_state)

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
            self.conv_norm_out = ops.GroupNorm(
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
        memory_state = None,
    ) -> torch.FloatTensor:

        sample = sample.to(next(self.parameters()).device)
        sample = self.conv_in(sample, memory_state=memory_state)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample, latent_embeds, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample

class VideoAutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = SEEDVR2_LATENT_CHANNELS,
        norm_num_groups: int = 32,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode = "pad",
        time_receptive_field: _receptive_field_t = "full",
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        slicing_sample_min_size = BYTEDANCE_SLICING_SAMPLE_MIN,
        *args,
        **kwargs,
    ):
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (2**temporal_scale_num)
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        block_out_channels = BYTEDANCE_BLOCK_OUT_CHANNELS
        down_block_types = ("DownEncoderBlock3D",) * 4
        up_block_types = ("UpDecoderBlock3D",) * 4
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

        self.use_slicing = True

    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        h = self.slicing_encode(x)
        posterior = DiagonalGaussianDistribution(h).mode()

        if not return_dict:
            return (posterior,)

        return posterior

    def decode_(
        self, z: torch.Tensor, return_dict: bool = True
    ):
        decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return decoded

    def _encode(
        self, x, memory_state = MemoryState.DISABLED
    ) -> torch.Tensor:
        _x = x.to(self.device)
        h = self.encoder(_x, memory_state=memory_state)
        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h
        return output.to(x.device)

    def _decode(
        self, z, memory_state = MemoryState.DISABLED
    ) -> torch.Tensor:
        _z = z.to(self.device)

        if self.post_quant_conv is not None:
            _z = self.post_quant_conv(_z, memory_state=memory_state)

        output = self.decoder(_z, memory_state=memory_state)
        return output.to(z.device)

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size =1
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            split_size = max(
                self.slicing_sample_min_size * sp_size,
                getattr(self, "temporal_downsample_factor", 1),
            )
            x_slices = list(x[:, :, 1:].split(split_size=split_size, dim=2))
            min_active_len = getattr(self, "temporal_downsample_factor", 1)
            if len(x_slices) > 1 and x_slices[-1].shape[2] < min_active_len:
                x_slices[-2] = torch.cat((x_slices[-2], x_slices[-1]), dim=2)
                x_slices.pop()
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(encoded_slices, dim=2)
            modules_with_memory = [m for m in self.modules()
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._encode(x)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = 1
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING
                )
            ]
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
            out = torch.cat(decoded_slices, dim=2)
            modules_with_memory = [m for m in self.modules()
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            return self._decode(z)

    def tiled_encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def tiled_decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        def _unwrap(value):
            return value[0] if isinstance(value, tuple) else value

        if mode == "encode":
            return _unwrap(self.encode(x))
        elif mode == "decode":
            return _unwrap(self.decode_(x))
        else:
            latent = _unwrap(self.encode(x))
            return _unwrap(self.decode_(latent))

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
        self.enable_tiling = False
        super().__init__(*args, **kwargs)
        self.set_memory_limit(BYTEDANCE_VAE_CONV_MEM_GIB, BYTEDANCE_VAE_NORM_MEM_GIB)

    def forward(self, x: torch.FloatTensor):
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z)
        return x, z, p

    def encode(self, x, orig_dims=None):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = x.to(dtype=next(self.parameters()).dtype)
        self.device = x.device
        p = super().encode(x)
        z = p.squeeze(2)
        return z, p

    def decode(self, z, seedvr2_tiling=None):
        seedvr2_tiling = {} if seedvr2_tiling is None else seedvr2_tiling
        if not isinstance(seedvr2_tiling, dict):
            raise RuntimeError(
                "SeedVR2 VideoAutoencoderKLWrapper.decode: `seedvr2_tiling` must be a dict; "
                f"got {type(seedvr2_tiling).__name__} with value {seedvr2_tiling!r}."
            )

        if z.ndim == 5:
            b, c, t_latent, h, w = z.shape
            if c != 16:
                raise RuntimeError(
                    "SeedVR2 VideoAutoencoderKLWrapper.decode: 5-D latent input must "
                    f"have 16 channels; got shape {tuple(z.shape)}."
                )
            latent = z
        elif z.ndim == 4:
            b, tc, h, w = z.shape
            if tc % 16 != 0:
                raise RuntimeError(
                    "SeedVR2 VideoAutoencoderKLWrapper.decode: 4-D latent input must "
                    "use collapsed channel layout (B, 16*T, H, W); "
                    f"got shape {tuple(z.shape)}."
                )
            latent = z.reshape(b, 16, -1, h, w)
        else:
            raise RuntimeError(
                "SeedVR2 VideoAutoencoderKLWrapper.decode: latent input must be "
                "4-D collapsed (B, 16*T, H, W) or 5-D (B, 16, T, H, W); "
                f"got shape {tuple(z.shape)}."
            )
        scale = BYTEDANCE_VAE_SCALING_FACTOR
        shift = BYTEDANCE_VAE_SHIFTING_FACTOR
        latent = latent / scale + shift

        self.device = latent.device
        self.enable_tiling = seedvr2_tiling.get("enable_tiling", False)

        if self.enable_tiling:
            decode_seedvr2_args = dict(seedvr2_tiling)
            tile_h, tile_w = decode_seedvr2_args.get("tile_size", (512, 512))
            ov_h, ov_w = decode_seedvr2_args.get("tile_overlap", (64, 64))
            decode_seedvr2_args["tile_overlap"] = (
                min(ov_h, max(0, tile_h - 8)),
                min(ov_w, max(0, tile_w - 8)),
            )
            x = tiled_vae(latent, self, **decode_seedvr2_args, encode=False)
            if x.ndim == 4:
                # tiled_vae squeezes the temporal axis when
                # temporal_downsample_factor == 1 AND latent T == 1
                # (see tiled_vae line 179-180); re-add it so the post-decode
                # pipeline can keep batch and time distinct on the tiled path.
                x = x.unsqueeze(2)
        else:
            x = super().decode_(latent)

        # ensure even dims for save video
        h, w = x.shape[-2:]
        w2 = w - (w % 2)
        h2 = h - (h % 2)
        x = x[..., :h2, :w2]

        return x

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float], memory_device = "same"):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))

        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

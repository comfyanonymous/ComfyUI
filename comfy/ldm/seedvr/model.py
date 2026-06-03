from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
import einops
from einops import rearrange
import torch.nn.functional as F
from math import ceil, pi
import torch
from itertools import chain
from comfy.ldm.modules.diffusionmodules.model import get_timestep_embedding
from comfy.ldm.modules.attention import optimized_var_attention
from torch.nn.modules.utils import _triple
from torch import nn
import math
from comfy.ldm.flux.math import apply_rope1
from comfy.ldm.seedvr.constants import (
    BYTEDANCE_720P_REF_AREA,
    BYTEDANCE_MAX_TEMPORAL_WINDOW,
    BYTEDANCE_ROPE_MAX_FREQ,
    BYTEDANCE_SINUSOIDAL_DIM,
    ROPE_THETA,
    SEEDVR2_7B_MLP_CHUNK,
    SEEDVR2_7B_VID_DIM,
    SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS,
)
import comfy.model_management
import numbers

def _torch_float8_types():
    return tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
        if hasattr(torch, name)
    )

class CustomRMSNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, device=None, dtype=None):
        super(CustomRMSNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape, device=device, dtype=dtype))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):

        dims = tuple(range(-len(self.normalized_shape), 0))

        normalized = input.float()
        variance = normalized.pow(2).mean(dim=dims, keepdim=True)
        rms = torch.sqrt(variance + self.eps)

        normalized = normalized / rms

        if self.elementwise_affine:
            return normalized * self.weight.to(input.dtype)
        return normalized

class Cache:
    def __init__(self, disable=False, prefix="", cache=None):
        self.cache = cache if cache is not None else {}
        self.disable = disable
        self.prefix = prefix

    def __call__(self, key: str, fn: Callable):
        if self.disable:
            return fn()

        key = self.prefix + key
        try:
            result = self.cache[key]
        except KeyError:
            result = fn()
            self.cache[key] = result
        return result

    def namespace(self, namespace: str):
        return Cache(
            disable=self.disable,
            prefix=self.prefix + namespace + ".",
            cache=self.cache,
        )

    def get(self, key: str):
        key = self.prefix + key
        return self.cache[key]

def repeat_concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: List,  # (n)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    txt = [[x] * n for x, n in zip(txt, txt_repeat)]
    txt = list(chain(*txt))
    return torch.cat(list(chain(*zip(vid, txt))))

def concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> torch.FloatTensor:  # (L ... c)
    vid = torch.split(vid, vid_len.tolist())
    txt = torch.split(txt, txt_len.tolist())
    return torch.cat(list(chain(*zip(vid, txt))))

def concat_idx(
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    tgt_idx = concat(vid_idx, txt_idx, vid_len, txt_len)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda vid, txt: torch.index_select(torch.cat([vid, txt]), 0, tgt_idx),
        lambda all: torch.index_select(all, 0, src_idx).split([len(vid_idx), len(txt_idx)]),
    )


def repeat_concat_idx(
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: torch.LongTensor,  # (n)
) -> Tuple[
    Callable,
    Callable,
]:
    device = vid_len.device
    vid_idx = torch.arange(vid_len.sum(), device=device)
    txt_idx = torch.arange(len(vid_idx), len(vid_idx) + txt_len.sum(), device=device)
    txt_repeat_list = txt_repeat.tolist()
    tgt_idx = repeat_concat(vid_idx, txt_idx, vid_len, txt_len, txt_repeat)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    repeat_txt_len = (txt_len * txt_repeat).tolist()

    def unconcat_coalesce(all):
        vid_out, txt_out = all[src_idx].split([len(vid_idx), txt_idx_len])
        txt_out_coalesced = []
        for txt, repeat_time in zip(txt_out.split(repeat_txt_len), txt_repeat_list):
            txt = txt.reshape(-1, repeat_time, *txt.shape[1:]).mean(1)
            txt_out_coalesced.append(txt)
        return vid_out, torch.cat(txt_out_coalesced)

    return (
        lambda vid, txt: torch.cat([vid, txt])[tgt_idx],
        lambda all: unconcat_coalesce(all),
    )


@dataclass
class MMArg:
    vid: Any
    txt: Any

def safe_pad_operation(x, padding, mode='constant', value=0.0):
    """Safe padding operation that handles Half precision only for problematic modes"""
    # Modes qui nécessitent le fix Half precision
    problematic_modes = ['replicate', 'reflect', 'circular']

    if mode in problematic_modes:
        try:
            return F.pad(x, padding, mode=mode, value=value)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                original_dtype = x.dtype
                return F.pad(x.float(), padding, mode=mode, value=value).to(original_dtype)
            else:
                raise e
    else:
        # Pour 'constant' et autres modes compatibles, pas de fix nécessaire
        return F.pad(x, padding, mode=mode, value=value)


def get_args(key: str, args: List[Any]) -> List[Any]:
    return [getattr(v, key) if isinstance(v, MMArg) else v for v in args]


def get_kwargs(key: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: getattr(v, key) if isinstance(v, MMArg) else v for k, v in kwargs.items()}


def get_window_op(name: str):
    if name == "720pwin_by_size_bysize":
        return make_720Pwindows_bysize
    if name == "720pswin_by_size_bysize":
        return make_shifted_720Pwindows_bysize
    raise ValueError(f"Unknown windowing method: {name}")


# -------------------------------- Windowing -------------------------------- #
def make_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt(BYTEDANCE_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, BYTEDANCE_MAX_TEMPORAL_WINDOW) / resized_nt)  # window size.
    nt, nh, nw = ceil(t / wt), ceil(h / wh), ceil(w / ww)  # window size.
    return [
        (
            slice(it * wt, min((it + 1) * wt, t)),
            slice(ih * wh, min((ih + 1) * wh, h)),
            slice(iw * ww, min((iw + 1) * ww, w)),
        )
        for iw in range(nw)
        if min((iw + 1) * ww, w) > iw * ww
        for ih in range(nh)
        if min((ih + 1) * wh, h) > ih * wh
        for it in range(nt)
        if min((it + 1) * wt, t) > it * wt
    ]

def make_shifted_720Pwindows_bysize(size: Tuple[int, int, int], num_windows: Tuple[int, int, int]):
    t, h, w = size
    resized_nt, resized_nh, resized_nw = num_windows
    #cal windows under 720p
    scale = math.sqrt(BYTEDANCE_720P_REF_AREA / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, BYTEDANCE_MAX_TEMPORAL_WINDOW) / resized_nt)  # window size.

    st, sh, sw = (  # shift size.
        0.5 if wt < t else 0,
        0.5 if wh < h else 0,
        0.5 if ww < w else 0,
    )
    nt, nh, nw = ceil((t - st) / wt), ceil((h - sh) / wh), ceil((w - sw) / ww)  # window size.
    nt, nh, nw = (  # number of window.
        nt + 1 if st > 0 else 1,
        nh + 1 if sh > 0 else 1,
        nw + 1 if sw > 0 else 1,
    )
    return [
        (
            slice(max(int((it - st) * wt), 0), min(int((it - st + 1) * wt), t)),
            slice(max(int((ih - sh) * wh), 0), min(int((ih - sh + 1) * wh), h)),
            slice(max(int((iw - sw) * ww), 0), min(int((iw - sw + 1) * ww), w)),
        )
        for iw in range(nw)
        if min(int((iw - sw + 1) * ww), w) > max(int((iw - sw) * ww), 0)
        for ih in range(nh)
        if min(int((ih - sh + 1) * wh), h) > max(int((ih - sh) * wh), 0)
        for it in range(nt)
        if min(int((it - st + 1) * wt), t) > max(int((it - st) * wt), 0)
    ]

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
        learned_freq = False,
        use_xpos = False,
        xpos_scale_base = 512,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
        seq_before_head_dim = False,
        cache_if_possible = True,
        cache_max_seq_len = 8192
    ):
        super().__init__()

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent = False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent = False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_axial_freqs(
        self,
        *dims,
        offsets = None
    ):
        Colon = slice(None)
        all_freqs = []

        # handle offset

        if exists(offsets):
            assert len(offsets) == len(dims)

        for ind, dim in enumerate(dims):

            offset = 0
            if exists(offsets):
                offset = offsets[ind]

            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps = dim, device = self.device)
            else:
                pos = torch.arange(dim, device = self.device)

            pos = pos + offset

            freqs = self.forward(pos, seq_len = dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        # concat all freqs

        all_freqs = torch.broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim = -1)

    def forward(
        self,
        t,
        seq_len: int | None = None,
        offset = 0
    ):
        should_cache = (
            self.cache_if_possible and
            not self.learned_freq and
            exists(seq_len) and
            self.freqs_for != 'pixel' and
            (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = einops.repeat(freqs, '... n -> ... (n r)', r = 2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs

class RotaryEmbeddingBase(nn.Module):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__()
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="pixel",
            max_freq=BYTEDANCE_ROPE_MAX_FREQ,
        )
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)

    def get_axial_freqs(self, *dims):
        return self.rope.get_axial_freqs(*dims)


class RotaryEmbedding3d(RotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)
        self.mm = False

    def forward(
        self,
        q: torch.FloatTensor,  # b h l d
        k: torch.FloatTensor,  # b h l d
        size: Tuple[int, int, int],
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        T, H, W = size
        freqs = self.get_axial_freqs(T, H, W)
        q = rearrange(q, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        k = rearrange(k, "b h (T H W) d -> b h T H W d", T=T, H=H, W=W)
        q = apply_rotary_emb(freqs, q.float()).to(q.dtype)
        k = apply_rotary_emb(freqs, k.float()).to(k.dtype)
        q = rearrange(q, "b h T H W d -> b h (T H W) d")
        k = rearrange(k, "b h T H W d -> b h (T H W) d")
        return q, k


class NaRotaryEmbedding3d(RotaryEmbedding3d):
    def forward(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        shape: torch.LongTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        freqs = cache("rope_freqs_3d", lambda: self.get_freqs(shape))
        freqs = freqs.to(device=q.device)
        q = rearrange(q, "L h d -> h L d")
        k = rearrange(k, "L h d -> h L d")
        q = _apply_seedvr2_rotary_emb(freqs, q.float()).to(q.dtype)
        k = _apply_seedvr2_rotary_emb(freqs, k.float()).to(k.dtype)
        q = rearrange(q, "h L d -> L h d")
        k = rearrange(k, "h L d -> L h d")
        return q, k

    @torch._dynamo.disable
    def get_freqs(
        self,
        shape: torch.LongTensor,
    ) -> torch.Tensor:
        # Primary provenance: ByteDance-Seed/SeedVR models/dit/rope.py builds
        # 7B pixel RoPE with the interleaved-angle convention, not Comfy's
        # Flux freqs_cis matrix.
        plain_rope = RotaryEmbedding(
            dim=self.rope.freqs.numel() * 2,
            freqs_for="pixel",
            max_freq=BYTEDANCE_ROPE_MAX_FREQ,
        )
        plain_rope = plain_rope.to(self.rope.dummy.device)
        freq_list = []
        for f, h, w in shape.tolist():
            freqs = plain_rope.get_axial_freqs(f, h, w)
            freq_list.append(freqs.view(-1, freqs.size(-1)))
        return torch.cat(freq_list, dim=0)


class MMRotaryEmbeddingBase(RotaryEmbeddingBase):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__(dim, rope_dim)
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=ROPE_THETA,
            cache_if_possible=False,
        )
        freqs = self.rope.freqs
        del self.rope.freqs
        self.rope.register_buffer("freqs", freqs.data)
        self.mm = True

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')
def exists(val):
    return val is not None

def apply_rotary_emb(
    freqs,
    t,
    start_index = 0,
    scale = 1.,
    seq_dim = -2,
    freqs_seq_dim = None
):
    dtype = t.dtype
    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim = freqs_seq_dim)

    rot_feats = freqs.shape[-1]
    end_index = start_index + rot_feats

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    angles = freqs.to(t_middle.device)[..., ::2]
    cos = torch.cos(angles) * scale
    sin = torch.sin(angles) * scale

    col0 = torch.stack([cos, sin], dim=-1)
    col1 = torch.stack([-sin, cos], dim=-1)
    freqs_mat = torch.stack([col0, col1], dim=-1)

    t_middle_out = apply_rope1(t_middle, freqs_mat)
    out = torch.cat((t_left, t_middle_out, t_right), dim=-1)
    return out.type(dtype)


def _apply_seedvr2_rotary_emb(
    freqs: torch.Tensor,
    t: torch.Tensor,
    start_index: int = 0,
    scale: float = 1.0,
    seq_dim: int = -2,
    freqs_seq_dim: int | None = None,
) -> torch.Tensor:
    dtype = t.dtype
    if freqs_seq_dim is None and (freqs.ndim == 2 or t.ndim == 3):
        freqs_seq_dim = 0

    if t.ndim == 3 or freqs_seq_dim is not None:
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_feats = freqs.shape[-1]
    end_index = start_index + rot_feats

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    freqs = freqs.to(device=t_middle.device, dtype=t_middle.dtype)
    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    t_middle = (t_middle * cos) + (rotate_half(t_middle) * sin)
    return torch.cat((t_left, t_middle, t_right), dim=-1).to(dtype)

def _to_flux_freqs_cis(freqs_interleaved: torch.Tensor) -> torch.Tensor:
    """Convert lucidrains-interleaved freqs to flux-canonical fp32 freqs_cis `[..., d/2, 2, 2]` (cos/-sin/sin/cos), per `comfy/ldm/flux/math.py:rope`."""
    angles = freqs_interleaved[..., ::2].float()
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    out = torch.stack([cos, -sin, sin, cos], dim=-1)
    return rearrange(out, "... d (i j) -> ... d i j", i=2, j=2)


def _apply_rope1_partial(t: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``rot_d = 2 * freqs_cis.shape[-3]`` dims of ``t`` and pass the rest
    through; in-place for inference, cloned for training (autograd). Mirrors the legacy
    ``apply_rotary_emb`` ``t_left``/``t_middle``/``t_right`` split: 3B ``rope_dim=128`` gives
    ``42*3 = 126`` rotated of head_dim 128 (trailing 2 unrotated). Fast path skips the cat when
    ``rot_d == t.shape[-1]``.
    """
    out = t.clone() if t.requires_grad or comfy.model_management.in_training else t
    rot_d = 2 * freqs_cis.shape[-3]
    seq_len = out.shape[-2]
    for start in range(0, seq_len, SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS):
        end = min(start + SEEDVR2_ROPE_PARTIAL_CHUNK_TOKENS, seq_len)
        freqs_chunk = freqs_cis[start:end]
        if rot_d == out.shape[-1]:
            out[..., start:end, :] = apply_rope1(out[..., start:end, :], freqs_chunk).to(out.dtype)
        else:
            out[..., start:end, :rot_d] = apply_rope1(out[..., start:end, :rot_d], freqs_chunk).to(out.dtype)
    return out


class NaMMRotaryEmbedding3d(MMRotaryEmbeddingBase):
    def __init__(self, dim: int):
        super().__init__(dim, rope_dim=3)

    def forward(
        self,
        vid_q: torch.FloatTensor,  # L h d
        vid_k: torch.FloatTensor,  # L h d
        vid_shape: torch.LongTensor,  # B 3
        txt_q: torch.FloatTensor,  # L h d
        txt_k: torch.FloatTensor,  # L h d
        txt_shape: torch.LongTensor,  # B 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_freqs, txt_freqs = cache(
            "mmrope_freqs_3d",
            lambda: self.get_freqs(vid_shape, txt_shape),
        )
        target_device = vid_q.device
        if vid_freqs.device != target_device:
            vid_freqs = vid_freqs.to(target_device)
        if txt_freqs.device != target_device:
            txt_freqs = txt_freqs.to(target_device)
        vid_q = rearrange(vid_q, "L h d -> h L d")
        vid_k = rearrange(vid_k, "L h d -> h L d")
        vid_q = _apply_rope1_partial(vid_q, vid_freqs)
        vid_k = _apply_rope1_partial(vid_k, vid_freqs)
        vid_q = rearrange(vid_q, "h L d -> L h d")
        vid_k = rearrange(vid_k, "h L d -> L h d")

        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = _apply_rope1_partial(txt_q, txt_freqs)
        txt_k = _apply_rope1_partial(txt_k, txt_freqs)
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return vid_q, vid_k, txt_q, txt_k

    @torch._dynamo.disable  # Disable compilation: .tolist() is data-dependent and causes graph breaks
    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # Calculate actual max dimensions needed for this batch
        max_temporal = 0
        max_height = 0
        max_width = 0
        max_txt_len = 0

        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            max_temporal = max(max_temporal, l + f)  # Need up to l+f for temporal
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            max_txt_len = max(max_txt_len, l)

        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(autocast_device, enabled=False):
            vid_freqs = self.get_axial_freqs(
                max_temporal + 16,
                max_height + 4,
                max_width + 4,
            ).float()
            txt_freqs = self.get_axial_freqs(max_txt_len + 16)

        # Now slice as before
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1))
            txt_freq = txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        vid_freqs_interleaved = torch.cat(vid_freq_list, dim=0)
        txt_freqs_interleaved = torch.cat(txt_freq_list, dim=0)

        # Convert from lucidrains-interleaved layout `[θ0, θ0, θ1, θ1, ...]`
        # (produced by `repeat(freqs, '... n -> ... (n r)', r=2)` in the
        # upstream `RotaryEmbedding.forward`) to flux-canonical `freqs_cis`
        # in shape `[..., d/2, 2, 2]` with `cos/-sin/sin/cos` baked in.
        # Mirrors `comfy/ldm/flux/math.py:rope` (line 27) so the trailing
        # 2x2 is the per-frequency rotation matrix that
        # `comfy.ldm.flux.math.apply_rope1` expects.
        return _to_flux_freqs_cis(vid_freqs_interleaved), _to_flux_freqs_cis(txt_freqs_interleaved)

class MMModule(nn.Module):
    def __init__(
        self,
        module: Callable[..., nn.Module],
        *args,
        shared_weights: bool = False,
        vid_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.vid_only = vid_only
        if self.shared_weights:
            assert get_args("vid", args) == get_args("txt", args)
            assert get_kwargs("vid", kwargs) == get_kwargs("txt", kwargs)
            self.all = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
        else:
            self.vid = module(*get_args("vid", args), **get_kwargs("vid", kwargs))
            self.txt = (
                module(*get_args("txt", args), **get_kwargs("txt", kwargs))
                if not vid_only
                else None
            )

    def forward(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_module = self.vid if not self.shared_weights else self.all
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            txt = txt.to(device=vid.device, dtype=vid.dtype)
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        return vid, txt

def get_na_rope(rope_type: Optional[str], dim: int):
    if rope_type is None:
        return None
    if rope_type == "rope3d":
        return NaRotaryEmbedding3d(dim=dim)
    if rope_type == "mmrope3d":
        return NaMMRotaryEmbedding3d(dim=dim)

class NaMMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int,
        head_dim: int,
        qk_bias: bool,
        qk_norm,
        qk_norm_eps: float,
        rope_type: Optional[str],
        rope_dim: int,
        shared_weights: bool,
        device, dtype, operations,
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.heads = heads
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            operations.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights, device=device, dtype=dtype
        )
        self.proj_out = MMModule(operations.Linear, inner_dim, dim, shared_weights=shared_weights, device=device, dtype=dtype)
        self.norm_q = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
            device=device, dtype=dtype
        )
        self.norm_k = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
            device=device, dtype=dtype
        )


        self.rope = get_na_rope(rope_type=rope_type, dim=rope_dim)

    def forward(self):
        pass

def window(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid = unflatten(hid, hid_shape)
    hid = list(map(window_fn, hid))
    hid_windows = torch.tensor(list(map(len, hid)), device=hid_shape.device)
    hid, hid_shape = flatten(list(chain(*hid)))
    return hid, hid_shape, hid_windows

def window_idx(
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows = window(hid_idx, hid_shape, window_fn)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
        tgt_windows,
    )

class NaSwinAttention(NaMMAttention):
    def __init__(
        self,
        *args,
        window: Union[int, Tuple[int, int, int]],
        window_method: bool, # shifted or not
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.version_7b = kwargs.get("version", False)
        self.window = _triple(window)
        self.window_method = window_method
        assert all(map(lambda v: isinstance(v, int) and v >= 0, self.window))

        self.window_op = get_window_op(window_method)

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:

        vid_qkv, txt_qkv = self.proj_qkv(vid, txt)

        # re-org the input seq for window attn
        cache_win = cache.namespace(f"{self.window_method}_{self.window}_sd3")

        def make_window(x: torch.Tensor):
            t, h, w, _ = x.shape
            window_slices = self.window_op((t, h, w), self.window)
            return [x[st, sh, sw] for (st, sh, sw) in window_slices]

        window_partition, window_reverse, window_shape, window_count = cache_win(
            "win_transform",
            lambda: window_idx(vid_shape, make_window),
        )
        vid_qkv_win = window_partition(vid_qkv)

        vid_qkv_win = rearrange(vid_qkv_win, "l (o h d) -> l o h d", o=3, d=self.head_dim)
        txt_qkv = rearrange(txt_qkv, "l (o h d) -> l o h d", o=3, d=self.head_dim)

        vid_q, vid_k, vid_v = vid_qkv_win.unbind(1)
        txt_q, txt_k, txt_v = txt_qkv.unbind(1)

        vid_q, txt_q = self.norm_q(vid_q, txt_q)
        vid_k, txt_k = self.norm_k(vid_k, txt_k)

        txt_len = cache("txt_len", lambda: txt_shape.prod(-1))

        vid_len_win = cache_win("vid_len", lambda: window_shape.prod(-1))
        txt_len = txt_len.to(window_count.device)

        # window rope
        if self.rope:
            if self.version_7b:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)
            elif self.rope.mm:
                # repeat text q and k for window mmrope
                _, num_h, _ = txt_q.shape
                txt_q_repeat = rearrange(txt_q, "l h d -> l (h d)")
                txt_q_repeat = unflatten(txt_q_repeat, txt_shape)
                txt_q_repeat = [[x] * n for x, n in zip(txt_q_repeat, window_count)]
                txt_q_repeat = list(chain(*txt_q_repeat))
                txt_q_repeat, txt_shape_repeat = flatten(txt_q_repeat)
                txt_q_repeat = rearrange(txt_q_repeat, "l (h d) -> l h d", h=num_h)

                txt_k_repeat = rearrange(txt_k, "l h d -> l (h d)")
                txt_k_repeat = unflatten(txt_k_repeat, txt_shape)
                txt_k_repeat = [[x] * n for x, n in zip(txt_k_repeat, window_count)]
                txt_k_repeat = list(chain(*txt_k_repeat))
                txt_k_repeat, _ = flatten(txt_k_repeat)
                txt_k_repeat = rearrange(txt_k_repeat, "l (h d) -> l h d", h=num_h)

                vid_q, vid_k, txt_q, txt_k = self.rope(
                    vid_q, vid_k, window_shape, txt_q_repeat, txt_k_repeat, txt_shape_repeat, cache_win
                )
            else:
                vid_q, vid_k = self.rope(vid_q, vid_k, window_shape, cache_win)

        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: repeat_concat_idx(vid_len_win, txt_len, window_count)
        )
        out = optimized_var_attention(
            q=concat_win(vid_q, txt_q),
            k=concat_win(vid_k, txt_k),
            v=concat_win(vid_v, txt_v),
            heads=self.heads, skip_reshape=True, skip_output_reshape=True,
            cu_seqlens_q=cache_win(
                "vid_seqlens_q", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
            cu_seqlens_k=cache_win(
                "vid_seqlens_k", lambda: safe_pad_operation(all_len_win.cumsum(0), (1, 0)).int()
            ),
        )
        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out

class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        device, dtype, operations
    ):
        super().__init__()
        self.proj_in = operations.Linear(dim, dim * expand_ratio, device=device, dtype=dtype)
        self.act = nn.GELU("tanh")
        self.proj_out = operations.Linear(dim * expand_ratio, dim, device=device, dtype=dtype)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.proj_in(x)
        x = self.act(x)
        x = self.proj_out(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
        multiple_of: int = 256,
        device=None, dtype=None, operations=None
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
        self.proj_out = operations.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
        self.proj_in = operations.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))

def get_mlp(mlp_type: Optional[str] = "normal"):
    # 3b and 7b uses different mlp types
    if mlp_type == "normal":
        return MLP
    elif mlp_type == "swiglu":
        return SwiGLUMLP

class NaMMSRTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        vid_dim: int,
        txt_dim: int,
        emb_dim: int,
        heads: int,
        head_dim: int,
        expand_ratio: int,
        norm,
        norm_eps: float,
        ada,
        qk_bias: bool,
        qk_norm,
        mlp_type: str,
        shared_weights: bool,
        rope_type: str,
        rope_dim: int,
        is_last_layer: bool,
        device, dtype, operations,
        **kwargs,
    ):
        super().__init__()
        version = kwargs.get("version", False)
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, normalized_shape=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, device=device, dtype=dtype)

        self.attn = NaSwinAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm=qk_norm,
            qk_norm_eps=norm_eps,
            rope_type=rope_type,
            rope_dim=rope_dim,
            shared_weights=shared_weights,
            window=kwargs.pop("window", None),
            window_method=kwargs.pop("window_method", None),
            version=version,
            device=device, dtype=dtype, operations=operations
        )

        self.mlp_norm = MMModule(norm, normalized_shape=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer, device=device, dtype=dtype)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer,
            device=device, dtype=dtype, operations=operations
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"], shared_weights=shared_weights, vid_only=is_last_layer, device=device, dtype=dtype)
        self.is_last_layer = is_last_layer
        self.version = version

    def _seedvr2_7b_mlp(
        self,
        vid: torch.FloatTensor,
        txt: torch.FloatTensor,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        vid_module = self.mlp.vid if not self.mlp.shared_weights else self.mlp.all
        if comfy.model_management.in_training or vid.requires_grad:
            vid = torch.cat([vid_module(chunk) for chunk in vid.split(SEEDVR2_7B_MLP_CHUNK, dim=0)], dim=0)
        else:
            vid_out = None
            offset = 0
            for chunk in vid.split(SEEDVR2_7B_MLP_CHUNK, dim=0):
                chunk_out = vid_module(chunk)
                if vid_out is None:
                    vid_out = chunk_out.new_empty((vid.shape[0], *chunk_out.shape[1:]))
                vid_out[offset:offset + chunk_out.shape[0]] = chunk_out
                offset += chunk_out.shape[0]
            vid = vid_out
        if not self.mlp.vid_only:
            txt_module = self.mlp.txt if not self.mlp.shared_weights else self.mlp.all
            txt = txt.to(device=vid.device, dtype=vid.dtype)
            txt = txt_module(txt)
        return vid, txt

    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        txt: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,  # b 3
        txt_shape: torch.LongTensor,  # b 1
        emb: torch.FloatTensor,
        cache: Cache,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.LongTensor,
    ]:
        hid_len = MMArg(
            cache("vid_len", lambda: vid_shape.prod(-1)),
            cache("txt_len", lambda: txt_shape.prod(-1)),
        )
        ada_kwargs = {
            "emb": emb,
            "hid_len": hid_len,
            "cache": cache,
            "branch_tag": MMArg("vid", "txt"),
        }

        vid_attn, txt_attn = self.attn_norm(vid, txt)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="in", **ada_kwargs)
        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape, cache)
        vid_attn, txt_attn = self.ada(vid_attn, txt_attn, layer="attn", mode="out", **ada_kwargs)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
        if self.version:
            vid_mlp, txt_mlp = self._seedvr2_7b_mlp(vid_mlp, txt_mlp)
        else:
            vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="out", **ada_kwargs)
        vid_mlp, txt_mlp = (vid_mlp + vid_attn), (txt_mlp + txt_attn)

        return vid_mlp, txt_mlp, vid_shape, txt_shape

class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        device, dtype, operations
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = operations.Linear(dim, out_channels * t * h * w, device=device, dtype=dtype)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        vid = self.proj(vid)
        vid = rearrange(vid, "b T H W (t h w c) -> b c (T t) (H h) (W w)", t=t, h=h, w=w)
        if t > 1:
            vid = vid[:, :, (t - 1) :]
        return vid

class NaPatchOut(PatchOut):
    def forward(
        self,
        vid: torch.FloatTensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
        vid_shape_before_patchify = None
    ) -> Tuple[
        torch.FloatTensor,
        torch.LongTensor,
    ]:

        t, h, w = self.patch_size
        vid = self.proj(vid)

        if not (t == h == w == 1):
            vid = unflatten(vid, vid_shape)
            for i in range(len(vid)):
                vid[i] = rearrange(vid[i], "T H W (t h w c) -> (T t) (H h) (W w) c", t=t, h=h, w=w)
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = vid[i][(t - vid_shape_before_patchify[i, 0] % t) :]
            vid, vid_shape = flatten(vid)

        return vid, vid_shape

class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[int, Tuple[int, int, int]],
        dim: int,
        device, dtype, operations
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = operations.Linear(in_channels * t * h * w, dim, device=device, dtype=dtype)

    def forward(
        self,
        vid: torch.Tensor,
    ) -> torch.Tensor:
        t, h, w = self.patch_size
        if t > 1:
            assert vid.size(2) % t == 1
            vid = torch.cat([vid[:, :, :1]] * (t - 1) + [vid], dim=2)
        vid = rearrange(vid, "b c (T t) (H h) (W w) -> b T H W (t h w c)", t=t, h=h, w=w)
        vid = self.proj(vid)
        return vid

class NaPatchIn(PatchIn):
    def forward(
        self,
        vid: torch.Tensor,  # l c
        vid_shape: torch.LongTensor,
        cache: Cache = Cache(disable=True),  # for test
    ) -> torch.Tensor:
        cache = cache.namespace("patch")
        vid_shape_before_patchify = cache("vid_shape_before_patchify", lambda: vid_shape)
        t, h, w = self.patch_size
        if not (t == h == w == 1):
            vid = unflatten(vid, vid_shape)
            for i in range(len(vid)):
                if t > 1 and vid_shape_before_patchify[i, 0] % t != 0:
                    vid[i] = torch.cat([vid[i][:1]] * (t - vid[i].size(0) % t) + [vid[i]], dim=0)
                vid[i] = rearrange(vid[i], "(T t) (H h) (W w) c -> T H W (t h w c)", t=t, h=h, w=w)
            vid, vid_shape = flatten(vid)

        vid = self.proj(vid)
        return vid, vid_shape

def expand_dims(x: torch.Tensor, dim: int, ndim: int):
    shape = x.shape
    shape = shape[:dim] + (1,) * (ndim - len(shape)) + shape[dim:]
    return x.reshape(shape)


class AdaSingle(nn.Module):
    def __init__(
        self,
        dim: int,
        emb_dim: int,
        layers: List[str],
        modes: List[str] = ["in", "out"],
        device = None, dtype = None,
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers

        randn_kwargs = {"device": device}
        fp8_types = _torch_float8_types()
        if dtype is not None and dtype not in fp8_types:
            randn_kwargs["dtype"] = dtype

        for l in layers:
            if "in" in modes:
                # Passing fp8 ``dtype=`` here would break CPU weight
                # loads: CPU has no ``normal_kernel_cpu`` for fp8.
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim, **randn_kwargs) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim, **randn_kwargs) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim, **randn_kwargs) / dim**0.5))

    def forward(
        self,
        hid: torch.FloatTensor,  # b ... c
        emb: torch.FloatTensor,  # b d
        layer: str,
        mode: str,
        cache: Cache = Cache(disable=True),
        branch_tag: str = "",
        hid_len: Optional[torch.LongTensor] = None,  # b
    ) -> torch.FloatTensor:
        idx = self.layers.index(layer)
        emb = rearrange(emb, "b (d l g) -> b d l g", l=len(self.layers), g=3)[..., idx, :]
        emb = expand_dims(emb, 1, hid.ndim + 1)

        if hid_len is not None:
            slice_inputs = lambda x, dim: x
            emb = cache(
                f"emb_repeat_{idx}_{branch_tag}",
                lambda: slice_inputs(
                    torch.repeat_interleave(emb, hid_len, dim=0),
                    dim=0,
                ),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        fp8_types = _torch_float8_types()
        if fp8_types:
            target_dtype = hid.dtype

            if shiftB is not None and shiftB.dtype in fp8_types:
                shiftB = shiftB.to(target_dtype)
            if scaleB is not None and scaleB.dtype in fp8_types:
                scaleB = scaleB.to(target_dtype)
            if gateB is not None and gateB.dtype in fp8_types:
                gateB = gateB.to(target_dtype)

        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            if gateB is not None:
                return hid.mul_(gateA + gateB)
            else:
                return hid.mul_(gateA)

        raise NotImplementedError


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
        device, dtype, operations
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = operations.Linear(sinusoidal_dim, hidden_dim, device=device, dtype=dtype)
        self.proj_hid = operations.Linear(hidden_dim, hidden_dim, device=device, dtype=dtype)
        self.proj_out = operations.Linear(hidden_dim, output_dim, device=device, dtype=dtype)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        ).to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb

def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    assert len(hid) > 0
    shape = torch.stack([torch.tensor(x.shape[:-1], device=hid[0].device) for x in hid])
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    hid_len = hid_shape.prod(-1)
    hid = hid.split(hid_len.tolist())
    hid = [x.unflatten(0, s.tolist()) for x, s in zip(hid, hid_shape)]
    return hid

def repeat(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, torch.LongTensor],  # (b)
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    hid = unflatten(hid, hid_shape)
    kwargs = [{k: v[i].item() for k, v in kwargs.items()} for i in range(len(hid))]
    return flatten([einops.repeat(h, pattern, **a) for h, a in zip(hid, kwargs)])

class NaDiT(nn.Module):

    def __init__(
        self,
        norm_eps,
        qk_rope,
        num_layers,
        mlp_type,
        vid_in_channels = 33,
        vid_out_channels = 16,
        vid_dim = 2560,
        txt_in_dim = 5120,
        heads = 20,
        head_dim = 128,
        mm_layers = 10,
        expand_ratio = 4,
        qk_bias = False,
        patch_size = [ 1,2,2 ],
        shared_qkv: bool = False,
        shared_mlp: bool = False,
        window_method: Optional[Tuple[str]] = None,
        temporal_window_size: int = None,
        temporal_shifted: bool = False,
        rope_dim = 128,
        rope_type = "mmrope3d",
        vid_out_norm: Optional[str] = None,
        device = None,
        dtype = None,
        operations = None,
        **kwargs,
    ):
        self._7b_version = vid_dim == SEEDVR2_7B_VID_DIM
        if self._7b_version:
            rope_type = "rope3d"
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        window_method = num_layers // 2 * ["720pwin_by_size_bysize","720pswin_by_size_bysize"]
        txt_dim = vid_dim
        emb_dim = vid_dim * 6
        block_type = ["mmdit_sr"] * num_layers
        window = num_layers * [(4,3,3)]
        ada = AdaSingle
        norm = CustomRMSNorm
        qk_norm = CustomRMSNorm
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        # ``torch.empty`` returns uninitialized memory, not zeros. The
        # SeedVR2Conditioning fail-loud guard at
        # ``comfy_extras/nodes_seedvr.py`` distinguishes "buffer was loaded"
        # from "buffer was never populated by the file" by checking
        # ``positive_conditioning.abs().sum() == 0``. That sentinel is only
        # reliable if the post-construction buffer state is deterministically
        # zero, so explicitly zero-fill here rather than relying on the
        # allocator's zero-on-alloc behavior (allocator-dependent and not
        # contractual). When ``load_state_dict`` populates these buffers
        # from a properly-baked SeedVR2 .safetensors, the in-place copy
        # overwrites the zeros with the universal SeedVR2 conditioning
        # tensors (shape (58, 5120) and (64, 5120) bf16).
        self.register_buffer("positive_conditioning", torch.zeros((58, 5120), device=device, dtype=dtype))
        self.register_buffer("negative_conditioning", torch.zeros((64, 5120), device=device, dtype=dtype))
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
            device=device, dtype=dtype, operations=operations
        )
        self.txt_in = (
            operations.Linear(txt_in_dim, txt_dim, **factory_kwargs)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=BYTEDANCE_SINUSOIDAL_DIM,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
            device=device, dtype=dtype, operations=operations
        )

        if window is None or isinstance(window[0], int):
            window = [window] * num_layers
        if window_method is None or isinstance(window_method, str):
            window_method = [window_method] * num_layers
        if temporal_window_size is None or isinstance(temporal_window_size, int):
            temporal_window_size = [temporal_window_size] * num_layers
        if temporal_shifted is None or isinstance(temporal_shifted, bool):
            temporal_shifted = [temporal_shifted] * num_layers

        rope_dim = rope_dim if rope_dim is not None else head_dim // 2
        self.blocks = nn.ModuleList(
            [
                NaMMSRTransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    emb_dim=emb_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    norm_eps=norm_eps,
                    ada=ada,
                    qk_bias=qk_bias,
                    qk_rope=qk_rope,
                    qk_norm=qk_norm,
                    shared_qkv=shared_qkv,
                    shared_mlp=shared_mlp,
                    mlp_type=mlp_type,
                    rope_dim = rope_dim,
                    window=window[i],
                    window_method=window_method[i],
                    temporal_window_size=temporal_window_size[i],
                    temporal_shifted=temporal_shifted[i],
                    is_last_layer=(i == num_layers - 1) and not self._7b_version,
                    rope_type = rope_type,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    version = self._7b_version,
                    operations = operations,
                    **kwargs,
                    **factory_kwargs
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
            device=device, dtype=dtype, operations=operations
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

        self.vid_out_norm = None
        if vid_out_norm is not None:
            self.vid_out_norm = CustomRMSNorm(
                normalized_shape=vid_dim,
                eps=norm_eps,
                elementwise_affine=True,
                device=device, dtype=dtype
            )
            self.vid_out_ada = ada(
                dim=vid_dim,
                emb_dim=emb_dim,
                layers=["out"],
                modes=["in"],
                device=device, dtype=dtype
            )

    def _resolve_text_conditioning(self, context, cond_or_uncond=None):
        if context is None or getattr(context, "numel", lambda: None)() == 0:
            context = self.positive_conditioning
            return flatten([context])
        if NaDiT._seedvr2_is_single_conditioning_branch(cond_or_uncond):
            if context.shape[0] == 1:
                context = context.squeeze(0)
                return flatten([context])
            return flatten(context.unbind(0))
        if context.shape[0] % 2 != 0:
            raise ValueError(f"SeedVR2 expected an even text-conditioning batch, got shape {tuple(context.shape)}")
        neg_cond, pos_cond = context.chunk(2, dim=0)
        if pos_cond.shape[0] == 1:
            pos_cond, neg_cond = pos_cond.squeeze(0), neg_cond.squeeze(0)
            return flatten([pos_cond, neg_cond])
        return flatten((*pos_cond.unbind(0), *neg_cond.unbind(0)))

    @staticmethod
    def _seedvr2_is_single_conditioning_branch(cond_or_uncond):
        if cond_or_uncond is None or len(cond_or_uncond) == 0:
            return False
        first = cond_or_uncond[0]
        return all(entry == first for entry in cond_or_uncond)

    def _swap_pos_neg_halves(self, out, cond_or_uncond=None):
        if NaDiT._seedvr2_is_single_conditioning_branch(cond_or_uncond):
            return out
        # ``dim=0`` is explicit on both calls. The contract is "split
        # the batch axis into two halves and swap them"; making the
        # axis load-bearing in source guards against silent drift if a
        # future refactor reorders tensor axes.
        pos, neg = out.chunk(2, dim=0)
        return torch.cat([neg, pos], dim=0)

    def forward(
        self,
        x,
        timestep,
        context,  # l c
        disable_cache: bool = False,  # for test # TODO ? // gives an error when set to True
        **kwargs
    ):
        transformer_options = kwargs.get("transformer_options", {})
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        conditions = kwargs.get("condition")
        b, tc, h, w = x.shape
        x = x.view(b, 16, -1, h, w)
        conditions = conditions.view(b, 17, -1, h, w)
        x = x.movedim(1, -1)
        conditions = conditions.movedim(1, -1)
        cache = Cache(disable=disable_cache)

        txt, txt_shape = self._resolve_text_conditioning(context, transformer_options.get("cond_or_uncond"))

        vid, vid_shape = flatten(x)
        cond_latent, _ = flatten(conditions)

        vid = torch.cat([vid, cond_latent], dim=-1)
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])

        txt = self.txt_in(txt)

        vid_shape_before_patchify = vid_shape
        vid, vid_shape = self.vid_in(vid, vid_shape, cache=cache)

        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        for i, block in enumerate(self.blocks):
            if ("block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["vid"], out["txt"], out["vid_shape"], out["txt_shape"] = block(
                            vid=args["vid"],
                            txt=args["txt"],
                            vid_shape=args["vid_shape"],
                            txt_shape=args["txt_shape"],
                            emb=args["emb"],
                            cache=args["cache"],
                        )
                    return out
                out = blocks_replace[("block", i)]({
                        "vid":vid,
                        "txt":txt,
                        "vid_shape":vid_shape,
                        "txt_shape":txt_shape,
                        "emb":emb,
                        "cache":cache,
                    }, {"original_block": block_wrap})
                vid, txt, vid_shape, txt_shape = out["vid"], out["txt"], out["vid_shape"], out["txt_shape"]
            else:
                vid, txt, vid_shape, txt_shape = block(
                    vid=vid,
                    txt=txt,
                    vid_shape=vid_shape,
                    txt_shape=txt_shape,
                    emb=emb,
                    cache=cache,
                )

        if self.vid_out_norm:
            vid = self.vid_out_norm(vid)
            vid = self.vid_out_ada(
                vid,
                emb=emb,
                layer="out",
                mode="in",
                hid_len=cache("vid_len", lambda: vid_shape.prod(-1)),
                cache=cache,
                branch_tag="vid",
            )

        vid, vid_shape = self.vid_out(vid, vid_shape, cache, vid_shape_before_patchify = vid_shape_before_patchify)
        vid = unflatten(vid, vid_shape)
        out =  torch.stack(vid)
        out = out.movedim(-1, 1)
        out = rearrange(out, "b c t h w -> b (c t) h w")
        return self._swap_pos_neg_halves(out, transformer_options.get("cond_or_uncond"))

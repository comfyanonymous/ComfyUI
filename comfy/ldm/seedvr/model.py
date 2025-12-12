from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict, Any, Callable
import einops
from einops import rearrange, repeat
import comfy.model_management
from torch import nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from math import ceil, pi
import torch
from itertools import chain
from comfy.ldm.modules.diffusionmodules.model import get_timestep_embedding
from comfy.ldm.modules.attention import optimized_attention
from comfy.rmsnorm import RMSNorm
from torch.nn.modules.utils import _triple
from torch import nn
import math

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
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.
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
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh, ww = ceil(resized_h / resized_nh), ceil(resized_w / resized_nw)  # window size.
    wt = ceil(min(t, 30) / resized_nt)  # window size.
    
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

        # get frequencies for each axis

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
            max_freq=256,
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


class MMRotaryEmbeddingBase(RotaryEmbeddingBase):
    def __init__(self, dim: int, rope_dim: int):
        super().__init__(dim, rope_dim)
        self.rope = RotaryEmbedding(
            dim=dim // rope_dim,
            freqs_for="lang",
            theta=10000,
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

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    freqs = freqs.to(t_middle.device)
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
        
    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)

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
        vid_q = rearrange(vid_q, "L h d -> h L d")
        vid_k = rearrange(vid_k, "L h d -> h L d")
        vid_q = apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
        vid_q = rearrange(vid_q, "h L d -> L h d")
        vid_k = rearrange(vid_k, "h L d -> L h d")

        txt_q = rearrange(txt_q, "L h d -> h L d")
        txt_k = rearrange(txt_k, "L h d -> h L d")
        txt_q = apply_rotary_emb(txt_freqs, txt_q.float()).to(txt_q.dtype)
        txt_k = apply_rotary_emb(txt_freqs, txt_k.float()).to(txt_k.dtype)
        txt_q = rearrange(txt_q, "h L d -> L h d")
        txt_k = rearrange(txt_k, "h L d -> L h d")
        return vid_q, vid_k, txt_q, txt_k

    def get_freqs(
        self,
        vid_shape: torch.LongTensor,
        txt_shape: torch.LongTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        vid_freqs = self.get_axial_freqs(1024, 128, 128)
        txt_freqs = self.get_axial_freqs(1024)
        vid_freq_list, txt_freq_list = [], []
        for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
            vid_freq = vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1))
            txt_freq = txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1))
            vid_freq_list.append(vid_freq)
            txt_freq_list.append(txt_freq)
        return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)

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
        device = comfy.model_management.get_torch_device()
        vid = vid.to(device)
        vid = vid_module(vid, *get_args("vid", args), **get_kwargs("vid", kwargs))
        if not self.vid_only:
            txt_module = self.txt if not self.shared_weights else self.all
            txt = txt.to(device=vid.device, dtype=vid.dtype)
            txt = txt_module(txt, *get_args("txt", args), **get_kwargs("txt", kwargs))
        return vid, txt

def get_na_rope(rope_type: Optional[str], dim: int):
    # 7b doesn't use rope
    if rope_type is None:
        return None
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
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.heads = heads
        inner_dim = heads * head_dim
        qkv_dim = inner_dim * 3
        self.head_dim = head_dim
        self.proj_qkv = MMModule(
            nn.Linear, dim, qkv_dim, bias=qk_bias, shared_weights=shared_weights
        )
        self.proj_out = MMModule(nn.Linear, inner_dim, dim, shared_weights=shared_weights)
        self.norm_q = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
        )
        self.norm_k = MMModule(
            qk_norm,
            normalized_shape=head_dim,
            eps=qk_norm_eps,
            elementwise_affine=True,
            shared_weights=shared_weights,
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
        txt_len_win = cache_win("txt_len", lambda: txt_len.repeat_interleave(window_count))
        all_len_win = cache_win("all_len", lambda: vid_len_win + txt_len_win)
        concat_win, unconcat_win = cache_win(
            "mm_pnp", lambda: repeat_concat_idx(vid_len_win, txt_len, window_count)
        )

        # window rope
        if self.rope:
            if self.rope.mm:
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
        
        # TODO: continue testing
        v_lens = vid_len_win.cpu().tolist()
        t_lens_batch = txt_len.cpu().tolist()
        win_counts = window_count.cpu().tolist()

        vq_l = torch.split(vid_q, v_lens)
        vk_l = torch.split(vid_k, v_lens)
        vv_l = torch.split(vid_v, v_lens) 

        tv_batch = torch.split(txt_v, t_lens_batch)
        tv_l = []
        for i, count in enumerate(win_counts):
            tv_l.extend([tv_batch[i]] * count)

        current_txt_len = txt_q.shape[0]
        expected_batch_len = sum(t_lens_batch)

        if current_txt_len != expected_batch_len:
            t_lens_win = txt_len_win.cpu().tolist()
            
            tq_l = torch.split(txt_q, t_lens_win)
            tk_l = torch.split(txt_k, t_lens_win)
        else:
            tq_batch = torch.split(txt_q, t_lens_batch)
            tk_batch = torch.split(txt_k, t_lens_batch)
            
            tq_l = []
            tk_l = []
            for i, count in enumerate(win_counts):
                tq_l.extend([tq_batch[i]] * count)
                tk_l.extend([tk_batch[i]] * count)

        q_list = [torch.cat([v, t], dim=0) for v, t in zip(vq_l, tq_l)]
        k_list = [torch.cat([v, t], dim=0) for v, t in zip(vk_l, tk_l)]
        v_list = [torch.cat([v, t], dim=0) for v, t in zip(vv_l, tv_l)]
        
        q = rnn_utils.pad_sequence(q_list, batch_first=True)
        k = rnn_utils.pad_sequence(k_list, batch_first=True)
        v = rnn_utils.pad_sequence(v_list, batch_first=True)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        B, Heads, Max_L, _ = q.shape
        combined_lens = [v.shape[0] + t.shape[0] for v, t in zip(vq_l, tq_l)]
        
        attn_mask = torch.zeros((B, 1, 1, Max_L), device=q.device, dtype=q.dtype)
        idx = torch.arange(Max_L, device=q.device).unsqueeze(0).expand(B, Max_L)
        len_tensor = torch.tensor(combined_lens, device=q.device).unsqueeze(1)
        
        padding_mask = idx >= len_tensor
        attn_mask.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        out = optimized_attention(q, k, v, heads=self.heads, mask=attn_mask, skip_reshape=True, skip_output_reshape=True)
        
        out = out.transpose(1, 2)
        
        out_flat_list = []
        for i, length in enumerate(combined_lens):
            out_flat_list.append(out[i, :length])
        
        out = torch.cat(out_flat_list, dim=0)

        vid_out, txt_out = unconcat_win(out)

        vid_out = rearrange(vid_out, "l h d -> l (h d)")
        txt_out = rearrange(txt_out, "l h d -> l (h d)")
        vid_out = window_reverse(vid_out)

        device = comfy.model_management.get_torch_device()
        vid_out, txt_out = vid_out.to(device), txt_out.to(device)
        self.proj_out = self.proj_out.to(device)
        vid_out, txt_out = self.proj_out(vid_out, txt_out)

        return vid_out, txt_out
    
class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int,
    ):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * expand_ratio)
        self.act = nn.GELU("tanh")
        self.proj_out = nn.Linear(dim * expand_ratio, dim)

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
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=False)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.to(next(self.proj_in.parameters()).device)
        self.proj_out = self.proj_out.to(x.device)
        x = self.proj_out(F.silu(self.proj_in_gate(x)) * self.proj_in(x))
        return x

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
        **kwargs,
    ):
        super().__init__()
        dim = MMArg(vid_dim, txt_dim)
        self.attn_norm = MMModule(norm, normalized_shape=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights,)

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
        )

        self.mlp_norm = MMModule(norm, normalized_shape=dim, eps=norm_eps, elementwise_affine=False, shared_weights=shared_weights, vid_only=is_last_layer)
        self.mlp = MMModule(
            get_mlp(mlp_type),
            dim=dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            vid_only=is_last_layer
        )
        self.ada = MMModule(ada, dim=dim, emb_dim=emb_dim, layers=["attn", "mlp"], shared_weights=shared_weights, vid_only=is_last_layer)
        self.is_last_layer = is_last_layer

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
        txt = txt.to(txt_attn.device)
        vid_attn, txt_attn = (vid_attn + vid), (txt_attn + txt)

        vid_mlp, txt_mlp = self.mlp_norm(vid_attn, txt_attn)
        vid_mlp, txt_mlp = self.ada(vid_mlp, txt_mlp, layer="mlp", mode="in", **ada_kwargs)
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
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(dim, out_channels * t * h * w)

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
    ):
        super().__init__()
        t, h, w = _triple(patch_size)
        self.patch_size = t, h, w
        self.proj = nn.Linear(in_channels * t * h * w, dim)

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
    ):
        assert emb_dim == 6 * dim, "AdaSingle requires emb_dim == 6 * dim"
        super().__init__()
        self.dim = dim
        self.emb_dim = emb_dim
        self.layers = layers
        for l in layers:
            if "in" in modes:
                self.register_parameter(f"{l}_shift", nn.Parameter(torch.randn(dim) / dim**0.5))
                self.register_parameter(
                    f"{l}_scale", nn.Parameter(torch.randn(dim) / dim**0.5 + 1)
                )
            if "out" in modes:
                self.register_parameter(f"{l}_gate", nn.Parameter(torch.randn(dim) / dim**0.5))

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
                    torch.cat([e.repeat(l, *([1] * e.ndim)) for e, l in zip(emb, hid_len)]),
                    dim=0,
                ),
            )

        shiftA, scaleA, gateA = emb.unbind(-1)
        shiftB, scaleB, gateB = (
            getattr(self, f"{layer}_shift", None),
            getattr(self, f"{layer}_scale", None),
            getattr(self, f"{layer}_gate", None),
        )

        if mode == "in":
            return hid.mul_(scaleA + scaleB).add_(shiftA + shiftB)
        if mode == "out":
            return hid.mul_(gateA + gateB)
        raise NotImplementedError


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
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
        )
        emb = emb.to(dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        device = next(self.proj_hid.parameters()).device
        emb = emb.to(device)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        device = next(self.proj_out.parameters()).device
        emb = emb.to(device)
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
        self.dtype = dtype
        window_method = num_layers // 2 * ["720pwin_by_size_bysize","720pswin_by_size_bysize"]
        txt_dim = vid_dim
        emb_dim = vid_dim * 6
        block_type = ["mmdit_sr"] * num_layers
        window = num_layers * [(4,3,3)]
        ada = AdaSingle
        norm = RMSNorm
        qk_norm = RMSNorm
        if isinstance(block_type, str):
            block_type = [block_type] * num_layers
        elif len(block_type) != num_layers:
            raise ValueError("The ``block_type`` list should equal to ``num_layers``.")
        super().__init__()
        self.register_buffer("positive_conditioning", torch.empty((58, 5120)))
        self.register_buffer("negative_conditioning", torch.empty((64, 5120)))
        self.vid_in = NaPatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )
        self.txt_in = (
            nn.Linear(txt_in_dim, txt_dim)
            if txt_in_dim and txt_in_dim != txt_dim
            else nn.Identity()
        )
        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
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
                    is_last_layer=(i == num_layers - 1),
                    rope_type = rope_type,
                    shared_weights=not (
                        (i < mm_layers) if isinstance(mm_layers, int) else mm_layers[i]
                    ),
                    **kwargs,
                )
                for i in range(num_layers)
            ]
        )
        self.vid_out = NaPatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

        self.need_txt_repeat = block_type[0] in [
            "mmdit_stwin",
            "mmdit_stwin_spatial",
            "mmdit_stwin_3d_spatial",
        ]

        self.vid_out_norm = None
        if vid_out_norm is not None:
            self.vid_out_norm = RMSNorm(
                normalized_shape=vid_dim,
                eps=norm_eps,
                elementwise_affine=True,
            )
            self.vid_out_ada = ada(
                dim=vid_dim,
                emb_dim=emb_dim,
                layers=["out"],
                modes=["in"],
            )

    def forward(
        self,
        x,
        timestep,
        context,  # l c
        disable_cache: bool = False,  # for test # TODO ? // gives an error when set to True
        **kwargs
    ):  
        transformer_options = kwargs.get("transformer_options", {})
        conditions = kwargs.get("condition")

        pos_cond, neg_cond = context.squeeze(0).chunk(2, dim=0)
        pos_cond, neg_cond = pos_cond.squeeze(0), neg_cond.squeeze(0)
        pos_cond, txt_shape = flatten([pos_cond])
        neg_cond, _ = flatten([neg_cond])
        txt = torch.cat([pos_cond, neg_cond], dim = 0)

        vid = x
        vid, vid_shape = flatten(x)
        cond_latent, _ = flatten(conditions)

        vid = torch.cat([cond_latent, vid], dim=-1)
        if txt_shape.size(-1) == 1 and self.need_txt_repeat:
            txt, txt_shape = repeat(txt, txt_shape, "l c -> t l c", t=vid_shape[:, 0])

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        txt = txt.to(device).to(dtype)
        vid = vid.to(device).to(dtype)
        txt = self.txt_in(txt.to(next(self.txt_in.parameters()).device))

        vid_shape_before_patchify = vid_shape
        vid, vid_shape = self.vid_in(vid, vid_shape)

        emb = self.emb_in(timestep, device=vid.device, dtype=vid.dtype)

        cache = Cache(disable=disable_cache)
        for i, block in enumerate(self.blocks):
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
        return vid[0]

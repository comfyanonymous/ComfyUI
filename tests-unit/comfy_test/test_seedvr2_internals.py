"""Consolidated SeedVR2 internals regression tests.

Sources (all merged verbatim, helper names disambiguated where colliding):

  * RoPE rewrite — NaMMRotaryEmbedding3d.forward must match the legacy
    apply_rotary_emb wrapper oracle at fp32.
  * GroupNorm limit gate — causal_norm_wrapper at vae.py:509 must compare
    memory_occupy against get_norm_limit(), not float('inf').
  * SeedVR2 variable-length attention split-loop contract.

Pre-import CPU-only guard is required because comfy.ldm.seedvr.model and
comfy.ldm.modules.attention transitively pull in comfy.model_management,
which probes torch.cuda.current_device() at import time unless args.cpu is
set first.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
import comfy.ldm.seedvr.vae as vae_mod  # noqa: E402
import comfy.ldm.modules.attention as attention  # noqa: E402
import comfy.ops as comfy_ops  # noqa: E402
from comfy.ldm.seedvr.model import (  # noqa: E402
    Cache,
    NaMMRotaryEmbedding3d,
)
from comfy.ldm.seedvr.vae import (  # noqa: E402
    causal_norm_wrapper,
    set_norm_limit,
)
from comfy.ldm.modules.attention import var_attention_optimized_split  # noqa: E402


# ---------------------------------------------------------------------------
# RoPE rewrite tests (test_seedvr_rope_rewrite.py)
# ---------------------------------------------------------------------------

# Test rig dimensions. dim=192 → per-axis rope dim = 64 (even, lucidrains
# requirement). vid_shape=(2,4,4) → L_vid = 32. txt_shape=(8,) → L_txt = 8.
_DIM = 192
_HEADS = 4
_VID_T, _VID_H, _VID_W = 2, 4, 4
_TXT_L = 8
_L_VID = _VID_T * _VID_H * _VID_W
_SEED = 0


def _make_inputs(dtype=torch.float32, device="cpu"):
    """Construct the 6 forward inputs + cache. Deterministic via local
    Generator so global RNG state is not mutated.
    """
    g = torch.Generator(device=device).manual_seed(_SEED)
    vid_q = torch.randn(_L_VID, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    vid_k = torch.randn(_L_VID, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    txt_q = torch.randn(_TXT_L, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    txt_k = torch.randn(_TXT_L, _HEADS, _DIM, dtype=dtype, device=device, generator=g)
    vid_shape = torch.tensor([[_VID_T, _VID_H, _VID_W]], dtype=torch.long, device=device)
    txt_shape = torch.tensor([[_TXT_L]], dtype=torch.long, device=device)
    cache = Cache(disable=True)
    return vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache


def _legacy_get_freqs(rope: NaMMRotaryEmbedding3d, vid_shape, txt_shape):
    """Reproduce the pre-rewrite ``get_freqs`` body verbatim against
    ``self.get_axial_freqs`` (parent ``RotaryEmbeddingBase`` method,
    unchanged by the rewrite).
    """
    max_temporal = 0
    max_height = 0
    max_width = 0
    max_txt_len = 0
    for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        max_temporal = max(max_temporal, l + f)
        max_height = max(max_height, h)
        max_width = max(max_width, w)
        max_txt_len = max(max_txt_len, l)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        vid_freqs_full = rope.get_axial_freqs(
            min(max_temporal + 16, 1024),
            min(max_height + 4, 128),
            min(max_width + 4, 128),
        ).float()
        txt_freqs_full = rope.get_axial_freqs(min(max_txt_len + 16, 1024))
    vid_freq_list, txt_freq_list = [], []
    for (f, h, w), l in zip(vid_shape.tolist(), txt_shape[:, 0].tolist()):
        vid_freq = vid_freqs_full[l : l + f, :h, :w].reshape(-1, vid_freqs_full.size(-1))
        txt_freq = txt_freqs_full[:l].repeat(1, 3).reshape(-1, vid_freqs_full.size(-1))
        vid_freq_list.append(vid_freq)
        txt_freq_list.append(txt_freq)
    return torch.cat(vid_freq_list, dim=0), torch.cat(txt_freq_list, dim=0)


def _legacy_forward(rope: NaMMRotaryEmbedding3d, vid_q, vid_k, vid_shape,
                    txt_q, txt_k, txt_shape):
    """Compute expected forward output via the unchanged
    ``apply_rotary_emb`` wrapper fed with legacy-shape freqs. This is the
    oracle. The wrapper itself is out of scope for the rewrite (Shape B).
    """
    vid_freqs, txt_freqs = _legacy_get_freqs(rope, vid_shape, txt_shape)
    vid_freqs = vid_freqs.to(vid_q.device)
    txt_freqs = txt_freqs.to(txt_q.device)

    from einops import rearrange

    vid_q = rearrange(vid_q, "L h d -> h L d")
    vid_k = rearrange(vid_k, "L h d -> h L d")
    vid_q_out = seedvr_model.apply_rotary_emb(vid_freqs, vid_q.float()).to(vid_q.dtype)
    vid_k_out = seedvr_model.apply_rotary_emb(vid_freqs, vid_k.float()).to(vid_k.dtype)
    vid_q_out = rearrange(vid_q_out, "h L d -> L h d")
    vid_k_out = rearrange(vid_k_out, "h L d -> L h d")

    txt_q = rearrange(txt_q, "L h d -> h L d")
    txt_k = rearrange(txt_k, "L h d -> h L d")
    txt_q_out = seedvr_model.apply_rotary_emb(txt_freqs, txt_q.float()).to(txt_q.dtype)
    txt_k_out = seedvr_model.apply_rotary_emb(txt_freqs, txt_k.float()).to(txt_k.dtype)
    txt_q_out = rearrange(txt_q_out, "h L d -> L h d")
    txt_k_out = rearrange(txt_k_out, "h L d -> L h d")
    return vid_q_out, vid_k_out, txt_q_out, txt_k_out


def test_namm_forward_output_tensor_equal_against_legacy_oracle():
    rope = NaMMRotaryEmbedding3d(dim=_DIM)
    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache = _make_inputs()

    expected_vid_q, expected_vid_k, expected_txt_q, expected_txt_k = _legacy_forward(
        rope,
        vid_q.clone(), vid_k.clone(), vid_shape,
        txt_q.clone(), txt_k.clone(), txt_shape,
    )

    actual_vid_q, actual_vid_k, actual_txt_q, actual_txt_k = rope.forward(
        vid_q.clone(), vid_k.clone(), vid_shape,
        txt_q.clone(), txt_k.clone(), txt_shape, cache,
    )

    torch.testing.assert_close(actual_vid_q, expected_vid_q, rtol=0, atol=0,
                                msg="vid_q output diverges from wrapper oracle")
    torch.testing.assert_close(actual_vid_k, expected_vid_k, rtol=0, atol=0,
                                msg="vid_k output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_q, expected_txt_q, rtol=0, atol=0,
                                msg="txt_q output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_k, expected_txt_k, rtol=0, atol=0,
                                msg="txt_k output diverges from wrapper oracle")


# ---------------------------------------------------------------------------
# GroupNorm limit tests (test_seedvr_groupnorm_limit.py)
# ---------------------------------------------------------------------------

_NUM_CHANNELS = 8
_NUM_GROUPS = 4
_TENSOR_SHAPE = (1, 8, 2, 4, 4)

_GROUPNORM_SUBCLASSES = [
    pytest.param(comfy_ops.disable_weight_init.GroupNorm, id="disable_weight_init"),
    pytest.param(comfy_ops.manual_cast.GroupNorm, id="manual_cast"),
]


@pytest.mark.parametrize("groupnorm_cls", _GROUPNORM_SUBCLASSES)
def test_seedvr_groupnorm_low_limit_uses_chunked_groupnorm_path(groupnorm_cls):
    real_group_norm = vae_mod.F.group_norm
    set_norm_limit(1e-9)
    try:
        gn = groupnorm_cls(num_channels=_NUM_CHANNELS, num_groups=_NUM_GROUPS)
        gn.eval()

        forward_hook_calls = []

        def _hook(module, inputs, output):
            forward_hook_calls.append(tuple(inputs[0].shape))

        spy_calls = []

        def _group_norm_spy(input_tensor, num_groups_arg, *args, **kwargs):
            spy_calls.append({"num_groups": int(num_groups_arg)})
            return real_group_norm(input_tensor, num_groups_arg, *args, **kwargs)

        handle = gn.register_forward_hook(_hook)
        try:
            with patch.object(vae_mod.F, "group_norm", side_effect=_group_norm_spy):
                out_tensor = causal_norm_wrapper(gn, torch.randn(*_TENSOR_SHAPE))
        finally:
            handle.remove()

        full_calls = len(forward_hook_calls)
        chunked_calls = sum(1 for entry in spy_calls if entry["num_groups"] < _NUM_GROUPS)

        assert tuple(int(s) for s in out_tensor.shape) == _TENSOR_SHAPE
        assert full_calls == 0, (
            f"low-limit GroupNorm gate must NOT take the full-forward path; got full_calls={full_calls}"
        )
        assert chunked_calls > 0, (
            f"low-limit GroupNorm gate must take the chunked path; got chunked_calls={chunked_calls}"
        )
    finally:
        set_norm_limit(None)


# ---------------------------------------------------------------------------
# SeedVR2 var_attention split-loop tests
# ---------------------------------------------------------------------------

def test_var_attention_registry_contains_always_available_entries():
    assert (
        attention.REGISTERED_ATTENTION_FUNCTIONS["var_attention_optimized_split"]
        is attention.var_attention_optimized_split
    )


def test_seedvr2_7b_swin_attention_forward_uses_optimized_var_attention(monkeypatch):
    dim = 8
    heads = 2
    head_dim = 4
    attn = seedvr_model.NaSwinAttention(
        vid_dim=dim,
        txt_dim=dim,
        heads=heads,
        head_dim=head_dim,
        qk_bias=False,
        qk_norm=seedvr_model.CustomRMSNorm,
        qk_norm_eps=1e-6,
        rope_type=None,
        rope_dim=head_dim,
        shared_weights=False,
        window=(2, 1, 1),
        window_method="720pwin_by_size_bysize",
        version=True,
        device="cpu",
        dtype=torch.float32,
        operations=comfy_ops.disable_weight_init,
    )
    generator = torch.Generator(device="cpu").manual_seed(11)
    vid = torch.randn(8, dim, generator=generator)
    txt = torch.randn(3, dim, generator=generator)
    vid_shape = torch.tensor([[2, 2, 2]], dtype=torch.long)
    txt_shape = torch.tensor([[3]], dtype=torch.long)
    calls = []

    def fake_optimized_var_attention(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(seedvr_model, "optimized_var_attention", fake_optimized_var_attention)

    vid_out, txt_out = attn(vid, txt, vid_shape, txt_shape, seedvr_model.Cache(disable=True))

    assert tuple(vid_out.shape) == (8, dim)
    assert tuple(txt_out.shape) == (3, dim)
    assert len(calls) == 1
    call = calls[0]
    assert tuple(call["q"].shape) == (14, heads, head_dim)
    assert tuple(call["k"].shape) == (14, heads, head_dim)
    assert tuple(call["v"].shape) == (14, heads, head_dim)
    assert call["heads"] == heads
    assert call["skip_reshape"] is True
    assert call["skip_output_reshape"] is True
    torch.testing.assert_close(
        call["cu_seqlens_q"],
        torch.tensor([0, 7, 14], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        call["cu_seqlens_k"],
        torch.tensor([0, 7, 14], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_var_attention_optimized_split_calls_dense_backend_per_window(monkeypatch):
    heads = 2
    head_dim = 3
    q = torch.arange(30, dtype=torch.float32).reshape(5, heads, head_dim)
    k = q + 100
    v = q + 200
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    calls = []

    def fake_optimized_attention(q_arg, k_arg, v_arg, heads_arg, **kwargs):
        calls.append(
            {
                "q_shape": tuple(q_arg.shape),
                "k_shape": tuple(k_arg.shape),
                "v_shape": tuple(v_arg.shape),
                "heads": heads_arg,
                "kwargs": kwargs,
            }
        )
        return q_arg + v_arg

    monkeypatch.setattr(attention, "optimized_attention", fake_optimized_attention)

    out = var_attention_optimized_split(
        q,
        k,
        v,
        heads,
        cu,
        cu,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    assert tuple(out.shape) == (5, heads, head_dim)
    assert len(calls) == 2
    assert calls[0]["q_shape"] == (1, heads, 2, head_dim)
    assert calls[1]["q_shape"] == (1, heads, 3, head_dim)
    assert all(call["heads"] == heads for call in calls)
    assert all(call["kwargs"]["skip_reshape"] is True for call in calls)
    assert all(call["kwargs"]["skip_output_reshape"] is True for call in calls)
    torch.testing.assert_close(out, q + v, rtol=0, atol=0)


def test_var_attention_optimized_split_rejects_bad_offsets():
    q = torch.randn(5, 2, 3)
    cu_bad = torch.tensor([0, 2, 6], dtype=torch.int32)
    cu_ok = torch.tensor([0, 2, 5], dtype=torch.int32)

    with pytest.raises(ValueError, match="cu_seqlens_q does not match token count"):
        var_attention_optimized_split(
            q,
            q,
            q,
            2,
            cu_bad,
            cu_ok,
            skip_reshape=True,
            skip_output_reshape=True,
        )

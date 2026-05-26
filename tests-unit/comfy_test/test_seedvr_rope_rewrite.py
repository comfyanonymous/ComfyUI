"""Regression tests for the SeedVR2 native RoPE rewrite that replaces the
``apply_rotary_emb`` wrapper inside ``NaMMRotaryEmbedding3d.forward`` with
direct calls to ``comfy.ldm.flux.math.apply_rope1`` — matching the pattern
used by the other 7 ComfyUI native-DiT models (flux, hidream, kandinsky5,
lumina, qwen_image, wan, sam3).

The wrapper builds a 2x2 ``freqs_mat`` and ends in ``torch.cat((t_left,
t_middle_out, t_right), dim=-1)``; that cat OOMs on the largest cell of the
SeedVR2 native_3b non-tiled corpus (VideoLQ_000 1280x960x100 on RTX 5090
32GB). Canonical and numz pass the same cell because both call
``rotary_embedding_torch.apply_rotary_emb`` directly. The fix moves the
NaMMRotaryEmbedding3d path onto ``apply_rope1`` directly with freqs in
flux-canonical shape ``[..., d/2, 2, 2]`` (cos/-sin/sin/cos baked in).

This test file pins four invariants the rewrite must satisfy:

1. ``NaMMRotaryEmbedding3d.forward`` calls ``apply_rope1`` 4 times per
   forward (vid_q, vid_k, txt_q, txt_k) and 0 times into the
   ``apply_rotary_emb`` wrapper.
2. ``NaMMRotaryEmbedding3d.get_freqs`` returns freqs in flux-canonical shape
   ``[..., d/2, 2, 2]`` with the cos/-sin/sin/cos pattern from
   ``comfy/ldm/flux/math.py:rope`` (line 27).
3. The forward output is tensor-equal at fp32 against an oracle computed
   from the unchanged ``apply_rotary_emb`` wrapper fed with the legacy
   freqs layout — proving the rewrite is algorithmically lossless.
4. AST: no ``apply_rotary_emb`` call sites remain inside
   ``NaMMRotaryEmbedding3d.forward``.

The wrapper itself stays in the file (still used by
``RotaryEmbedding3d.forward`` lines 434-435 and the staticmethod
registration on lucidrains' ``RotaryEmbedding`` line 323). Out of scope
here.

Pre-import CPU-only guard mirrors ``test_seedvr_rope_delegation.py`` —
``comfy.ldm.seedvr.model`` transitively imports ``comfy.model_management``
which probes ``torch.cuda.current_device()`` at import time unless
``args.cpu`` is set first.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import patch

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402
from comfy.ldm.seedvr.model import (  # noqa: E402
    Cache,
    NaMMRotaryEmbedding3d,
)


# Test rig dimensions. dim=192 → per-axis rope dim = 64 (even, lucidrains
# requirement). vid_shape=(2,4,4) → L_vid = 32. txt_shape=(8,) → L_txt = 8.
# heads = 4. These are all small enough to run on CPU in milliseconds.
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
    unchanged by the rewrite). Used by Test 3 to compute the oracle from
    the wrapper path post-rewrite, when ``rope.get_freqs`` itself returns
    the new flux-canonical shape.
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


# Test 1 — drives AC-4 (call-graph): forward must reach apply_rope1 directly,
# never via the apply_rotary_emb wrapper.

def test_namm_forward_calls_apply_rope1_directly():
    rope = NaMMRotaryEmbedding3d(dim=_DIM)
    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache = _make_inputs()

    with patch.object(
        seedvr_model, "apply_rotary_emb", wraps=seedvr_model.apply_rotary_emb
    ) as wrapper_spy, patch.object(
        seedvr_model, "apply_rope1", wraps=seedvr_model.apply_rope1
    ) as rope1_spy:
        rope.forward(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache)

    assert wrapper_spy.call_count == 0, (
        f"NaMMRotaryEmbedding3d.forward must not call apply_rotary_emb "
        f"(saw {wrapper_spy.call_count} calls); the rewrite must rewire "
        f"the 4 forward sites to apply_rope1 directly"
    )
    assert rope1_spy.call_count == 4, (
        f"NaMMRotaryEmbedding3d.forward must call apply_rope1 exactly 4 "
        f"times (vid_q, vid_k, txt_q, txt_k); saw {rope1_spy.call_count}"
    )


# Test 2 — drives the get_freqs shape change to flux-canonical layout.

def test_get_freqs_emits_flux_canonical_shape():
    rope = NaMMRotaryEmbedding3d(dim=_DIM)
    vid_shape = torch.tensor([[_VID_T, _VID_H, _VID_W]], dtype=torch.long)
    txt_shape = torch.tensor([[_TXT_L]], dtype=torch.long)

    vid_freqs, txt_freqs = rope.get_freqs(vid_shape, txt_shape)

    # Flux's `rope()` (comfy/ldm/flux/math.py:17-29) emits freqs in shape
    # [..., d/2, 2, 2] via stack([cos, -sin, sin, cos], dim=-1) +
    # rearrange("b n d (i j) -> b n d i j", i=2, j=2). The rewrite must
    # match: ndim >= 4, last two dims both == 2.
    assert vid_freqs.ndim >= 4, (
        f"vid_freqs.ndim must be >= 4 (flux-canonical layout has trailing "
        f"[..., d/2, 2, 2]); got ndim={vid_freqs.ndim}, shape={tuple(vid_freqs.shape)}"
    )
    assert vid_freqs.shape[-1] == 2, (
        f"vid_freqs.shape[-1] must be 2 (rotation matrix column); got "
        f"shape={tuple(vid_freqs.shape)}"
    )
    assert vid_freqs.shape[-2] == 2, (
        f"vid_freqs.shape[-2] must be 2 (rotation matrix row); got "
        f"shape={tuple(vid_freqs.shape)}"
    )
    assert txt_freqs.ndim >= 4, (
        f"txt_freqs must also be flux-canonical; got ndim={txt_freqs.ndim}, "
        f"shape={tuple(txt_freqs.shape)}"
    )
    assert txt_freqs.shape[-1] == 2 and txt_freqs.shape[-2] == 2, (
        f"txt_freqs trailing dims must be (2, 2); got shape={tuple(txt_freqs.shape)}"
    )

    # Verify the cos/-sin/sin/cos pattern at index 0:
    #   freqs_cis[..., 0, 0] = cos
    #   freqs_cis[..., 0, 1] = -sin
    #   freqs_cis[..., 1, 0] = sin
    #   freqs_cis[..., 1, 1] = cos
    # so [0,0] == [1,1] (both cos) and [0,1] == -[1,0] (=-sin vs +sin).
    cos_a = vid_freqs[..., 0, 0]
    cos_b = vid_freqs[..., 1, 1]
    neg_sin = vid_freqs[..., 0, 1]
    sin = vid_freqs[..., 1, 0]
    assert torch.allclose(cos_a, cos_b, rtol=0, atol=0), (
        "vid_freqs[..., 0, 0] must equal vid_freqs[..., 1, 1] (both = cos)"
    )
    assert torch.allclose(neg_sin, -sin, rtol=0, atol=0), (
        "vid_freqs[..., 0, 1] must equal -vid_freqs[..., 1, 0] (= -sin vs +sin)"
    )


# Test 3 — drives AC-1: forward output is tensor-equal against the wrapper-
# fed oracle. Pre-rewrite: trivially passes (forward IS the wrapper path).
# Post-rewrite: must remain equal. Exact equality (rtol=atol=0) at fp32.

def test_namm_forward_output_tensor_equal_against_legacy_oracle():
    rope = NaMMRotaryEmbedding3d(dim=_DIM)
    vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, cache = _make_inputs()

    # Oracle: the unchanged apply_rotary_emb wrapper fed with legacy-shape
    # freqs produced by reproducing the pre-rewrite get_freqs body.
    expected_vid_q, expected_vid_k, expected_txt_q, expected_txt_k = _legacy_forward(
        rope,
        vid_q.clone(), vid_k.clone(), vid_shape,
        txt_q.clone(), txt_k.clone(), txt_shape,
    )

    # Actual: NaMMRotaryEmbedding3d.forward (under test).
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


# Test 5 — partial-rope coverage. The real SeedVR2-3B model is constructed
# with rope_dim=128, which integer-divides into 3 axes as 128//3 = 42 per-
# axis; total rope freq dims = 42*3 = 126. head_dim is 128, so the trailing
# 2 dims of each q/k must be passed through unrotated (matching the legacy
# wrapper's `t_right = t[..., end_index:]` behavior). The fp32-CPU oracle
# test (Test 3) uses dim=192 where rot_d == head_dim and the partial-rope
# path collapses to a single apply_rope1 call. This test exercises the
# partial path explicitly with dim=128 and asserts the rewired forward
# still tensor-equals the wrapper oracle in that regime.

def test_namm_forward_partial_rope_passthrough_matches_wrapper_oracle():
    rope = NaMMRotaryEmbedding3d(dim=128)
    g = torch.Generator(device="cpu").manual_seed(_SEED)
    vid_q = torch.randn(_L_VID, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    vid_k = torch.randn(_L_VID, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    txt_q = torch.randn(_TXT_L, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    txt_k = torch.randn(_TXT_L, _HEADS, 128, dtype=torch.float32, device="cpu", generator=g)
    vid_shape = torch.tensor([[_VID_T, _VID_H, _VID_W]], dtype=torch.long)
    txt_shape = torch.tensor([[_TXT_L]], dtype=torch.long)
    cache = Cache(disable=True)

    expected_vid_q, expected_vid_k, expected_txt_q, expected_txt_k = _legacy_forward(
        rope, vid_q.clone(), vid_k.clone(), vid_shape, txt_q.clone(), txt_k.clone(), txt_shape,
    )
    actual_vid_q, actual_vid_k, actual_txt_q, actual_txt_k = rope.forward(
        vid_q.clone(), vid_k.clone(), vid_shape, txt_q.clone(), txt_k.clone(), txt_shape, cache,
    )

    # Confirm the partial-rope contract: rot_d (= 2 * freqs_cis.shape[-3]) is
    # 126 (= 42*3), strictly less than head_dim 128. The trailing 2 head-dims
    # are pass-through.
    vid_freqs, _ = rope.get_freqs(vid_shape, txt_shape)
    rot_d = 2 * vid_freqs.shape[-3]
    assert rot_d == 126, f"expected rot_d=126 for dim=128 model; got {rot_d}"
    assert rot_d < 128, "partial-rope path must trigger (rot_d < head_dim)"

    torch.testing.assert_close(actual_vid_q, expected_vid_q, rtol=0, atol=0,
                                msg="vid_q partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_vid_k, expected_vid_k, rtol=0, atol=0,
                                msg="vid_k partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_q, expected_txt_q, rtol=0, atol=0,
                                msg="txt_q partial-rope output diverges from wrapper oracle")
    torch.testing.assert_close(actual_txt_k, expected_txt_k, rtol=0, atol=0,
                                msg="txt_k partial-rope output diverges from wrapper oracle")


# Test 4 — drives AC-4 statically: AST walk over NaMMRotaryEmbedding3d.forward
# must find zero references to the apply_rotary_emb symbol.

def test_namm_forward_ast_has_no_apply_rotary_emb_calls():
    source_path = Path(inspect.getsourcefile(NaMMRotaryEmbedding3d))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    namm_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "NaMMRotaryEmbedding3d":
            namm_class = node
            break
    assert namm_class is not None, (
        f"could not locate class NaMMRotaryEmbedding3d in {source_path}"
    )

    forward_fn = None
    for node in namm_class.body:
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            forward_fn = node
            break
    assert forward_fn is not None, (
        "could not locate NaMMRotaryEmbedding3d.forward"
    )

    offending = []
    for node in ast.walk(forward_fn):
        if isinstance(node, ast.Name) and node.id == "apply_rotary_emb":
            offending.append((node.lineno, node.col_offset))

    assert not offending, (
        f"NaMMRotaryEmbedding3d.forward must not reference apply_rotary_emb; "
        f"found {len(offending)} reference(s) at line:col positions {offending}. "
        f"The rewrite must rewire to apply_rope1 directly."
    )

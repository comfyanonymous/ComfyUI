"""Regression test for SparseRotaryPositionEmbedder RoPE shape handling.

Accelerated RoPE backends (e.g. comfy-kitchen's triton kernel, which every
ComfyUI install depends on via requirements.txt) unpack their input tensor as
exactly (batch, dim1, dim2, head_dim) -- see comfy_kitchen/backends/triton/rope.py.
Trellis2's sparse attention path (shape/texture generation) feeds it
[N, heads, head_dim], 3-D with no explicit batch axis, which used to raise
"ValueError: not enough values to unpack (expected 4, got 3)" (issue #16028).
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.quant_ops as quant_ops  # noqa: E402
from comfy.ldm.flux.math import _apply_rope1  # noqa: E402
from comfy.ldm.trellis2.model import SparseRotaryPositionEmbedder  # noqa: E402
from comfy.ldm.trellis2.vae import SparseTensor  # noqa: E402


class _StrictCK:
    """Stand-in for comfy_kitchen's triton backend: requires 4-D tensors."""

    @staticmethod
    def apply_rope1(x, freqs_cis):
        _batch, _dim1, _dim2, _head_dim = x.shape  # raises ValueError if x isn't 4-D
        return _apply_rope1(x, freqs_cis)

    @staticmethod
    def apply_rope(xq, xk, freqs_cis):
        return _StrictCK.apply_rope1(xq, freqs_cis), _StrictCK.apply_rope1(xk, freqs_cis)


def _make_sparse_qk(num_tokens: int, heads: int, head_dim: int, resolution: int = 4):
    coords = torch.stack(
        [
            torch.zeros(num_tokens, dtype=torch.int32),
            torch.randint(0, resolution, (num_tokens,), dtype=torch.int32),
            torch.randint(0, resolution, (num_tokens,), dtype=torch.int32),
            torch.randint(0, resolution, (num_tokens,), dtype=torch.int32),
        ],
        dim=1,
    )
    q_feats = torch.randn(num_tokens, heads, head_dim)
    k_feats = torch.randn(num_tokens, heads, head_dim)
    q = SparseTensor(feats=q_feats, coords=coords, shape=torch.Size([1, heads, head_dim]))
    k = SparseTensor(feats=k_feats, coords=coords, shape=torch.Size([1, heads, head_dim]))
    return q, k


def test_sparse_rope_works_with_4d_only_backend():
    heads, head_dim, num_tokens = 3, 16, 5
    rope = SparseRotaryPositionEmbedder(head_dim=head_dim, dim=3)
    q, k = _make_sparse_qk(num_tokens=num_tokens, heads=heads, head_dim=head_dim)

    with patch.object(quant_ops, "ck", _StrictCK(), create=True):
        q_out, k_out = rope(q, k)

    assert q_out.feats.shape == (num_tokens, heads, head_dim)
    assert k_out.feats.shape == (num_tokens, heads, head_dim)


def test_sparse_rope_single_input_works_with_4d_only_backend():
    heads, head_dim, num_tokens = 2, 8, 4
    rope = SparseRotaryPositionEmbedder(head_dim=head_dim, dim=3)
    q, _ = _make_sparse_qk(num_tokens=num_tokens, heads=heads, head_dim=head_dim)

    with patch.object(quant_ops, "ck", _StrictCK(), create=True):
        q_out = rope(q)

    assert q_out.feats.shape == (num_tokens, heads, head_dim)

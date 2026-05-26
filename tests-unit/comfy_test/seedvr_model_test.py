"""Regression tests for SeedVR2 conditioning split hardening.

Two bare ``except:`` clauses in ``NaDiT.forward`` previously swallowed
every failure mode on (1) the input-side text-conditioning split and
(2) the output-side positive/negative split, silently substituting
wrong fallbacks: the ``positive_conditioning`` buffer (which prior to
explicit zero-init held **uninitialized** memory — NaN, residual heap
contents, never guaranteed-zero) for the input, and the un-split
tensor for the output. Real prompt-shape, dtype, OOM, and downstream
tensor failures were re-routed to "no prompt supplied" with arbitrary
buffer contents standing in for actual prompt embeddings, or to a
wrong-order output, with no diagnostic.

The fix:

  1. Input-side: explicit absence predicate (``context is None`` or
     ``context.numel() == 0``) → fall back to ``positive_conditioning``
     buffer. Any other failure (wrong rank, odd batch, dtype, OOM)
     propagates the original torch exception.
  2. Output-side: no try/except at all. ``out.chunk(2)`` of the
     network output is a contract: an unsplittable result is a bug,
     not a recoverable condition.

The two blocks were extracted into named private methods on
``NaDiT`` (``_resolve_text_conditioning`` and ``_swap_pos_neg_halves``)
so the regression evidence drives the actual production code paths
without standing up a full transformer. The methods are called from
``forward`` exactly where the original try/except blocks lived.
"""

from comfy.cli_args import args
import torch

if not torch.cuda.is_available():
    args.cpu = True

import ast  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402

import pytest  # noqa: E402

from comfy.ldm.seedvr.model import NaDiT  # noqa: E402


def _make_standin(positive_conditioning):
    class _StandIn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "positive_conditioning", positive_conditioning
            )

        _resolve_text_conditioning = NaDiT._resolve_text_conditioning
        _swap_pos_neg_halves = NaDiT._swap_pos_neg_halves

    return _StandIn()


def test_no_bare_except_in_forward_path():
    """Source-level pin: neither ``NaDiT.forward`` nor its split helpers
    may carry the bare ``except:`` clauses that swallowed real torch
    failures on the SeedVR2 conditioning paths. AST-walked rather than
    substring-matched so that ``except:`` appearing in a docstring or
    comment does not false-positive, and so that ``except Exception:``
    (a typed handler, fine to have) does not false-negative.
    """
    sources = [
        inspect.getsource(NaDiT.forward),
        inspect.getsource(NaDiT._resolve_text_conditioning),
        inspect.getsource(NaDiT._swap_pos_neg_halves),
    ]
    for src in sources:
        tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                assert node.type is not None, (
                    "Bare 'except:' (ast.ExceptHandler with type=None) "
                    f"must not appear on the SeedVR2 forward path:\n{src}"
                )


def test_valid_context_splits_pos_neg():
    """AC: valid (neg, pos)-stacked context (shape ``(2, L, C)``)
    produces a flattened ``[pos, neg]`` text tensor — first ``L`` rows
    are positive, next ``L`` rows are negative — matching the original
    semantics of the ``flatten([pos_cond, neg_cond])`` call.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    seq_len, channels = 7, 5120
    neg = torch.full((1, seq_len, channels), -1.0)
    pos = torch.full((1, seq_len, channels), 1.0)
    context = torch.cat([neg, pos], dim=0)
    txt, txt_shape = standin._resolve_text_conditioning(context)
    assert txt.shape == (2 * seq_len, channels)
    assert (txt[:seq_len] == 1.0).all(), "first half must be positive cond"
    assert (txt[seq_len:] == -1.0).all(), "second half must be negative cond"
    assert txt_shape.shape == (2, 1)
    assert txt_shape[0].item() == seq_len
    assert txt_shape[1].item() == seq_len


def test_missing_context_falls_back_to_positive_buffer():
    """AC: ``context is None`` falls back to the registered
    ``positive_conditioning`` buffer and runs to completion — no
    silent zero substitution, no raised exception.
    """
    pos_buffer = torch.full((58, 5120), 7.0)
    standin = _make_standin(pos_buffer)
    txt, txt_shape = standin._resolve_text_conditioning(None)
    assert txt.shape == (58, 5120)
    assert (txt == 7.0).all(), (
        "fallback path must use the positive_conditioning buffer "
        "verbatim, not a zero tensor"
    )
    assert txt_shape.shape == (1, 1)
    assert txt_shape[0, 0].item() == 58


def test_empty_context_falls_back_to_positive_buffer():
    """AC: ``context.numel() == 0`` falls back to the registered
    ``positive_conditioning`` buffer and runs to completion.
    """
    pos_buffer = torch.full((58, 5120), 13.0)
    standin = _make_standin(pos_buffer)
    empty = torch.empty((0, 5120))
    assert empty.numel() == 0
    txt, txt_shape = standin._resolve_text_conditioning(empty)
    assert txt.shape == (58, 5120)
    assert (txt == 13.0).all()
    assert txt_shape.shape == (1, 1)
    assert txt_shape[0, 0].item() == 58


def test_wrong_rank_context_raises_original_torch_exception():
    """AC: a 1-D context tensor cannot be split into ``[pos, neg]``
    via the ``chunk + squeeze + flatten`` chain; the original torch
    exception must propagate rather than silently falling back.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    bad = torch.zeros(10)
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        standin._resolve_text_conditioning(bad)


def test_odd_batch_context_raises_original_exception():
    """AC: a context whose batch dim cannot be split into two equal
    chunks (here batch=1 so ``chunk(2, dim=0)`` returns a single
    tensor) must propagate the original exception — no silent fallback.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    bad = torch.zeros((1, 7, 5120))
    with pytest.raises((RuntimeError, ValueError)):
        standin._resolve_text_conditioning(bad)


def test_output_side_misshaped_tensor_raises():
    """AC: the post-network output split must raise on an unsplittable
    tensor (no silent return of the un-split tensor in the wrong
    order/shape). Here a batch=1 tensor cannot be ``chunk(2, dim=0)``
    into two halves; ``pos, neg = out.chunk(2, dim=0)`` raises on
    unpacking — matching the production helper's explicit-dim contract
    (``_swap_pos_neg_halves`` calls ``chunk(2, dim=0)`` and
    ``torch.cat(..., dim=0)``).
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    bad_out = torch.zeros((1, 4, 8, 8))
    with pytest.raises((RuntimeError, ValueError)):
        standin._swap_pos_neg_halves(bad_out)


def test_output_side_swaps_pos_neg_halves():
    """AC complement: ``_swap_pos_neg_halves`` reorders the post-network
    output so the first half (positive) and second half (negative) trade
    places. For a 2-batch tensor with distinguishable halves, the
    returned tensor must be the swap — first half becomes negative,
    second half becomes positive — matching the original
    ``torch.cat([neg, pos])`` semantics from the pre-fix forward path.
    """
    pos_buffer = torch.zeros((58, 5120))
    standin = _make_standin(pos_buffer)
    pos_half = torch.full((1, 4, 8, 8), 1.0)
    neg_half = torch.full((1, 4, 8, 8), -1.0)
    out = torch.cat([pos_half, neg_half], dim=0)
    swapped = standin._swap_pos_neg_halves(out)
    assert swapped.shape == out.shape
    assert (swapped[0] == -1.0).all(), "first half of swapped output must be the original negative half"
    assert (swapped[1] == 1.0).all(), "second half of swapped output must be the original positive half"

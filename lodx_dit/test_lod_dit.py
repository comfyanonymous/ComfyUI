"""Acceptance tests for the bidirectional LoD read.

The anchor is the same one the LLM port uses: with ``top_pages >= n_pages``
tier S is empty and the read must equal dense attention.  If that fails, no
other number in this package means anything.  Everything else here exists so
the anchor cannot pass for an accidental reason -- the exact prefix, the
remainder rows, the forced local pages and the block pooling each have to
leave every token in the denominator exactly once.

No pytest: this runs on a bare interpreter so it can be checked anywhere the
model runs.

    python lodx_dit/test_lod_dit.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lodx_dit.lod_dit import (PagedLayout, lod_attention, lod_attention_ref,
                              page_sums, select_pages)

CUDA = torch.cuda.is_available()
CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def gpu_case(fn):
    fn.needs_cuda = True
    CASES.append(fn)
    return fn


def dense(query, key, value, scale=None):
    """Bidirectional dense attention in fp32 -- no mask of any kind."""
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    logits = torch.einsum("bhqd,bhkd->bhqk", query.float(),
                          key.float()) * scale
    return torch.einsum("bhqk,bhkd->bhqd", logits.softmax(-1), value.float())


def qkv(b=1, h=4, s=256, d=32, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda: torch.randn(b, h, s, d, generator=g).to(device=device,
                                                           dtype=dtype)
    return make(), make(), make()


def rel_err(got, want):
    return ((got.float() - want.float()).norm()
            / want.float().norm().clamp_min(1e-30)).item()


# --------------------------------------------------------------- the anchor

@case
def full_expansion_equals_dense():
    """kP >= n_pages: tier S empty, leaves + exact rows cover [0, S) once."""
    for s, prefix, ps in [(256, 0, 64),      # no prefix, exact multiple
                          (256, 64, 64),     # prefix, exact multiple
                          (300, 44, 64),     # remainder rows at the end
                          (200, 200, 64),    # all prefix, no pages at all
                          (129, 1, 32)]:     # odd sizes, tiny prefix
        q, k, v = qkv(s=s)
        layout = PagedLayout(s, prefix, ps)
        got = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps,
                                top_pages=max(layout.n_pages, 1),
                                select_block=32)
        err = (got - dense(q, k, v)).abs().max().item()
        assert err < 2e-6, f"S={s} prefix={prefix} ps={ps}: max|d|={err:.2e}"


@case
def full_expansion_ignores_local_forcing():
    """Forcing changes WHICH pages are leaves, never the identity."""
    s, prefix, ps = 320, 64, 64
    q, k, v = qkv(s=s)
    layout = PagedLayout(s, prefix, ps)
    for radius in (-1, 0, 1):
        got = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps,
                                top_pages=layout.n_pages, select_block=32,
                                local_radius=radius)
        err = (got - dense(q, k, v)).abs().max().item()
        assert err < 2e-6, f"radius={radius}: max|d|={err:.2e}"


@case
def partial_budget_differs_from_dense():
    """A real budget must actually approximate, or the anchor is vacuous."""
    s, prefix, ps = 1024, 64, 64
    q, k, v = qkv(s=s, seed=3)
    got = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps, top_pages=2,
                            select_block=32)
    err = (got - dense(q, k, v)).abs().max().item()
    assert err > 1e-4, "budget 2 of 15 pages produced no visible difference"


# ------------------------------------------------------------- book-keeping

@case
def page_sums_match_a_plain_loop():
    s, prefix, ps = 320, 64, 64
    _, k, v = qkv(s=s)
    layout = PagedLayout(s, prefix, ps)
    ks, vs = page_sums(k, v, layout)
    for p in range(layout.n_pages):
        lo = prefix + p * ps
        assert torch.allclose(ks[:, :, p], k[:, :, lo:lo + ps].float().sum(2),
                              atol=1e-5)
        assert torch.allclose(vs[:, :, p], v[:, :, lo:lo + ps].float().sum(2),
                              atol=1e-5)


@case
def layout_covers_every_row_exactly_once():
    for s, prefix, ps in [(256, 0, 64), (300, 44, 64), (129, 1, 32),
                          (38010, 714, 64)]:
        layout = PagedLayout(s, prefix, ps)
        assert layout.exact_len + layout.n_pages * ps == s, repr(layout)


@case
def forced_pages_are_selected():
    s, prefix, ps, blk = 1024, 64, 64, 32
    q, k, v = qkv(s=s, seed=5)
    layout = PagedLayout(s, prefix, ps)
    ks, _ = page_sums(k, v, layout)
    sel, _ = select_pages(q, ks, layout, top_pages=2, block=blk,
                          local_radius=0)
    for g in range(sel.size(1)):
        lo = g * blk
        if lo < prefix or lo >= layout.paged_end:
            continue
        own = (lo - prefix) // ps
        assert own in sel[0, g].tolist(), f"block {g} lost its own page {own}"


@case
def selection_chunking_is_transparent():
    s, prefix, ps = 2048, 64, 64
    q, k, v = qkv(s=s, seed=7)
    layout = PagedLayout(s, prefix, ps)
    ks, _ = page_sums(k, v, layout)
    a, _ = select_pages(q, ks, layout, 8, block=32, chunk_blocks=2)
    b, _ = select_pages(q, ks, layout, 8, block=32, chunk_blocks=1000)
    assert torch.equal(a, b)


@case
def ties_break_toward_the_lower_page_id():
    s, ps = 512, 64
    q = torch.zeros(1, 2, s, 16)          # every score identical
    k = torch.ones(1, 2, s, 16)
    v = torch.randn(1, 2, s, 16)
    layout = PagedLayout(s, 0, ps)
    ks, _ = page_sums(k, v, layout)
    sel, _ = select_pages(q, ks, layout, 3, block=32, local_radius=-1)
    assert sel[0, 0].tolist() == [0, 1, 2], sel[0, 0].tolist()


# ---------------------------------------------------------------- ordering

@case
def reordering_is_an_identity_at_full_expansion():
    """Permuting q/k/v together relabels the key set; softmax cannot notice."""
    from lodx_dit.ordering import sequence_order
    grid = (4, 8, 10)                       # 320 video rows
    prefix, ps = 64, 64
    s = prefix + grid[0] * grid[1] * grid[2]
    q, k, v = qkv(s=s, seed=21)
    order = sequence_order(s, prefix, grid, (1, 4, 4))
    layout = PagedLayout(s, prefix, ps)
    got = lod_attention_ref(q, k, v, order=order, prefix=prefix, page_size=ps,
                            top_pages=layout.n_pages, select_block=32)
    err = (got - dense(q, k, v)).abs().max().item()
    assert err < 2e-6, f"max|d|={err:.2e}"


@case
def reordering_is_a_permutation():
    from lodx_dit.ordering import invert_order, sequence_order, tile_order
    grid = (3, 24, 42)                      # H3's patch grid, non-divisible w
    order = tile_order(grid, (1, 8, 8))
    n = grid[0] * grid[1] * grid[2]
    assert torch.equal(order.sort().values, torch.arange(n))
    assert torch.equal(invert_order(order)[order], torch.arange(n))
    seq = sequence_order(700 + n, 700, grid, (1, 8, 8))
    assert torch.equal(seq.sort().values, torch.arange(700 + n))
    assert torch.equal(seq[:700], torch.arange(700))


@case
def tiling_shrinks_the_page_bounding_box():
    """The point of the reorder: what a page sum has to stand for."""
    from lodx_dit.ordering import best_tile, page_extent, tile_order
    for grid in [(37, 24, 42), (107, 24, 42), (37, 24, 24)]:
        tile, ps = best_tile(grid, 64)
        raster = page_extent(grid, None, 64)
        tiled = page_extent(grid, None, ps, order=tile_order(grid, tile))
        # an exact tile has a bounding box equal to its own area, by definition
        assert tiled == ps, f"{grid}: tile {tile} not exact ({tiled} vs {ps})"
        assert tiled <= raster, f"{grid}: raster {raster:.0f} vs {tiled:.0f}"


@case
def a_tile_that_does_not_divide_is_worse_than_raster():
    """Why best_tile exists -- the naive (1,8,8) loses on H3's 42-wide grid."""
    from lodx_dit.ordering import page_extent, tile_order
    grid = (37, 24, 42)
    raster = page_extent(grid, None, 64)
    naive = page_extent(grid, None, 64, order=tile_order(grid, (1, 8, 8)))
    assert naive > raster, f"raster {raster:.0f} vs naive tile {naive:.0f}"


# ------------------------------------------------------------------ kernel

@gpu_case
def kernel_matches_reference():
    """Same selection handed to both, so this measures the kernel alone."""
    s, prefix, ps, blk = 1024, 64, 64, 32
    for head_dim in (128, 256):
        q, k, v = qkv(b=1, h=4, s=s, d=head_dim, dtype=torch.bfloat16,
                      device="cuda", seed=11)
        layout = PagedLayout(s, prefix, ps)
        ks, _ = page_sums(k, v, layout)
        selection = select_pages(q, ks, layout, 8, block=blk)
        ref = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps,
                                top_pages=8, select_block=blk,
                                selection=selection)
        fast = lod_attention(q, k, v, prefix=prefix, page_size=ps, top_pages=8,
                             select_block=blk, selection=selection)
        err = rel_err(fast, ref)
        assert err < 5e-3, f"D={head_dim}: rel={err:.2e}"


@gpu_case
def kernel_full_expansion_equals_dense():
    s, prefix, ps, blk = 512, 64, 64, 32
    q, k, v = qkv(b=1, h=4, s=s, d=128, dtype=torch.bfloat16, device="cuda",
                  seed=13)
    layout = PagedLayout(s, prefix, ps)
    fast = lod_attention(q, k, v, prefix=prefix, page_size=ps,
                         top_pages=layout.n_pages, select_block=blk)
    err = rel_err(fast, dense(q, k, v))
    assert err < 5e-3, f"rel={err:.2e}"


@gpu_case
def kernel_handles_a_budget_not_dividing_the_leaf_tile():
    """KP*ps need not divide BN; reading past the list invents phantom leaves."""
    s, prefix, ps, blk = 1024, 0, 64, 32
    q, k, v = qkv(b=1, h=4, s=s, d=128, dtype=torch.bfloat16, device="cuda",
                  seed=17)
    layout = PagedLayout(s, prefix, ps)
    ks, _ = page_sums(k, v, layout)
    selection = select_pages(q, ks, layout, 3, block=blk)  # 3*64=192, BN=128
    ref = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps, top_pages=3,
                            select_block=blk, selection=selection)
    fast = lod_attention(q, k, v, prefix=prefix, page_size=ps, top_pages=3,
                         select_block=blk, selection=selection)
    err = rel_err(fast, ref)
    assert err < 5e-3, f"rel={err:.2e}"


@gpu_case
def kernel_handles_h3_head_count():
    """56 heads, no GQA -- H3's shape, at a size the reference can still hold."""
    s, prefix, ps, blk = 2048, 128, 64, 32
    q, k, v = qkv(b=1, h=56, s=s, d=128, dtype=torch.bfloat16, device="cuda",
                  seed=19)
    layout = PagedLayout(s, prefix, ps)
    ks, _ = page_sums(k, v, layout)
    selection = select_pages(q, ks, layout, 8, block=blk)
    ref = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps, top_pages=8,
                            select_block=blk, selection=selection)
    fast = lod_attention(q, k, v, prefix=prefix, page_size=ps, top_pages=8,
                         select_block=blk, selection=selection)
    err = rel_err(fast, ref)
    assert err < 5e-3, f"rel={err:.2e}"


# ----------------------------------------------------- the ComfyUI ablation

def _wrapper_setup(top_pages, tiled=True, prefix=64, grid=(4, 8, 7), heads=4,
                   mode="lod", contiguous=False):
    """Drive the patched optimized_attention the way the DiT block does."""
    from comfy.ldm.modules.attention import AttentionTensorContainer
    from lodx_dit.comfy_node import _install
    import comfy.ldm.minimax.model as h3

    _install()
    video_rows = grid[0] * grid[1] * grid[2]
    s = prefix + video_rows
    q, k, v = qkv(1, heads, s, 128, torch.bfloat16, "cuda", seed=23)
    to = {"lod": dict(mode=mode, contiguous_qkv=contiguous,
                      kernel_variant="default",
                      top_pages=top_pages, select_block=32, page_size=64,
                      local_radius=0, tiled_pages=tiled,
                      start_percent=0.0, end_percent=1.0,
                      sigma_start=1.0, sigma_end=0.0),
          "_lod_grid": grid}
    c = AttentionTensorContainer
    out = h3.optimized_attention(c(q), c(k), c(v), heads, mask=None,
                                 skip_reshape=True, transformer_options=to)
    return q, k, v, s, out


@gpu_case
def comfy_wrapper_equals_dense_at_full_expansion():
    """The ablation is only meaningful if the two branches differ in one thing."""
    for tiled in (True, False):
        q, k, v, s, out = _wrapper_setup(10 ** 6, tiled=tiled)
        ref = dense(q, k, v).transpose(1, 2).reshape(1, s, -1)
        err = rel_err(out, ref)
        assert err < 5e-3, f"tiled={tiled}: rel={err:.2e}"
        assert out.shape == ref.shape, (out.shape, ref.shape)


@gpu_case
def comfy_wrapper_approximates_at_a_real_budget():
    """...and it must actually be doing the sparse read, not silently falling back."""
    q, k, v, s, out = _wrapper_setup(1)
    ref = dense(q, k, v).transpose(1, 2).reshape(1, s, -1)
    err = rel_err(out, ref)
    assert err > 1e-3, f"budget 1 of 4 pages changed nothing (rel={err:.2e})"


@gpu_case
def comfy_wrapper_is_a_passthrough_without_config():
    """Bypassing the node must give the untouched dense path, bit for bit."""
    from comfy.ldm.modules.attention import AttentionTensorContainer
    from lodx_dit.comfy_node import _install
    import comfy.ldm.minimax.model as h3

    _install()
    q, k, v = qkv(1, 4, 512, 128, torch.bfloat16, "cuda", seed=29)
    c = AttentionTensorContainer
    got = h3.optimized_attention(c(q), c(k), c(v), 4, mask=None,
                                 skip_reshape=True, transformer_options={})
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False).transpose(1, 2).reshape(1, 512, -1)
    assert torch.equal(got, want), rel_err(got, want)


@gpu_case
def comfy_wrapper_leaves_the_token_refiner_dense():
    """The refiner runs the same Attention over the text rows only."""
    from comfy.ldm.modules.attention import AttentionTensorContainer
    from lodx_dit.comfy_node import _install
    import comfy.ldm.minimax.model as h3

    _install()
    grid = (4, 8, 7)
    q, k, v = qkv(1, 4, 100, 128, torch.bfloat16, "cuda", seed=31)
    to = {"lod": dict(mode="lod", contiguous_qkv=False,
                      kernel_variant="default", top_pages=1,
                      select_block=32, page_size=64, local_radius=0,
                      tiled_pages=True, start_percent=0.0, end_percent=1.0,
                      sigma_start=1.0, sigma_end=0.0),
          "_lod_grid": grid}          # 100 rows < 224 video rows
    c = AttentionTensorContainer
    got = h3.optimized_attention(c(q), c(k), c(v), 4, mask=None,
                                 skip_reshape=True, transformer_options=to)
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False).transpose(1, 2).reshape(1, 100, -1)
    assert torch.equal(got, want), rel_err(got, want)


@gpu_case
def comfy_dense_mode_equals_the_stock_read():
    """mode=dense must be the untouched path, or the A/B has two variables."""
    q, k, v, s, out = _wrapper_setup(128, mode="dense", contiguous=False)
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False).transpose(1, 2).reshape(1, s, -1)
    assert torch.equal(out, want), rel_err(out, want)


@gpu_case
def comfy_contiguous_option_does_not_change_the_answer():
    """It is a layout change; it may pick another SDPA kernel but not another result."""
    _, _, _, _, a = _wrapper_setup(128, mode="dense", contiguous=False)
    _, _, _, _, b = _wrapper_setup(128, mode="dense", contiguous=True)
    err = rel_err(b, a)
    assert err < 5e-3, err


@gpu_case
def comfy_contiguous_option_applies_in_both_modes():
    """Otherwise switching mode would also switch the layout fix on or off."""
    for mode in ("dense", "lod"):
        _, _, _, _, a = _wrapper_setup(1, mode=mode, contiguous=False)
        _, _, _, _, b = _wrapper_setup(1, mode=mode, contiguous=True)
        err = rel_err(b, a)
        assert err < 5e-3, f"mode={mode}: rel={err:.2e}"


@gpu_case
def every_kernel_variant_matches_the_reference():
    """A variant only counts if it still computes the same read."""
    from lodx_dit.kernel_exp import VARIANTS
    s, prefix, ps, blk = 2048, 128, 64, 64
    q, k, v = qkv(b=1, h=8, s=s, d=128, dtype=torch.bfloat16, device="cuda",
                  seed=37)
    layout = PagedLayout(s, prefix, ps)
    ks, _ = page_sums(k, v, layout)
    selection = select_pages(q, ks, layout, 8, block=blk)
    ref = lod_attention_ref(q, k, v, prefix=prefix, page_size=ps, top_pages=8,
                            select_block=blk, selection=selection)
    for name in VARIANTS:
        got = lod_attention(q, k, v, prefix=prefix, page_size=ps, top_pages=8,
                            select_block=blk, selection=selection,
                            variant=name)
        err = rel_err(got, ref)
        assert err < 2e-2, f"{name}: rel={err:.2e}"


@gpu_case
def every_kernel_variant_keeps_the_dense_anchor():
    """Full expansion must still be dense, whatever the accumulator does."""
    from lodx_dit.kernel_exp import VARIANTS
    s, prefix, ps, blk = 1024, 128, 64, 64
    q, k, v = qkv(b=1, h=8, s=s, d=128, dtype=torch.bfloat16, device="cuda",
                  seed=41)
    layout = PagedLayout(s, prefix, ps)
    want = dense(q, k, v)
    for name in VARIANTS:
        got = lod_attention(q, k, v, prefix=prefix, page_size=ps,
                            top_pages=layout.n_pages, select_block=blk,
                            variant=name)
        err = rel_err(got, want)
        assert err < 2e-2, f"{name}: rel={err:.2e}"


def main():
    passed = failed = skipped = 0
    for fn in CASES:
        name = fn.__name__
        if getattr(fn, "needs_cuda", False) and not CUDA:
            print(f"  SKIP  {name} (no GPU)")
            skipped += 1
            continue
        try:
            fn()
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"  ok    {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

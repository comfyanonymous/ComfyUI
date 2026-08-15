"""LoD sparse attention for a bidirectional DiT.

Ported from ``Qwen3.5-2B-Lodx`` (which ports llama.cpp's LoD).  The read is the
same refinement -- one softmax over selected leaves, unselected page summaries,
and an exactly-read region -- but a diffusion transformer is not an
autoregressive decoder, so three things change:

no cache
    K/V are rebuilt every layer of every sampling step and never outlive the
    forward.  ``LodxCache`` and its whole state machine (append, catch_up,
    truncate, the sums watermark, the suffix-removal zeroing) are gone; page
    sums are a segmented reduction over the K/V that were just computed.  The
    two headline pitfalls of the LLM port -- deriving ``n_pages`` from the
    cache length, and rewinding the watermark without zeroing -- cannot be
    expressed here.

no causal tail, an exact PREFIX instead
    Tier E existed because the newest tokens have no complete page yet.  A DiT
    has no "newest".  What it does have is a contiguous prefix of conditioning
    rows -- in MiniMax H3 the packed sequence is
    ``[text | cond | ref | audio | video]`` -- that must never be summarised:
    losing the text rows loses prompt adherence.  So tier E becomes tier X, the
    same contiguous flash-friendly slice, merged by the same LSE, just at the
    front instead of the back.  Any rows left over after the last complete page
    join it.

forced local pages
    In the LLM, tier E guarantees a query always reads its own neighbourhood
    exactly.  Nothing guarantees that here: a video row sits *inside* a page
    that selection may not pick, and video attention is locality-dominated.
    So the pages a query block occupies are forced into its selection.  This
    cannot break the dense anchor -- when ``top_pages >= n_pages`` every page
    is selected either way.

The full-expansion invariant is unchanged and is still the only thing holding
the rest up: with ``top_pages >= n_pages`` tier S is empty, the leaves plus the
exact rows cover ``[0, S)`` exactly once, and the read is dense attention.
"""

from __future__ import annotations

import math

import torch

__all__ = ["page_sums", "select_pages", "lod_attention_ref", "lod_attention",
           "PagedLayout", "reordered"]


def reordered(fn):
    """Run ``fn`` with the sequence axis permuted by ``order``.

    Permuting q, k and v together is an exact identity for attention -- softmax
    does not care what order the keys arrive in -- so this changes only which
    rows share a page, never the dense result.  See ordering.py for why that
    matters: H3 emits video rows in raster order, where a 64-row page is a
    strip 42 wide and 2 tall rather than a compact block.
    """
    import functools

    inverses = {}

    @functools.wraps(fn)
    def wrapper(query, key, value, *, order=None, **kw):
        if order is None:
            return fn(query, key, value, **kw)
        from .ordering import invert_order
        # the inverse is a pure function of the order and the order is built
        # once per shape, so rebuilding it per call is a scatter for nothing
        key_ = (id(order), order.numel(), str(order.device))
        inv = inverses.get(key_)
        if inv is None:
            inv = inverses[key_] = invert_order(order)
        out = fn(query.index_select(2, order), key.index_select(2, order),
                 value.index_select(2, order), **kw)
        return out.index_select(2, inv)
    return wrapper


class PagedLayout:
    """Which rows are paged and which are read exactly.

    ``prefix`` rows at the front are always exact.  The rest is cut into
    complete pages of ``page_size``; a remainder shorter than a page cannot be
    summarised (there is no half-full summary in this design) so it is exact
    too.
    """

    def __init__(self, seq_len: int, prefix: int, page_size: int):
        if not 0 <= prefix <= seq_len:
            raise ValueError("prefix outside the sequence")
        self.seq_len = seq_len
        self.prefix = prefix
        self.page_size = page_size
        self.n_pages = (seq_len - prefix) // page_size
        self.paged_end = prefix + self.n_pages * page_size

    @property
    def exact_len(self) -> int:
        return self.prefix + (self.seq_len - self.paged_end)

    def exact_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """The exactly-read rows of a (B, H, S, D) tensor, front then tail."""
        head = tensor[:, :, :self.prefix]
        tail = tensor[:, :, self.paged_end:]
        if tail.size(2) == 0:
            return head
        if head.size(2) == 0:
            return tail
        return torch.cat([head, tail], dim=2)

    def __repr__(self) -> str:
        return ("PagedLayout(S={}, prefix={}, ps={}, pages={}, exact={})"
                .format(self.seq_len, self.prefix, self.page_size,
                        self.n_pages, self.exact_len))


def page_sums(key: torch.Tensor, value: torch.Tensor, layout: PagedLayout):
    """Sum raw K/V over each complete page.  Returns fp32 (B, H, P, D) pairs.

    Pages are contiguous and complete by construction, so this is a reshape and
    a reduction -- the LLM port needed ``scatter_add_`` only because tokens
    arrived a few at a time.
    """
    ps, n = layout.page_size, layout.n_pages
    if n == 0:
        b, h, _, d = key.shape
        empty = key.new_zeros(b, h, 0, d, dtype=torch.float32)
        return empty, value.new_zeros(b, h, 0, value.size(-1),
                                      dtype=torch.float32)
    b, h = key.shape[0], key.shape[1]
    ks = key[:, :, layout.prefix:layout.paged_end]
    vs = value[:, :, layout.prefix:layout.paged_end]
    # accumulate in fp32 rather than casting first: .float() would materialise
    # the whole paged region at 4 bytes a value -- 1.2 GB at a 15 s clip -- only
    # to reduce it away
    ks = ks.reshape(b, h, n, ps, ks.size(-1)).sum(3, dtype=torch.float32)
    vs = vs.reshape(b, h, n, ps, vs.size(-1)).sum(3, dtype=torch.float32)
    return ks, vs


def _forced_pages(layout: PagedLayout, n_blocks: int, block: int,
                  radius: int, device) -> torch.Tensor:
    """(n_blocks, n_pages) mask of pages a query block must read at leaf detail.

    A block that lies entirely in the exact prefix has no page of its own and
    forces nothing; its selection is decided by score alone.
    """
    n = layout.n_pages
    starts = torch.arange(n_blocks, device=device) * block
    ends = (starts + block - 1).clamp(max=layout.seq_len - 1)
    lo = ((starts - layout.prefix) // layout.page_size).clamp(0, n - 1) - radius
    hi = ((ends - layout.prefix) // layout.page_size).clamp(0, n - 1) + radius
    overlaps = (ends >= layout.prefix) & (starts < layout.paged_end)
    pages = torch.arange(n, device=device)
    return (overlaps[:, None]
            & (pages[None, :] >= lo[:, None])
            & (pages[None, :] <= hi[:, None]))


def _block_scores_torch(query, sums_k, n, block, n_blocks, chunk_blocks):
    """Reference for :func:`select_pages`; used when triton is unavailable."""
    b, h, s, _ = query.shape
    sums = sums_k.to(query.dtype)
    pooled = torch.empty(b, n_blocks, n, dtype=torch.float32,
                         device=query.device)
    step = chunk_blocks * block
    for lo in range(0, s, step):
        hi = min(lo + step, s)
        qs = query[:, :, lo:hi]
        nb = (hi - lo + block - 1) // block
        pad = nb * block - (hi - lo)
        scores = torch.einsum("bhqd,bhpd->bhpq", qs, sums)
        if pad:
            scores = torch.cat(
                [scores, scores.new_full((b, h, n, pad), -torch.inf)], dim=3)
        pooled[:, lo // block:lo // block + nb] = (
            scores.view(b, h, n, nb, block).amax(dim=(1, 4))
            .permute(0, 2, 1).float())
    return pooled


def select_pages(query: torch.Tensor, sums_k: torch.Tensor,
                 layout: PagedLayout, top_pages: int, *, block: int = 32,
                 local_radius: int = 0, chunk_blocks: int = 64):
    """Rank pages by ``q . Ks[p]``, pooled over heads and over a query block.

    Pooling granularity is the knob the LLM port found to matter most (block
    pooling captured ~0.72 of the far-page attention mass against ~0.31 for one
    set per call), and a DiT needs at least that: rows far apart in the packed
    sequence attend to completely different places.

    The score tensor is built in chunks of query blocks.  Unchunked it is
    ``(B, H, P, S)`` -- 2.5 GB in bf16 at H3's default shape, where the LLM's
    2048-token segment made it 17 MB.

    Returns ``(selection, set_of)``: ``selection`` is (B, n_blocks, k) page ids,
    ``set_of`` is (S,) mapping each query to its row of ``selection``.
    """
    b, h, s, _ = query.shape
    device = query.device
    n = layout.n_pages
    n_blocks = (s + block - 1) // block
    set_of = torch.arange(s, device=device) // block
    if n == 0:
        return (torch.zeros(b, n_blocks, 0, dtype=torch.long, device=device),
                set_of)

    # scores come from a fused kernel: the torch path had to materialise
    # (B, H, P, S) to reduce it away, and its bf16 accumulation misranked
    # roughly a fifth of a top-32 selection
    from .kernel import HAVE_TRITON, block_scores
    if HAVE_TRITON and query.is_cuda and n_blocks * n > 0:
        pooled = block_scores(query, sums_k.to(query.dtype), block)
    else:
        pooled = _block_scores_torch(query, sums_k, n, block, n_blocks,
                                     chunk_blocks)

    if local_radius >= 0:
        forced = _forced_pages(layout, n_blocks, block, local_radius, device)
        pooled = pooled.masked_fill(forced.unsqueeze(0), float("inf"))

    k = min(top_pages, n)
    # A stable sort on the negated score breaks ties toward the LOWER page id;
    # topk gives no such guarantee and the llama.cpp paths specify this rule.
    order = torch.argsort(-pooled, dim=-1, stable=True)
    return order[..., :k].contiguous(), set_of


def _exact_with_lse(query, key, value, scale, chunk: int = 4096):
    """Exact attention over a contiguous key set, plus its logsumexp.

    The LSE is the weight this branch carries into the merge, so a softmax-only
    kernel cannot be used; flash computes it internally and hands it over.  The
    fallback materialises logits in query chunks, because the DiT hands over the
    whole sequence at once and the full tensor at H3's shape would be
    38010 x 778 x 56.  Measured at that shape: flash 36 ms, fallback 135 ms.
    """
    b, h, s, _ = query.shape
    dv = value.size(-1)
    if (key.size(2) > 0 and query.is_cuda
            and query.dtype in (torch.float16, torch.bfloat16)):
        # An empty key set has to keep its -inf logsumexp so the merge reads
        # this branch as contributing nothing; flash hangs on a zero-length
        # key instead of saying so.
        try:
            r = torch.ops.aten._scaled_dot_product_flash_attention(
                query, key, value, dropout_p=0.0, is_causal=False, scale=scale)
            return r[0].float(), r[1].float()
        except Exception:
            pass
    out = torch.empty(b, h, s, dv, device=query.device, dtype=torch.float32)
    lse = torch.empty(b, h, s, device=query.device, dtype=torch.float32)
    kf, vf = key.float(), value.float()
    for lo in range(0, s, chunk):
        hi = min(lo + chunk, s)
        logits = torch.einsum("bhqd,bhkd->bhqk", query[:, :, lo:hi].float(),
                              kf) * scale
        lse[:, :, lo:hi] = torch.logsumexp(logits, dim=-1)
        out[:, :, lo:hi] = torch.einsum("bhqk,bhkd->bhqd",
                                        logits.softmax(-1), vf)
    return out, lse


def _merge_lse(out_a, lse_a, out_b, lse_b, both_finite=False):
    """Combine two attentions over DISJOINT key sets by their logsumexps.

    ``both_finite`` says the caller knows neither branch can be empty -- which
    it does, from the layout: the far branch has pages and the exact branch has
    the conditioning prefix.  That turns the merge from five passes over a
    (1, H, S, D) fp32 tensor into one ``lerp``.  Measured at 1344x768/124f the
    merge and its reshapes were 19% of the read; the guarded path only exists
    for the degenerate shapes the tests cover.
    """
    if both_finite:
        mx = torch.maximum(lse_a, lse_b)
        wa = (lse_a - mx).exp_()
        wb = (lse_b - mx).exp_()
        wb = wb.div_(wa.add_(wb)).unsqueeze(-1)
        return torch.lerp(out_a, out_b, wb)

    fa, fb = torch.isfinite(lse_a), torch.isfinite(lse_b)
    oa = torch.where(fa.unsqueeze(-1), out_a.float(), 0.0)
    ob = torch.where(fb.unsqueeze(-1), out_b.float(), 0.0)
    mx = torch.maximum(lse_a, lse_b)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    wa, wb = (lse_a - mx).exp(), (lse_b - mx).exp()
    den = (wa + wb).clamp_min(torch.finfo(torch.float32).tiny)
    return (wa.unsqueeze(-1) * oa + wb.unsqueeze(-1) * ob) / den.unsqueeze(-1)


@reordered
def lod_attention_ref(query, key, value, *, prefix: int, page_size: int = 64,
                      top_pages: int = 32, select_block: int = 32,
                      local_radius: int = 0, scale: float | None = None,
                      selection=None):
    """The reference read: ordinary matmuls, one softmax, no kernels.

    This is what the fast path is checked against, and what the dense anchor is
    proved on.  It walks query blocks because every query in a block shares one
    page set, which keeps the leaf gather at (kP*ps) rows per block instead of
    per query.
    """
    b, h, s, dk = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(dk)
    layout = PagedLayout(s, prefix, page_size)
    ks, vs = page_sums(key, value, layout)
    if selection is None:
        selection = select_pages(query, ks, layout, top_pages,
                                 block=select_block,
                                 local_radius=local_radius)
    sel, _ = selection

    ex_k = layout.exact_rows(key).float()
    ex_v = layout.exact_rows(value).float()
    paged_k = key[:, :, layout.prefix:layout.paged_end].float()
    paged_v = value[:, :, layout.prefix:layout.paged_end].float()
    mean_k = (ks / page_size)
    mean_v = (vs / page_size)
    n = layout.n_pages
    offsets = torch.arange(page_size, device=query.device)

    out = torch.empty(b, h, s, value.size(-1), device=query.device,
                      dtype=torch.float32)
    for g in range(sel.size(1)):
        lo = g * select_block
        hi = min(lo + select_block, s)
        qs = query[:, :, lo:hi].float()
        logits = [torch.einsum("bhqd,bhkd->bhqk", qs, ex_k) * scale]
        parts = [ex_v]
        if n:
            # tier S: every complete page, with the selected ones silenced.
            # A page contributes its summary or its tokens, never both and
            # never neither -- that is what makes this refinement.
            page_logits = (torch.einsum("bhqd,bhpd->bhqp", qs, mean_k) * scale
                           + math.log(page_size))
            chosen = torch.zeros(b, n, dtype=torch.bool, device=query.device)
            if sel.size(-1):
                chosen.scatter_(1, sel[:, g], True)
            logits.append(page_logits.masked_fill(
                chosen[:, None, None, :], -torch.inf))
            parts.append(mean_v)
            if sel.size(-1):
                ids = (sel[:, g, :, None] * page_size + offsets).reshape(b, -1)
                leaf_k = torch.gather(
                    paged_k, 2, ids[:, None, :, None].expand(b, h, ids.size(1), dk))
                leaf_v = torch.gather(
                    paged_v, 2,
                    ids[:, None, :, None].expand(b, h, ids.size(1), value.size(-1)))
                logits.append(
                    torch.einsum("bhqd,bhkd->bhqk", qs, leaf_k) * scale)
                parts.append(leaf_v)
        weights = torch.softmax(torch.cat(logits, dim=-1), dim=-1)
        acc = None
        cursor = 0
        for part in parts:
            width = part.size(2)
            piece = torch.einsum("bhqk,bhkd->bhqd",
                                 weights[..., cursor:cursor + width], part)
            acc = piece if acc is None else acc + piece
            cursor += width
        out[:, :, lo:hi] = acc
    return out.to(query.dtype)


@reordered
def lod_attention(query, key, value, *, prefix: int, page_size: int = 64,
                  top_pages: int = 32, select_block: int = 32,
                  local_radius: int = 0, scale: float | None = None,
                  selection=None, kernel_opts=None, variant: str = "default"):
    """Kernel-backed read.  Same contract as :func:`lod_attention_ref`.

    ``variant`` selects an experimental kernel from ``kernel_exp``; "default"
    is the shipping one in ``kernel``.
    """
    from .kernel_exp import lod_far_read_exp

    b, h, s, dk = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(dk)
    layout = PagedLayout(s, prefix, page_size)
    ks, vs = page_sums(key, value, layout)

    ex_k = layout.exact_rows(key)
    ex_v = layout.exact_rows(value)
    if layout.n_pages == 0:
        # nothing to page: flash reads the whole thing better than this kernel
        out, _ = _exact_with_lse(query, ex_k, ex_v, scale)
        return out.to(query.dtype)

    if selection is None:
        selection = select_pages(query, ks, layout, top_pages,
                                 block=select_block,
                                 local_radius=local_radius)
    sel, _ = selection
    n_groups = int(sel.size(1))
    pad = n_groups * select_block - s
    q = query
    if pad:
        q = torch.cat([q, q.new_zeros(b, h, pad, dk)], dim=2)
    grouped = q.view(b, h, n_groups, select_block, dk).contiguous()

    mask = torch.zeros(b, n_groups, layout.n_pages, dtype=torch.bool,
                       device=query.device)
    if sel.size(-1):
        mask.scatter_(2, sel, True)

    # tiers X, S and L all land in one online softmax, so the kernel returns
    # the finished read -- no second attention and no logsumexp merge
    out, _ = lod_far_read_exp(
        variant, grouped,
        (ks / page_size).contiguous(), (vs / page_size).contiguous(),
        mask, sel,
        key[:, :, layout.prefix:layout.paged_end],
        value[:, :, layout.prefix:layout.paged_end],
        ex_k, ex_v,
        page_size=page_size, scale=scale, **(kernel_opts or {}))
    return out.reshape(b, h, n_groups * select_block, dk)[:, :, :s].to(query.dtype)

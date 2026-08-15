"""Triton kernel for the LoD read, with a head dim of 128 added.

Ported from ``Qwen3.5-2B-Lodx/lodx_kernel.py``.  The kernel body is the same
online softmax over tier S (page summaries) and tier L (leaves of the selected
pages); the only change is that the accumulator halves are now conditional on
``D`` so a 128-wide head works.  The original hard-coded ``a0``/``a1`` and
stored at ``base`` and ``base + 128``, which writes 256 lanes for a 128-wide
head -- the ``D % 128`` guard let that through.

MiniMax H3 uses ``attention_head_dim=128`` (comfy/ldm/minimax/model.py), so
this is the shape the DiT port needs.

TILE SHAPE is picked per architecture by :func:`default_tiles`; see the table
there for what was measured where.  The CDNA3 row is the MI325X tuning the LLM
kernels were built on, the RDNA3 row was measured here at H3's shape, and any
call can override both.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                   # pragma: no cover
    from types import SimpleNamespace
    HAVE_TRITON = False
    triton = SimpleNamespace(jit=lambda f: f, cdiv=lambda a, b: -(-a // b))
    tl = SimpleNamespace(constexpr=int)


@triton.jit
def _lod_read_kernel(
    q_ptr, ks_ptr, vs_ptr, pmask_ptr, sel_ptr, k_ptr, v_ptr,
    ek_ptr, ev_ptr,
    o_ptr, lse_ptr,
    sq_b, sq_h, sq_g, sq_q,
    sks_b, sks_h, sks_p,
    spm_b, spm_g,
    ssel_b, ssel_g,
    sk_b, sk_h, sk_t,
    se_b, se_h, se_t,
    so_b, so_h, so_g, so_q,
    sl_b, sl_h, sl_g,
    H, HKV, G, P, KP, QB, EX,
    scale_f, logps,
    BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr,
    PS: tl.constexpr, BN: tl.constexpr, D: tl.constexpr,
):
    """Tiers X, S and L for one tile of queries, in one online softmax.

    Tier X is the exactly-read region -- the conditioning prefix plus any rows
    left over after the last complete page.  It used to be a separate flash
    call merged by logsumexp, which cost two full-size fp32 outputs and their
    merge; folding it in here makes the kernel return the finished result.
    """
    pid_bhg = tl.program_id(0)
    pid_q = tl.program_id(1)
    g = pid_bhg % G
    h = (pid_bhg // G) % H
    b = pid_bhg // (G * H)
    hkv = h // (H // HKV)

    qoff = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qmask = qoff < QB
    d0 = tl.arange(0, 64)
    d1 = tl.arange(0, 128)

    # The query tile is invariant across every tier and every iteration, so it
    # is loaded once here.  Reloading it inside the loops (as the LLM kernel
    # did) costs 422 tile loads per program at a 15 s clip -- on the order of
    # 135 GB of HBM traffic, which is the whole kernel time.
    dq = tl.arange(0, D)
    qt = tl.load(q_ptr + b * sq_b + h * sq_h + g * sq_g
                 + qoff[:, None] * sq_q + dq[None, :],
                 mask=qmask[:, None], other=0.0).to(tl.bfloat16)

    m_i = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    a0 = tl.zeros((BLOCK_Q, 128), tl.float32)
    if D >= 256:
        a1 = tl.zeros((BLOCK_Q, 128), tl.float32)
    if D == 512:
        a2 = tl.zeros((BLOCK_Q, 128), tl.float32)
        a3 = tl.zeros((BLOCK_Q, 128), tl.float32)

    # ---- tier X: the exactly-read rows, identical for every query --------
    rex = tl.arange(0, BN)
    for e0 in range(0, EX, BN):
        idx = e0 + rex
        inb = idx < EX
        off = b * se_b + hkv * se_h + tl.where(inb, idx, 0) * se_t
        kk = tl.load(ek_ptr + off[:, None] + dq[None, :],
                     mask=inb[:, None], other=0.0)
        s = tl.dot(qt, tl.trans(kk))
        s = tl.where(inb[None, :], s * scale_f, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                               m_i - m_safe))
        p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
        vv = tl.load(ev_ptr + off[:, None] + d1[None, :], mask=inb[:, None],
                     other=0.0)
        a0 = a0 * corr[:, None] + tl.dot(p, vv)
        if D >= 256:
            vv = tl.load(ev_ptr + off[:, None] + 128 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a1 = a1 * corr[:, None] + tl.dot(p, vv)
        if D == 512:
            vv = tl.load(ev_ptr + off[:, None] + 256 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a2 = a2 * corr[:, None] + tl.dot(p, vv)
            vv = tl.load(ev_ptr + off[:, None] + 384 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a3 = a3 * corr[:, None] + tl.dot(p, vv)
        l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
        m_i = m_new

    # ---- tier S: one term per complete page, selected ones silenced -------
    for p0 in range(0, P, BLOCK_N):
        idx = p0 + tl.arange(0, BLOCK_N)
        ok = idx < P
        idc = tl.where(ok, idx, 0)
        dead = tl.load(pmask_ptr + b * spm_b + g * spm_g + idc, mask=ok,
                       other=1)
        kk = tl.load(ks_ptr + b * sks_b + hkv * sks_h
                     + idc[:, None] * sks_p + dq[None, :],
                     mask=ok[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(qt, tl.trans(kk))
        # The summary already carries 1/ps; log(ps) restores the count that a
        # single term has to stand in for.
        s = s * scale_f + logps
        s = tl.where(ok[None, :] & (dead == 0), s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                               m_i - m_safe))
        p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
        vv = tl.load(vs_ptr + b * sks_b + hkv * sks_h + idc[:, None] * sks_p
                     + d1[None, :], mask=ok[:, None],
                     other=0.0).to(tl.bfloat16)
        a0 = a0 * corr[:, None] + tl.dot(p, vv)
        if D >= 256:
            vv = tl.load(vs_ptr + b * sks_b + hkv * sks_h
                         + idc[:, None] * sks_p + 128 + d1[None, :],
                         mask=ok[:, None], other=0.0).to(tl.bfloat16)
            a1 = a1 * corr[:, None] + tl.dot(p, vv)
        if D == 512:
            vv = tl.load(vs_ptr + b * sks_b + hkv * sks_h
                         + idc[:, None] * sks_p + 256 + d1[None, :],
                         mask=ok[:, None], other=0.0).to(tl.bfloat16)
            a2 = a2 * corr[:, None] + tl.dot(p, vv)
            vv = tl.load(vs_ptr + b * sks_b + hkv * sks_h
                         + idc[:, None] * sks_p + 384 + d1[None, :],
                         mask=ok[:, None], other=0.0).to(tl.bfloat16)
            a3 = a3 * corr[:, None] + tl.dot(p, vv)
        l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
        m_i = m_new

    # ---- tier L: the tokens of the selected pages ------------------------
    rbn = tl.arange(0, BN)
    for i0 in range(0, KP * PS, BN):
        # KP*PS need not divide BN, and reading past the selection list would
        # pull in a garbage page id AND a garbage mask -- phantom leaves that
        # nothing downstream can detect.
        inb = (i0 + rbn) < KP * PS
        slot = tl.where(inb, (i0 + rbn) // PS, 0)
        pid_ = tl.load(sel_ptr + b * ssel_b + g * ssel_g + slot, mask=inb,
                       other=0)
        leaf = (i0 + rbn) % PS
        off = b * sk_b + hkv * sk_h + (pid_ * PS + leaf) * sk_t
        kk = tl.load(k_ptr + off[:, None] + dq[None, :],
                     mask=inb[:, None], other=0.0)
        s = tl.dot(qt, tl.trans(kk))
        s = tl.where(inb[None, :], s * scale_f, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                               m_i - m_safe))
        p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
        vv = tl.load(v_ptr + off[:, None] + d1[None, :], mask=inb[:, None],
                     other=0.0)
        a0 = a0 * corr[:, None] + tl.dot(p, vv)
        if D >= 256:
            vv = tl.load(v_ptr + off[:, None] + 128 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a1 = a1 * corr[:, None] + tl.dot(p, vv)
        if D == 512:
            vv = tl.load(v_ptr + off[:, None] + 256 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a2 = a2 * corr[:, None] + tl.dot(p, vv)
            vv = tl.load(v_ptr + off[:, None] + 384 + d1[None, :],
                         mask=inb[:, None], other=0.0)
            a3 = a3 * corr[:, None] + tl.dot(p, vv)
        l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
        m_i = m_new

    den = tl.maximum(l_i, 1e-30)
    base = (o_ptr + b * so_b + h * so_h + g * so_g + qoff[:, None] * so_q)
    tl.store(base + d1[None, :], a0 / den[:, None], mask=qmask[:, None])
    if D >= 256:
        tl.store(base + 128 + d1[None, :], a1 / den[:, None],
                 mask=qmask[:, None])
    if D == 512:
        tl.store(base + 256 + d1[None, :], a2 / den[:, None],
                 mask=qmask[:, None])
        tl.store(base + 384 + d1[None, :], a3 / den[:, None],
                 mask=qmask[:, None])
    # -inf when the tier pair was empty, which the LSE merge reads as "this
    # branch contributes nothing" rather than as a zero-weight NaN.
    tl.store(lse_ptr + b * sl_b + h * sl_h + g * sl_g + qoff,
             tl.where(m_i == float("-inf"), float("-inf"), tl.log(den) + m_i),
             mask=qmask)


#: Tiles per architecture.  CDNA3 numbers are the ones measured on MI325X while
#: building the LLM kernels -- a register-resident 32x32x256 dot tops out at 159
#: TFLOPS while 32x128x64 reaches 424, so the rate is set by the N width and the
#: leaf walk wants BN=128.
#:
#: On RDNA3 that inverts: both N widths want to be as narrow as WMMA allows.
#: Measured at H3's shape on a W7900 (torch 2.13+rocm7.2, triton 3.7.1), kP=32:
#:
#:     bn=64  block_n=32   117.6 ms   24.4 TFLOPS   (the CDNA3-shaped guess)
#:     bn=32  block_n=16    90.2 ms   31.8 TFLOPS
#:     bn=16  block_n=16    83.1 ms   34.5 TFLOPS
#:
#: block_n=8 does not compile: WMMA has no intrinsic below N=16, so 16 is the
#: floor rather than a tuning choice.
#:
#: The limit on RDNA3 is REGISTERS, not LDS: every config sits at the 256 VGPR
#: ceiling while using 16 KB of the 64 KB LDS, and the ranking follows the
#: spill count.  Measured at 640x640/15 s, kP=64 (spill / ms):
#:
#:     bn=32 stages=1   81 / 187.4      the tuning before the query hoist
#:     bn=16 stages=1    0 / 185.3
#:     bn=32 stages=2   38 / 184.4
#:     bn=16 stages=2   24 / 172.4      <- current
#:     bn=16 stages=3   99 / 189.9
#:
#: stages=2 only became viable once the query tile was hoisted out of the tier
#: loops; before that it spilled hard and cost 4-8x.  More warps removes the
#: spills entirely (231 regs, 0 spills at 8) but costs 1.6x, because the
#: per-wave register budget collapses with it.
#:
#: The warp count has to track the query tile, which is what made 4 warps look
#: bad in the first sweep: at a 32-wide tile 4 warps costs 40%, at a 64-wide
#: tile it is 1.7x FASTER than 2 (73.4 ms vs 125.9).  Raising the query tile is
#: worth doing -- the vendor flash reaches 59.6 TFLOPS on this shape and a
#: 32-wide tile only gets to 34.5, a 64-wide one to 41.2 -- but the tile is
#: capped by select_block, so it is a QUALITY decision as much as a speed one
#: and the caller owns it.  128 is past the peak (106 ms).
_TILES_RDNA = {
    32: dict(bn=16, block_n=16, num_warps=2, num_stages=1),
    64: dict(bn=16, block_n=16, num_warps=4, num_stages=2),
}
_TILES_CDNA = {
    32: dict(bn=128, block_n=64, num_warps=2, num_stages=1),
}


def default_tiles(device=None, query_block: int = 32) -> dict:
    """Launch parameters for this GPU and query-block size.

    ``query_block`` is the caller's ``select_block``: one page set per that many
    queries, and the widest dot the kernel can issue.
    """
    try:
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    except Exception:
        arch = ""
    table = _TILES_RDNA if arch.startswith(("gfx11", "gfx12")) else _TILES_CDNA
    key = max((k for k in table if k <= query_block), default=min(table))
    return dict(table[key], block_q=min(key, query_block))


def lod_far_read(query, page_mean_k, page_mean_v, page_mask, selection,
                 key, value, exact_key=None, exact_value=None, *,
                 page_size: int, scale: float,
                 block_q: int | None = None, block_n: int | None = None,
                 bn: int | None = None, num_warps: int | None = None,
                 num_stages: int | None = None):
    """Tiers S+L for grouped queries.  Returns (out, lse), both fp32.

    ``query`` is (B, H, G, QB, D) -- G query groups that each own one page
    set.  ``page_mean_*`` are the page sums already divided by ``page_size``
    at KV-head resolution.  ``key``/``value`` are the paged region only: page
    ``p`` covers rows ``[p*ps, (p+1)*ps)`` of these tensors, so the caller
    passes a slice and the exact rows outside it are handled elsewhere.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton が見つかりません")
    tiles = default_tiles(query.device, int(query.size(3)))
    block_q = tiles["block_q"] if block_q is None else block_q
    block_q = min(block_q, int(query.size(3)))   # never wider than the group
    block_n = tiles["block_n"] if block_n is None else block_n
    bn = tiles["bn"] if bn is None else bn
    num_warps = tiles["num_warps"] if num_warps is None else num_warps
    num_stages = tiles["num_stages"] if num_stages is None else num_stages
    B, H, G, QB, D = query.shape
    P = int(page_mean_k.size(2))
    KP = int(selection.size(-1))
    if D not in (128, 256, 512):
        raise ValueError("LoD kernel supports head dims 128, 256 and 512")
    if exact_key is None:
        exact_key = query.new_empty((B, page_mean_k.size(1), 0, D))
        exact_value = exact_key
    EX = int(exact_key.size(2))
    out = torch.empty((B, H, G, QB, D), device=query.device,
                      dtype=torch.float32)
    lse = torch.empty((B, H, G, QB), device=query.device, dtype=torch.float32)
    pm = page_mask.to(torch.int32).contiguous()
    sel = selection.to(torch.int32).contiguous()
    _lod_read_kernel[(B * H * G, triton.cdiv(QB, block_q))](
        query, page_mean_k, page_mean_v, pm, sel, key, value,
        exact_key, exact_value, out, lse,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        page_mean_k.stride(0), page_mean_k.stride(1), page_mean_k.stride(2),
        pm.stride(0), pm.stride(1),
        sel.stride(0), sel.stride(1),
        key.stride(0), key.stride(1), key.stride(2),
        exact_key.stride(0), exact_key.stride(1), exact_key.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        H, int(page_mean_k.size(1)), G, P, KP, QB, EX,
        scale, math.log(page_size),
        BLOCK_Q=block_q, BLOCK_N=block_n, PS=page_size, BN=bn, D=D,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out, lse


__all__ = ["lod_far_read", "HAVE_TRITON"]


@triton.jit
def _block_score_kernel(
    q_ptr, ks_ptr, out_ptr,
    sq_b, sq_h, sq_s, sks_b, sks_h, sks_p, so_b, so_g,
    H, HKV, P, S, BLK,
    BLOCK_P: tl.constexpr, D: tl.constexpr, CHUNK: tl.constexpr,
):
    """max over (heads, queries in the block) of q . Ks[p], for one query block.

    Written because the torch version has to materialise (B, H, P, q) to reduce
    it away -- 196 MB per chunk at a 15 s clip, which made selection memory
    bound at 24.5 TFLOPS.  Here the scores never leave registers.
    """
    g = tl.program_id(0)
    p0 = tl.program_id(1) * BLOCK_P
    b = 0
    pidx = p0 + tl.arange(0, BLOCK_P)
    pok = pidx < P
    acc = tl.full((BLOCK_P,), float("-inf"), tl.float32)
    dd = tl.arange(0, D)

    for h in range(H):
        hkv = h // (H // HKV)
        kk = tl.load(ks_ptr + b * sks_b + hkv * sks_h
                     + tl.where(pok, pidx, 0)[:, None] * sks_p + dd[None, :],
                     mask=pok[:, None], other=0.0).to(tl.bfloat16)
        for c0 in range(0, BLK, CHUNK):
            qrow = g * BLK + c0 + tl.arange(0, CHUNK)
            qok = qrow < S
            qq = tl.load(q_ptr + b * sq_b + h * sq_h
                         + tl.where(qok, qrow, 0)[:, None] * sq_s + dd[None, :],
                         mask=qok[:, None], other=0.0).to(tl.bfloat16)
            s = tl.dot(qq, tl.trans(kk))                    # (CHUNK, BLOCK_P)
            s = tl.where(qok[:, None] & pok[None, :], s, float("-inf"))
            acc = tl.maximum(acc, tl.max(s, 0))
    tl.store(out_ptr + b * so_b + g * so_g + pidx, acc, mask=pok)


def block_scores(query, sums_k, block: int):
    """(1, n_blocks, P) block-and-head-pooled selection scores.

    Also more accurate than the torch path it replaces: einsum on bf16 inputs
    rounds the accumulation, and at H3's score magnitudes (~293, spread ~28)
    that is 0.5 absolute on average -- enough to misrank about 20% of a top-32
    selection against an fp64 reference, which this kernel matches exactly.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton が見つかりません")
    B, H, S, D = query.shape
    P = int(sums_k.size(2))
    n_blocks = (S + block - 1) // block
    out = torch.full((B, n_blocks, P), float("-inf"), device=query.device,
                     dtype=torch.float32)
    # measured at 640x640/15 s on a W7900: (128, 32, 8) 10.5 ms against
    # (32, 32, 4) 20.7 ms and 22.4 ms for the torch einsum it replaces
    chunk = 16 if block < 32 else 32
    block_p = 128 if P >= 128 else 32
    num_warps = 8 if block_p == 128 else 4
    _block_score_kernel[(n_blocks, triton.cdiv(P, block_p))](
        query, sums_k, out,
        query.stride(0), query.stride(1), query.stride(2),
        sums_k.stride(0), sums_k.stride(1), sums_k.stride(2),
        out.stride(0), out.stride(1),
        H, int(sums_k.size(1)), P, S, block,
        BLOCK_P=block_p, D=D, CHUNK=chunk,
        num_warps=num_warps, num_stages=1,
    )
    return out


__all__.append("block_scores")

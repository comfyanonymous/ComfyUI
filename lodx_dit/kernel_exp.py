"""Experimental LoD kernels, kept apart from the shipping one.

``kernel.py`` is the tuned, tested read and is not touched by anything here.
This file exists so variants can be tried against it on the same shapes, with
the same acceptance tests, and switched from the ComfyUI node -- a variant only
graduates if it is both faster and still matches the reference.

What is being probed, and why:

The limit on RDNA3 is registers, not LDS.  Every config sits at the 256 VGPR
ceiling while using 16 KB of the 64 KB LDS, and the ranking follows the spill
count.  The fp32 accumulator is the biggest single consumer: ``(BLOCK_Q, 128)``
floats is 64 VGPR per thread at 4 warps.  So the variants attack register
pressure from three directions:

acc_dtype
    Carry the online-softmax accumulator in bf16.  Halves its register
    footprint.  The accumulator is a running weighted sum over up to a few
    hundred tiles, so this is the variant most likely to cost accuracy -- the
    tests measure that rather than assuming it.

tiers
    Run the tiers as separate launches and merge by logsumexp.  Each launch
    holds fewer live values, but the partial (out, lse) pairs go through HBM,
    so it trades register pressure for bandwidth.

imprecise
    ``tl.dot(max_num_imprecise_acc=...)`` lets the backend accumulate part of
    the K reduction at lower precision.  Cheap to try, and only meaningful if
    the AMD backend honours it.
"""

from __future__ import annotations

import math

import torch

from .kernel import HAVE_TRITON, default_tiles

try:
    import triton
    import triton.language as tl
except ImportError:                                   # pragma: no cover
    from types import SimpleNamespace
    triton = SimpleNamespace(jit=lambda f: f, cdiv=lambda a, b: -(-a // b))
    tl = SimpleNamespace(constexpr=int)

#: name -> (accumulate in bf16, tiers per launch)
VARIANTS = {
    "default": None,                       # kernel.py, untouched
    "bf16_acc": dict(acc_bf16=True, split=False, imprecise=0),
    "split_tiers": dict(acc_bf16=False, split=True, imprecise=0),
    "imprecise_acc": dict(acc_bf16=False, split=False, imprecise=1),
    "bf16_split": dict(acc_bf16=True, split=True, imprecise=0),
}


@triton.jit
def _exp_kernel(
    q_ptr, ks_ptr, vs_ptr, pmask_ptr, sel_ptr, k_ptr, v_ptr, ek_ptr, ev_ptr,
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
    ACC_BF16: tl.constexpr, DO_X: tl.constexpr, DO_S: tl.constexpr,
    DO_L: tl.constexpr, IMPRECISE: tl.constexpr,
):
    """One online softmax over the enabled tiers.  D is 128 only.

    Kept to a single head width on purpose: this is a probe, and the shipping
    kernel already covers 256 and 512.
    """
    pid_bhg = tl.program_id(0)
    pid_q = tl.program_id(1)
    g = pid_bhg % G
    h = (pid_bhg // G) % H
    b = pid_bhg // (G * H)
    hkv = h // (H // HKV)

    qoff = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qmask = qoff < QB
    dq = tl.arange(0, D)
    qt = tl.load(q_ptr + b * sq_b + h * sq_h + g * sq_g
                 + qoff[:, None] * sq_q + dq[None, :],
                 mask=qmask[:, None], other=0.0).to(tl.bfloat16)

    m_i = tl.full((BLOCK_Q,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_Q,), tl.float32)
    if ACC_BF16:
        acc = tl.zeros((BLOCK_Q, D), tl.bfloat16)
    else:
        acc = tl.zeros((BLOCK_Q, D), tl.float32)

    if DO_X:
        rex = tl.arange(0, BN)
        for e0 in range(0, EX, BN):
            idx = e0 + rex
            inb = idx < EX
            off = b * se_b + hkv * se_h + tl.where(inb, idx, 0) * se_t
            kk = tl.load(ek_ptr + off[:, None] + dq[None, :],
                         mask=inb[:, None], other=0.0)
            s = tl.dot(qt, tl.trans(kk), max_num_imprecise_acc=IMPRECISE)
            s = tl.where(inb[None, :], s * scale_f, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                                   m_i - m_safe))
            p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
            vv = tl.load(ev_ptr + off[:, None] + dq[None, :],
                         mask=inb[:, None], other=0.0)
            upd = tl.dot(p, vv, max_num_imprecise_acc=IMPRECISE)
            if ACC_BF16:
                acc = (acc.to(tl.float32) * corr[:, None] + upd).to(tl.bfloat16)
            else:
                acc = acc * corr[:, None] + upd
            l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
            m_i = m_new

    if DO_S:
        for p0 in range(0, P, BLOCK_N):
            idx = p0 + tl.arange(0, BLOCK_N)
            ok = idx < P
            idc = tl.where(ok, idx, 0)
            dead = tl.load(pmask_ptr + b * spm_b + g * spm_g + idc, mask=ok,
                           other=1)
            kk = tl.load(ks_ptr + b * sks_b + hkv * sks_h
                         + idc[:, None] * sks_p + dq[None, :],
                         mask=ok[:, None], other=0.0).to(tl.bfloat16)
            s = tl.dot(qt, tl.trans(kk), max_num_imprecise_acc=IMPRECISE)
            s = s * scale_f + logps
            s = tl.where(ok[None, :] & (dead == 0), s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                                   m_i - m_safe))
            p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
            vv = tl.load(vs_ptr + b * sks_b + hkv * sks_h
                         + idc[:, None] * sks_p + dq[None, :],
                         mask=ok[:, None], other=0.0).to(tl.bfloat16)
            upd = tl.dot(p, vv, max_num_imprecise_acc=IMPRECISE)
            if ACC_BF16:
                acc = (acc.to(tl.float32) * corr[:, None] + upd).to(tl.bfloat16)
            else:
                acc = acc * corr[:, None] + upd
            l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
            m_i = m_new

    if DO_L:
        rbn = tl.arange(0, BN)
        for i0 in range(0, KP * PS, BN):
            inb = (i0 + rbn) < KP * PS
            slot = tl.where(inb, (i0 + rbn) // PS, 0)
            pid_ = tl.load(sel_ptr + b * ssel_b + g * ssel_g + slot, mask=inb,
                           other=0)
            leaf = (i0 + rbn) % PS
            off = b * sk_b + hkv * sk_h + (pid_ * PS + leaf) * sk_t
            kk = tl.load(k_ptr + off[:, None] + dq[None, :],
                         mask=inb[:, None], other=0.0)
            s = tl.dot(qt, tl.trans(kk), max_num_imprecise_acc=IMPRECISE)
            s = tl.where(inb[None, :], s * scale_f, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            corr = tl.exp(tl.where(m_i == float("-inf"), float("-inf"),
                                   m_i - m_safe))
            p = tl.exp(s - m_safe[:, None]).to(tl.bfloat16)
            vv = tl.load(v_ptr + off[:, None] + dq[None, :],
                         mask=inb[:, None], other=0.0)
            upd = tl.dot(p, vv, max_num_imprecise_acc=IMPRECISE)
            if ACC_BF16:
                acc = (acc.to(tl.float32) * corr[:, None] + upd).to(tl.bfloat16)
            else:
                acc = acc * corr[:, None] + upd
            l_i = l_i * corr + tl.sum(p.to(tl.float32), 1)
            m_i = m_new

    den = tl.maximum(l_i, 1e-30)
    base = (o_ptr + b * so_b + h * so_h + g * so_g + qoff[:, None] * so_q)
    tl.store(base + dq[None, :], acc.to(tl.float32) / den[:, None],
             mask=qmask[:, None])
    tl.store(lse_ptr + b * sl_b + h * sl_h + g * sl_g + qoff,
             tl.where(m_i == float("-inf"), float("-inf"), tl.log(den) + m_i),
             mask=qmask)


def _launch(query, mk, mv, pm, sel, key, value, ek, ev, page_size, scale,
            tiles, do_x, do_s, do_l, acc_bf16, imprecise):
    B, H, G, QB, D = query.shape
    P = int(mk.size(2))
    KP = int(sel.size(-1))
    EX = int(ek.size(2))
    out = torch.empty((B, H, G, QB, D), device=query.device,
                      dtype=torch.float32)
    lse = torch.empty((B, H, G, QB), device=query.device, dtype=torch.float32)
    block_q = min(tiles["block_q"], QB)
    _exp_kernel[(B * H * G, triton.cdiv(QB, block_q))](
        query, mk, mv, pm, sel, key, value, ek, ev, out, lse,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        mk.stride(0), mk.stride(1), mk.stride(2),
        pm.stride(0), pm.stride(1),
        sel.stride(0), sel.stride(1),
        key.stride(0), key.stride(1), key.stride(2),
        ek.stride(0), ek.stride(1), ek.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        H, int(mk.size(1)), G, P, KP, QB, EX,
        scale, math.log(page_size),
        BLOCK_Q=block_q, BLOCK_N=tiles["block_n"], PS=page_size,
        BN=tiles["bn"], D=D, ACC_BF16=acc_bf16, DO_X=do_x, DO_S=do_s,
        DO_L=do_l, IMPRECISE=imprecise,
        num_warps=tiles["num_warps"], num_stages=tiles["num_stages"],
    )
    return out, lse


def lod_far_read_exp(variant, query, page_mean_k, page_mean_v, page_mask,
                     selection, key, value, exact_key, exact_value, *,
                     page_size: int, scale: float, **tile_overrides):
    """Same contract as ``kernel.lod_far_read``, run through a variant."""
    if not HAVE_TRITON:
        raise RuntimeError("triton が見つかりません")
    cfg = VARIANTS[variant]
    if cfg is None:
        from .kernel import lod_far_read
        return lod_far_read(query, page_mean_k, page_mean_v, page_mask,
                            selection, key, value, exact_key, exact_value,
                            page_size=page_size, scale=scale, **tile_overrides)
    if query.size(4) != 128:
        raise ValueError("experimental variants are head-dim 128 only")

    tiles = default_tiles(query.device, int(query.size(3)))
    tiles.update({k: v for k, v in tile_overrides.items() if v is not None})
    pm = page_mask.to(torch.int32).contiguous()
    sel = selection.to(torch.int32).contiguous()
    args = (query, page_mean_k, page_mean_v, pm, sel, key, value,
            exact_key, exact_value, page_size, scale, tiles)

    if not cfg["split"]:
        return _launch(*args, True, True, True, cfg["acc_bf16"],
                       cfg["imprecise"])

    # X and S are small and share a walk pattern; L is the heavy one.  Split
    # there, then combine by logsumexp -- exact, because the key sets are
    # disjoint by construction.
    from .lod_dit import _merge_lse
    a_out, a_lse = _launch(*args, True, True, False, cfg["acc_bf16"],
                           cfg["imprecise"])
    b_out, b_lse = _launch(*args, False, False, True, cfg["acc_bf16"],
                           cfg["imprecise"])
    return _merge_lse(a_out, a_lse, b_out, b_lse), a_lse


__all__ = ["VARIANTS", "lod_far_read_exp"]

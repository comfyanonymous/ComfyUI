"""Single-layer benchmark at MiniMax H3's real attention shape.

The point of this file is one number: how much of the theoretical read
reduction the kernel actually delivers on a bidirectional DiT.  The LLM port
measured 50-64% of ideal on MI325X against a CAUSAL flash baseline; a DiT's
baseline is full quadratic attention, so the same read volume should buy
roughly twice as much.

Shapes come from comfy/ldm/minimax/model.py and PackedLayout:

    1344x768, 124 frames -> latent 37 x 48 x 84, DiT patch 1x2x2
      video rows  37 * 24 * 42 = 37,296
      audio rows  207 * 2      =    414   } exact prefix, never summarised
      text rows   ~300                    }
      S = 38,010   H = 56   head_dim = 128   MHA (no GQA)

    python lodx_dit/bench_h3.py            # default sweep
    python lodx_dit/bench_h3.py --long     # also the 362-frame shape
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lodx_dit.lod_dit import (PagedLayout, _exact_with_lse, _merge_lse,
                              lod_attention, page_sums, select_pages)
from lodx_dit.kernel import lod_far_read

H3 = dict(heads=56, head_dim=128)


def h3_shape(width, height, frames, text=300):
    """Reproduce PackedLayout's row counts without importing ComfyUI."""
    latent_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    video = latent_t * (height // 16 // 2) * (width // 16 // 2)
    audio = round(frames / 24 * 40) * 2
    return text + audio, video          # (prefix, paged rows)


def timed(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def bench(prefix, paged, top_pages, page_size=64, select_block=32,
          local_radius=0, kernel_opts=None, iters=3, label=""):
    h, d = H3["heads"], H3["head_dim"]
    s = prefix + paged
    dev = "cuda"
    g = torch.Generator(device="cpu").manual_seed(0)
    q, k, v = [torch.randn(1, h, s, d, generator=g).to(dev, torch.bfloat16)
               for _ in range(3)]

    dense_t = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False), iters=iters)

    layout = PagedLayout(s, prefix, page_size)
    ko = kernel_opts or {}

    # stage breakdown -- each stage timed on its own so the total is auditable
    t_sums = timed(lambda: page_sums(k, v, layout), iters=iters)
    ks, vs = page_sums(k, v, layout)
    t_sel = timed(lambda: select_pages(q, ks, layout, top_pages,
                                       block=select_block,
                                       local_radius=local_radius), iters=iters)
    sel, _ = select_pages(q, ks, layout, top_pages, block=select_block,
                          local_radius=local_radius)
    t_exact = timed(lambda: _exact_with_lse(q, layout.exact_rows(k),
                                            layout.exact_rows(v),
                                            d ** -0.5), iters=iters)

    n_groups = int(sel.size(1))
    pad = n_groups * select_block - s
    qg = q if not pad else torch.cat([q, q.new_zeros(1, h, pad, d)], dim=2)
    grouped = qg.view(1, h, n_groups, select_block, d).contiguous()
    mask = torch.zeros(1, n_groups, layout.n_pages, dtype=torch.bool,
                       device=dev)
    mask.scatter_(2, sel, True)
    mk = (ks / page_size).contiguous()
    mv = (vs / page_size).contiguous()
    pk = k[:, :, layout.prefix:layout.paged_end]
    pv = v[:, :, layout.prefix:layout.paged_end]
    t_kern = timed(lambda: lod_far_read(grouped, mk, mv, mask, sel, pk, pv,
                                        page_size=page_size, scale=d ** -0.5,
                                        **ko), iters=iters)

    total_t = timed(lambda: lod_attention(
        q, k, v, prefix=prefix, page_size=page_size, top_pages=top_pages,
        select_block=select_block, local_radius=local_radius,
        kernel_opts=ko), iters=iters)

    read = layout.n_pages + top_pages * page_size + layout.exact_len
    ideal = s / read
    print(f"  {label:<22} kP={top_pages:<4} "
          f"dense {dense_t*1000:8.1f} ms | LoD {total_t*1000:8.1f} ms | "
          f"{dense_t/total_t:5.2f}x | read {100*read/s:4.1f}% "
          f"ideal {ideal:5.1f}x | 実現率 {100*(dense_t/total_t)/ideal:4.0f}%")
    print(f"  {'':22} 内訳: sums {t_sums*1000:6.1f} sel {t_sel*1000:6.1f} "
          f"exact {t_exact*1000:6.1f} kernel {t_kern*1000:6.1f} "
          f"(合計 {(t_sums+t_sel+t_exact+t_kern)*1000:6.1f} / 実測 {total_t*1000:6.1f})")
    del q, k, v, ks, vs, grouped, mk, mv, mask
    torch.cuda.empty_cache()
    return dense_t, total_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true",
                    help="also run the 362-frame shape (slow: dense is ~13 s)")
    ap.add_argument("--tune", action="store_true",
                    help="sweep the kernel tile parameters for RDNA3")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--select-block", type=int, default=32)
    args = ap.parse_args()

    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"H3 attention shape: H={H3['heads']} D={H3['head_dim']} MHA, "
          f"bidirectional\n")

    prefix, paged = h3_shape(1344, 768, 124)
    print(f"[1344x768 / 124f]  prefix={prefix} paged={paged} S={prefix+paged}")
    for kP in (16, 32, 64, 128):
        bench(prefix, paged, kP, iters=args.iters, select_block=args.select_block, label="1344x768/124f")

    print(f"\n[768x768 / 124f]  (S < 27k, 線形層が支配的な領域)")
    p2, g2 = h3_shape(768, 768, 124)
    print(f"  prefix={p2} paged={g2} S={p2+g2}")
    for kP in (32, 64):
        bench(p2, g2, kP, iters=args.iters, select_block=args.select_block, label="768x768/124f")

    if args.tune:
        print("\n[kernel tile sweep @ 1344x768/124f kP=32] RDNA3 用の再チューニング")
        for bq, bn, warps, stages in [(32, 128, 2, 1), (32, 128, 4, 1),
                                      (32, 64, 2, 1), (32, 256, 2, 1),
                                      (64, 128, 4, 1), (32, 128, 4, 2)]:
            try:
                bench(prefix, paged, 32, iters=args.iters,
                      kernel_opts=dict(block_q=bq, bn=bn, num_warps=warps,
                                       num_stages=stages),
                      label=f"bq{bq} bn{bn} w{warps} s{stages}")
            except Exception as e:
                print(f"  bq{bq} bn{bn} w{warps} s{stages}: "
                      f"{type(e).__name__}: {str(e)[:80]}")

    if args.long:
        print(f"\n[1344x768 / 362f]  (学習範囲の上限、attention が FLOPs の 80%)")
        p3, g3 = h3_shape(1344, 768, 362)
        print(f"  prefix={p3} paged={g3} S={p3+g3}")
        for kP in (128, 256):
            bench(p3, g3, kP, iters=1, select_block=args.select_block, label="1344x768/362f")


if __name__ == "__main__":
    main()

"""Measure LoD quality on MiniMax H3's real activations.

Synthetic q/k/v cannot answer this.  Gaussian noise has no locality, so
selection scores are effectively random, coverage lands at the read fraction by
construction, and the tiled ordering shows no benefit whatever it is worth.
Every number below therefore comes from the real model mid-denoise.

Two stages, so the 27 GB text encoder and the 34 GB DiT never have to be
resident together:

    python lodx_dit/probe_h3.py encode     # -> conditioning tensor on disk
    python lodx_dit/probe_h3.py probe      # -> metrics on disk

The probe hooks ``comfy.ldm.minimax.model.optimized_attention``, which receives
q/k/v already shaped (1, heads, S, head_dim) and post-RoPE -- exactly the LoD
input contract.  The model's own dense result is kept as the reference, so the
expensive part is free.

Metrics per (step, layer, ordering, kP):

rel
    relative error of the LoD attention output against the model's dense output
coverage
    fraction of the true softmax mass that lands on rows LoD actually reads
    (exact prefix + selected leaves).  This is the metric the LLM port used to
    justify block pooling -- 0.72 for block, 0.31 for one set per call -- and it
    is the one that predicts retrieval failures before they show up as pixels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]                      # comfy's cli_args parses argv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_out")
PROMPT = ("A calico cat naps on a sunlit windowsill, tail flicking, while "
          "leaves move outside; soft ambient room tone.")
CFG = dict(width=1344, height=768, length=124, steps=8,
           layers=(0, 12, 25, 37, 49), sample_steps=(0, 3, 7),
           kps=(16, 32, 64, 128, 256), coverage_queries=128, tag="")


# --------------------------------------------------------------------- stage 1

def stage_encode():
    import comfy.sd
    import folder_paths

    te = folder_paths.get_full_path_or_raise(
        "text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    print(f"loading text encoder: {te}", flush=True)
    clip = comfy.sd.load_clip([te], clip_type=comfy.sd.CLIPType.MINIMAX)
    print("tokenizing", flush=True)
    tokens = clip.tokenize(PROMPT)
    cond = clip.encode_from_tokens_scheduled(tokens)

    os.makedirs(OUT, exist_ok=True)
    payload = [[c[0].cpu(), {k: (v.cpu() if torch.is_tensor(v) else v)
                             for k, v in c[1].items()}] for c in cond]
    torch.save(payload, os.path.join(OUT, "cond.pt"))
    print(f"saved conditioning: {payload[0][0].shape}", flush=True)


# --------------------------------------------------------------------- metrics

def _coverage(q, k, sel, set_of, layout, scale, n_queries, generator):
    """Fraction of the true softmax mass landing on the rows LoD reads.

    Done per head and on a query subsample: the full weight tensor at H3's
    shape is 56 x 38010 x 38010.
    """
    _, heads, s, _ = q.shape
    idx = torch.randperm(s, generator=generator, device=q.device)[:n_queries]
    idx = idx.sort().values
    qs = q[:, :, idx]

    read = torch.zeros(n_queries, s, dtype=torch.bool, device=q.device)
    read[:, :layout.prefix] = True
    read[:, layout.paged_end:] = True
    pages = sel[0][set_of[idx]]                       # (n_queries, kP)
    offs = torch.arange(layout.page_size, device=q.device)
    cols = layout.prefix + (pages[:, :, None] * layout.page_size + offs).reshape(
        n_queries, -1)
    read.scatter_(1, cols, True)

    total = 0.0
    for h in range(heads):
        logits = (qs[0, h].float() @ k[0, h].float().T) * scale
        w = logits.softmax(-1)
        total += float((w * read).sum(-1).mean())
    return total / heads


def _evaluate(q, k, v, dense_out, layout, prefix, grid, tile, scale, gen):
    """One (step, layer) sample: rel error and coverage for each kP/ordering."""
    from lodx_dit.lod_dit import lod_attention, page_sums, select_pages
    from lodx_dit.ordering import invert_order, sequence_order

    s = q.size(2)
    rows = []
    orders = {"raster": None,
              "tiled": sequence_order(s, prefix, grid, tile, device=q.device)}
    for oname, order in orders.items():
        if order is None:
            qo, ko, vo = q, k, v
        else:
            qo, ko, vo = (t.index_select(2, order) for t in (q, k, v))
        ks, _ = page_sums(ko, vo, layout)
        for kp in CFG["kps"]:
            if kp > layout.n_pages:
                continue
            sel, set_of = select_pages(qo, ks, layout, kp, block=32)
            out = lod_attention(qo, ko, vo, prefix=prefix,
                                page_size=layout.page_size, top_pages=kp,
                                select_block=32, selection=(sel, set_of))
            if order is not None:
                out = out.index_select(2, invert_order(order))
            rel = float((out.float() - dense_out.float()).norm()
                        / dense_out.float().norm())
            cov = _coverage(qo, ko, sel, set_of, layout, scale,
                            CFG["coverage_queries"], gen)
            rows.append(dict(ordering=oname, kP=kp, rel=rel, coverage=cov))
            del out
        del ks
    return rows


# --------------------------------------------------------------------- stage 2

def stage_probe():
    import comfy.ldm.minimax.model as h3
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import folder_paths
    import nodes
    from lodx_dit.lod_dit import PagedLayout
    from lodx_dit.ordering import best_tile, video_grid_shape
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    cond = torch.load(os.path.join(OUT, "cond.pt"), weights_only=False)
    text_len = cond[0][0].shape[1]
    grid = video_grid_shape(CFG["width"], CFG["height"], CFG["length"])
    video_rows = grid[0] * grid[1] * grid[2]
    tile, page_size = best_tile(grid, 64)
    print(f"grid={grid} video_rows={video_rows} text_len={text_len} "
          f"tile={tile} ps={page_size}", flush=True)

    unet = folder_paths.get_full_path_or_raise(
        "diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors")
    print(f"loading DiT: {unet}", flush=True)
    model = comfy.sd.load_diffusion_model(unet)

    latent, _ = _empty_av_latent(CFG["width"], CFG["height"], CFG["length"])
    state = dict(step=-1, layer=0, rows=[], t0=time.time())
    gen = torch.Generator(device="cuda").manual_seed(0)
    real_attn = h3.optimized_attention
    real_forward = h3.MiniMaxH3Model._forward

    def forward(self, *a, **kw):
        state["step"] += 1
        state["layer"] = 0
        return real_forward(self, *a, **kw)

    def attn(q, k, v, heads, *a, **kw):
        layer = state["layer"]
        seq = q.peek().size(2)
        # the token refiner runs the same Attention over the text rows only
        sample = (seq > text_len and state["step"] in CFG["sample_steps"]
                  and layer in CFG["layers"])
        if seq > text_len:
            state["layer"] = layer + 1
        keep = None
        if sample:
            keep = tuple(t.peek().clone() for t in (q, k, v))
        out = real_attn(q, k, v, heads, *a, **kw)
        if sample:
            qq, kk, vv = keep
            s = qq.size(2)
            prefix = s - video_rows
            dense = out.view(1, s, heads, -1).transpose(1, 2)
            layout = PagedLayout(s, prefix, page_size)
            try:
                got = _evaluate(qq, kk, vv, dense, layout, prefix, grid, tile,
                                qq.size(-1) ** -0.5, gen)
            except torch.OutOfMemoryError as e:
                print(f"    OOM at step {state['step']} layer {layer}: "
                      f"{str(e)[:80]}", flush=True)
                torch.cuda.empty_cache()
                got = []
            for r in got:
                r.update(step=state["step"], layer=layer, S=s, prefix=prefix,
                         n_pages=layout.n_pages)
            state["rows"].extend(got)
            print(f"[{time.time()-state['t0']:6.0f}s] step {state['step']} "
                  f"layer {layer} S={s} prefix={prefix} "
                  f"pages={layout.n_pages}", flush=True)
            for r in got:
                print(f"    {r['ordering']:>6} kP={r['kP']:<4} "
                      f"rel={r['rel']:.4f} coverage={r['coverage']:.4f}",
                      flush=True)
            del keep, qq, kk, vv, dense
            torch.cuda.empty_cache()
        return out

    h3.optimized_attention = attn
    h3.MiniMaxH3Model._forward = forward
    try:
        nodes.common_ksampler(model, 0, CFG["steps"], 1.0, "euler", "simple",
                              cond, cond, latent, denoise=1.0)
    finally:
        h3.optimized_attention = real_attn
        h3.MiniMaxH3Model._forward = real_forward
        os.makedirs(OUT, exist_ok=True)
        name = "metrics{}.json".format(CFG["tag"])
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(state["rows"], f, indent=1)
        print(f"wrote {len(state['rows'])} rows to {OUT}/{name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["encode", "probe"])
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--length", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--layers", type=str, help="comma separated")
    ap.add_argument("--sample-steps", type=str, help="comma separated")
    ap.add_argument("--kps", type=str, help="comma separated")
    ap.add_argument("--coverage-queries", type=int)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args(_ARGV)
    for key in ("width", "height", "length", "steps", "coverage_queries",
                "tag"):
        if getattr(a, key) is not None:
            CFG[key] = getattr(a, key)
    for key, src in (("layers", a.layers), ("sample_steps", a.sample_steps),
                     ("kps", a.kps)):
        if src:
            CFG[key] = tuple(int(x) for x in src.split(","))
    print("config:", CFG, flush=True)
    {"encode": stage_encode, "probe": stage_probe}[a.stage]()

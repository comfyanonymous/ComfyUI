"""Tensor-parallel one MiniMax H3 DiT block across GPUs -- standalone probe.

Nothing here touches comfy.  It reads the real quantized weights for one block
straight out of the checkpoint (~230 MB, not the 34 GB model), rebuilds the
block's arithmetic, and runs it two ways:

    reference   everything on one GPU
    TP          split across N GPUs, with the two all-reduces a real TP needs

so the question "does splitting these layers give the same answer, and is it
faster" gets an answer without committing to an implementation.

    CUDA_VISIBLE_DEVICES=0,1 python lodx_dit/tp_probe.py --seq 15514

How the four linear layers split (docs/lod-dit-results.md section 8):

    qkv_proj  [21504, 5376]   column (output).  Rows are [q|k|v], 56 heads of
                              128 each, so GPU p wants heads [p*28,(p+1)*28)
                              out of ALL THREE -- three strided slices, not a
                              contiguous half.
    out_proj  [5376, 7168]    row (input), matching the head split.
                              3584 = 14*256, so the convrot groups stay whole.
    fc1       [28672, 5376]   column.  Rows are [gate|up] (comfy/ops.py:947),
                              so again two strided slices.
    fc2       [5376, 14336]   row.  7168 = 28*256.

Column splits are bit-exact: convrot's Hadamard is block-diagonal along the
INPUT dim, so slicing output rows never touches it, and weight_scale is
per-output-row.  Row splits are the approximate ones -- each GPU's
``int8_linear`` computes its own activation scale from its own slice, where the
unsplit layer used one scale for the whole row.  That is what ``--shared-scale``
measures the cost of.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import comfy.quant_ops as quant_ops
import folder_paths

ck = quant_ops.ck

HIDDEN, HEADS, HEAD_DIM, FFN = 5376, 56, 128, 14336
INNER = HEADS * HEAD_DIM
GROUP = 256


# --------------------------------------------------------------- weights

def load_block(index: int, device):
    """Read one block's four quantized linears out of the checkpoint."""
    from safetensors import safe_open

    path = folder_paths.get_full_path_or_raise(
        "diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors")
    want = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
    out = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for name in want:
            base = f"blocks.{index}.{name}"
            out[name] = (f.get_tensor(base + ".weight").to(device),
                         f.get_tensor(base + ".weight_scale").to(device))
        for name in ("attn.q_norm", "attn.k_norm"):
            out[name] = f.get_tensor(f"blocks.{index}.{name}.weight").to(device)
    return out


def _rows(qdata, scale, pieces, part, n):
    """Concatenate the same fractional slice out of each contiguous piece.

    qkv_proj and fc1 both hold several logically separate matrices stacked on
    the output axis, so a GPU's share is one slice per piece rather than one
    slice of the whole thing.
    """
    idx = []
    for start, size in pieces:
        step = size // n
        idx.append(torch.arange(start + part * step, start + (part + 1) * step,
                                device=qdata.device))
    idx = torch.cat(idx)
    return qdata.index_select(0, idx).contiguous(), scale.index_select(0, idx).contiguous()


def _cols(qdata, part, n):
    """Slice the input axis, on a convrot group boundary."""
    k = qdata.size(1)
    step = k // n
    if step % GROUP:
        raise SystemExit(f"input split {step} is not a multiple of the convrot "
                         f"group {GROUP}; the rotation would be cut in half")
    return qdata[:, part * step:(part + 1) * step].contiguous()


def shard(w, part, n, device):
    """One GPU's slice of the block."""
    with torch.cuda.device(device):
        qkv_q, qkv_s = _rows(*w["attn.qkv_proj"],
                             [(0, INNER), (INNER, INNER), (2 * INNER, INNER)], part, n)
        fc1_q, fc1_s = _rows(*w["mlp.fc1"], [(0, FFN), (FFN, FFN)], part, n)
        return dict(
            qkv=(qkv_q.to(device), qkv_s.to(device)),
            out=(_cols(w["attn.out_proj"][0], part, n).to(device),
                 w["attn.out_proj"][1].to(device)),
            fc1=(fc1_q.to(device), fc1_s.to(device)),
            fc2=(_cols(w["mlp.fc2"][0], part, n).to(device),
                 w["mlp.fc2"][1].to(device)),
            qn=w["attn.q_norm"].to(device), kn=w["attn.k_norm"].to(device),
            heads=HEADS // n, device=device)


# --------------------------------------------------------------- block math

def _rmsnorm_heads(x, weight, heads, eps=1e-6):
    x = x.view(x.size(0), heads, HEAD_DIM).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(torch.bfloat16)


def block_forward(x, s, *, heads, qkv, out, fc1, fc2, qn, kn, device):
    """attention + MLP for one shard.  Returns the two partial results."""
    qkv_out = ck.int8_linear(x, qkv[0], qkv[1], None, x.dtype,
                             convrot=True, convrot_groupsize=GROUP)
    q, k, v = qkv_out.split(heads * HEAD_DIM, dim=-1)
    q = _rmsnorm_heads(q, qn, heads).transpose(0, 1).unsqueeze(0)
    k = _rmsnorm_heads(k, kn, heads).transpose(0, 1).unsqueeze(0)
    v = v.view(s, heads, HEAD_DIM).transpose(0, 1).unsqueeze(0)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    a = a.squeeze(0).transpose(0, 1).reshape(s, heads * HEAD_DIM)
    attn_partial = ck.int8_linear(a, out[0], out[1], None, x.dtype,
                                  convrot=True, convrot_groupsize=GROUP)

    h = ck.int8_linear(x, fc1[0], fc1[1], None, x.dtype,
                       convrot=True, convrot_groupsize=GROUP)
    mlp_partial = quant_ops.ck.int8_linear(
        h, fc2[0], fc2[1], None, x.dtype, convrot=True,
        convrot_groupsize=GROUP, input_act="swiglu")
    return attn_partial, mlp_partial


# --------------------------------------------------------------- all-reduce

def allreduce(parts, mode):
    """Sum a list of same-shaped tensors living on different GPUs, in place.

    No P2P on this box, so every copy goes through host memory whatever we do;
    what we can choose is how many bytes cross.
    """
    n = len(parts)
    if n == 1:
        return parts
    if mode == "bf16":
        wire = parts
        scales = [None] * n
    elif mode == "fp8":
        wire = [p.to(torch.float8_e4m3fn) for p in parts]
        scales = [None] * n
    elif mode == "fp8s":
        wire, scales = [], []
        for p in parts:
            sc = p.abs().amax(dim=1, keepdim=True).float().clamp_(min=1e-12) / 448.0
            wire.append((p.float() / sc).to(torch.float8_e4m3fn))
            scales.append(sc)
    elif mode == "int8":
        wire, scales = [], []
        for p in parts:
            sc = p.abs().amax(dim=1, keepdim=True).float().clamp_(min=1e-12) / 127.0
            wire.append((p.float() / sc).round_().clamp_(-127, 127).to(torch.int8))
            scales.append(sc)
    else:
        raise SystemExit(f"unknown all-reduce mode {mode}")

    outs = []
    for i, dst in enumerate(parts):
        acc = dst.float()
        for j in range(n):
            if i == j:
                continue
            with torch.cuda.device(dst.device):
                got = wire[j].to(dst.device, non_blocking=True)
                if scales[j] is None:
                    acc = acc + got.to(torch.float32)
                else:
                    acc = acc + got.to(torch.float32) * scales[j].to(dst.device)
        outs.append(acc.to(dst.dtype))
    return outs


# --------------------------------------------------------------- runs

def run_reference(sh, x, s):
    with torch.cuda.device(sh["device"]):
        a, m = block_forward(x, s, **sh)
        return (a + m)


def run_tp(shards, xs, s, mode):
    """Issue every shard before syncing any, so the GPUs actually overlap."""
    parts = []
    for sh, x in zip(shards, xs):
        with torch.cuda.device(sh["device"]):
            parts.append(block_forward(x, s, **sh))
    attn = allreduce([p[0] for p in parts], mode)
    mlp = allreduce([p[1] for p in parts], mode)
    return [a + m for a, m in zip(attn, mlp)]


def split_sweep(w, seq, device):
    """How much does the row split cost numerically, and does it get worse?

    The reference is the unsplit INT8 result, because reproducing that is
    exactly what a TP implementation owes.  Splitting finer gives each chunk its
    own activation scale, which quantizes that chunk *better* but rounds
    differently -- so the error saturates instead of accumulating.
    """
    print("\nrow-split error vs the unsplit INT8 result:")
    for name, k in (("attn.out_proj", INNER), ("mlp.fc2", FFN)):
        q, sc = w[name]
        q, sc = q.to(device), sc.to(device)
        x = torch.randn(min(seq, 4096), k, dtype=torch.bfloat16, device=device) * 0.5
        ref = ck.int8_linear(x, q, sc, None, torch.bfloat16,
                             convrot=True, convrot_groupsize=GROUP)
        cells = []
        for n in (2, 4, 8, 14):
            step = k // n
            if step % GROUP:
                continue
            acc = None
            for i in range(n):
                part = ck.int8_linear(
                    x[:, i * step:(i + 1) * step].contiguous(),
                    q[:, i * step:(i + 1) * step].contiguous(), sc, None,
                    torch.float32, convrot=True, convrot_groupsize=GROUP)
                acc = part if acc is None else acc + part
            cells.append(f"{n}-way {((acc - ref.float()).norm() / ref.float().norm()).item():.2e}")
        print(f"  {name:<14} K={k:<6} " + "  ".join(cells))
    print("  (bf16's own output granularity is ~4e-3, for scale)")


def timed(fn, iters, devices):
    for _ in range(3):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    for d in devices:
        torch.cuda.synchronize(d)
    return (time.perf_counter() - t) / iters * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=15514, help="tokens (5 s clip)")
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()

    n_vis = torch.cuda.device_count()
    if n_vis < a.gpus:
        raise SystemExit(f"asked for {a.gpus} GPUs, {n_vis} visible")
    devices = [torch.device("cuda", i) for i in range(a.gpus)]
    print(f"GPUs {[str(d) for d in devices]}  S={a.seq}  block={a.block}")

    w = load_block(a.block, "cpu")
    print("loaded block weights:",
          {k: tuple(v[0].shape) for k, v in w.items() if isinstance(v, tuple)})

    torch.manual_seed(0)
    x0 = torch.randn(a.seq, HIDDEN, dtype=torch.bfloat16, device=devices[0]) * 0.5

    ref_shard = shard(w, 0, 1, devices[0])
    ref = run_reference(ref_shard, x0, a.seq)
    torch.cuda.synchronize(devices[0])
    print(f"\nreference (1 GPU): {tuple(ref.shape)}  |ref| = {ref.float().norm():.1f}")

    shards = [shard(w, p, a.gpus, devices[p]) for p in range(a.gpus)]
    xs = [x0.to(d) for d in devices]

    print(f"\n{'all-reduce':>12} {'rel err vs 1 GPU':>18} {'ms':>9} {'vs 1 GPU':>10}")
    t_ref = timed(lambda: run_reference(ref_shard, x0, a.seq),
                  a.iters, devices[:1])
    print(f"{'(reference)':>12} {'-':>18} {t_ref:>8.2f} {'1.00x':>10}")
    for mode in ("bf16", "int8", "fp8s", "fp8"):
        got = run_tp(shards, xs, a.seq, mode)[0]
        torch.cuda.synchronize()
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
        ms = timed(lambda m=mode: run_tp(shards, xs, a.seq, m), a.iters, devices)
        print(f"{mode:>12} {rel:>18.3e} {ms:>8.2f} {t_ref/ms:>9.2f}x")

    split_sweep(w, a.seq, devices[0])

    print(f"\nper-step extrapolation (50 blocks, this block's cost x50):")
    print(f"  1 GPU {t_ref*50/1000:6.2f} s")


if __name__ == "__main__":
    main()

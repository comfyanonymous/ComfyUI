"""Run the real H3 DiT once on one GPU and once tensor-parallel, and compare.

Both runs happen in the same process off the same loaded model with the same
seed, so the only difference is the split.  What comes out is the thing
``tp_probe.py`` could not answer: whether the row split's ~1e-2 per-layer error
survives 50 blocks and a full sampler trajectory.

    CUDA_VISIBLE_DEVICES=0,1 python lodx_dit/tp_run.py --width 640 --height 640 \
        --length 48 --steps 4 --mode int8

``--mode`` picks the all-reduce wire format (bf16 / int8 / fp8); bf16 isolates
the row-split error, the other two add the transport error on top.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_A = sys.argv[1:]
sys.argv = [sys.argv[0]]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import comfy_aimdo.control
comfy_aimdo.control.init()

import torch

import comfy.memory_management
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import folder_paths
import nodes
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=640)
ap.add_argument("--length", type=int, default=48)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--gpus", type=int, default=2,
                help="1 = self-check: this file's block code, unsplit")
ap.add_argument("--mode", default="int8", choices=["bf16", "int8", "fp8"])
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--exact-rows", action="store_true",
                help="share the activation scale across row-split halves")
ap.add_argument("--unet", default="minimax_h3_fl2va_int8_convrot.safetensors")
ap.add_argument("--cond", default=os.path.join(_ROOT, "lodx_dit/probe_out/cond.pt"))
ap.add_argument("--skip-ref", action="store_true",
                help="only run the TP side (for timing a big shape)")
a = ap.parse_args(_A)

comfy_aimdo.control.init_devices(
    (d.index, 0) for d in comfy.model_management.get_all_torch_devices())
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True

model = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", a.unet))
cond = torch.load(a.cond, weights_only=False)
latent, frames = _empty_av_latent(a.width, a.height, a.length)
print(f"{a.width}x{a.height} length={a.length} -> {frames} frames, "
      f"{a.steps} steps, all-reduce {a.mode}", flush=True)


# capture the first DiT forward of each run: a whole sampler trajectory
# amplifies any difference, so comparing one forward separates "my block code
# is wrong" from "diffusion is chaotic"
import comfy.ldm.minimax.model as _h3
_first = {}
_real_fwd = _h3.MiniMaxH3Model._forward


def _capture(self, *args, **kwargs):
    out = _real_fwd(self, *args, **kwargs)
    if "cur" not in _first:
        got = out[0] if isinstance(out, (tuple, list)) else out
        _first["cur"] = [c.detach().float().reshape(-1).cpu()
                         for c in (got.unbind() if hasattr(got, "unbind")
                                   else [got])]
    return out


_h3.MiniMaxH3Model._forward = _capture


def sample():
    """Returns the latent flattened to one vector -- it is a NestedTensor of
    (video, audio), so unbind before comparing."""
    torch.manual_seed(a.seed)
    t = time.perf_counter()
    out = nodes.common_ksampler(model, a.seed, a.steps, 1.0, "euler", "simple",
                                cond, cond, latent, denoise=1.0)[0]["samples"]
    torch.cuda.synchronize()
    flat = torch.cat([c.float().reshape(-1).cpu() for c in out.unbind()])
    first = torch.cat(_first.pop("cur")) if "cur" in _first else None
    return flat, time.perf_counter() - t, first


ref = None
if not a.skip_ref:
    ref, t_ref, ref1 = sample()
    print(f"\n1 GPU   {t_ref:7.2f}s  |x| = {ref.norm():.4f}", flush=True)

from lodx_dit.tp import install
install(a.gpus, mode=a.mode, exact_rows=a.exact_rows)

got, t_tp, got1 = sample()
print(f"{a.gpus} GPU TP {t_tp:7.2f}s  |x| = {got.norm():.4f}", flush=True)

if ref is not None:
    rel = (got - ref).norm() / ref.norm()
    if ref1 is not None and got1 is not None:
        r1 = (got1 - ref1).norm() / ref1.norm()
        print(f"\nafter ONE DiT forward:                  {r1:.4e}")
    print(f"after the whole {a.steps}-step trajectory:      {rel:.4e}")
    print(f"speedup: {t_ref / t_tp:.2f}x")
    for d in range(a.gpus):
        free, total = torch.cuda.mem_get_info(d)
        print(f"  cuda:{d} {(total-free)/2**30:5.1f} / {total/2**30:.1f} GiB used")

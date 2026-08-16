"""Where does a MiniMax H3 step actually go?

The attention numbers in ``docs/lod-dit-results.md`` §2.1 are standalone, and
the DiT totals in §2.2 divide the linear layers by a measured GEMM rate.  This
times the real forward instead, component by component with CUDA events, so
"everything else" stops being a guess.  It is what produced §2.3.

    CUDA_VISIBLE_DEVICES=1 python lodx_dit/dit_profile.py --length 48

Read step 2, not step 1: the first forward pays for DynamicVRAM streaming the
weights in (16.2 s of a 5 s step at 640x640).

``--cond`` wants a saved conditioning pair, which ``ab_h3.py`` writes out.  The
profile is of the DiT alone, so what the prompt says does not matter -- only
that the row count is realistic.
"""

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

import comfy.ldm.minimax.model as h3
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
ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--unet", default="minimax_h3_fl2va_int8_convrot.safetensors")
ap.add_argument("--cond", default=os.path.join(_ROOT, "lodx_dit/ab_out/cond.pt"))
a = ap.parse_args(_A)

ok = comfy_aimdo.control.init_devices(
    (d.index, 0) for d in comfy.model_management.get_all_torch_devices())
comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
comfy.memory_management.aimdo_enabled = True
print(f"DynamicVRAM {ok}  device {comfy.model_management.get_torch_device()}",
      flush=True)

ACC = {}


class Timer:
    """CUDA-event scope that accumulates into ACC under one name."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.s = torch.cuda.Event(True)
        self.e = torch.cuda.Event(True)
        self.s.record()
        return self

    def __exit__(self, *exc):
        self.e.record()
        self.e.synchronize()
        ACC[self.name] = ACC.get(self.name, 0.0) + self.s.elapsed_time(self.e)


def wrap(cls, name, attr="forward"):
    real = getattr(cls, attr)

    def inner(self, *args, **kwargs):
        with Timer(name):
            return real(self, *args, **kwargs)
    setattr(cls, attr, inner)
    return real


# the attention module owns qkv_proj/out_proj too, so time the quadratic core
# separately or the projections hide inside it
real_attn_core = h3.optimized_attention


def attn_core(*args, **kwargs):
    with Timer("  attention core (S^2)"):
        return real_attn_core(*args, **kwargs)


h3.optimized_attention = attn_core
wrap(h3.Attention, "attention total")
wrap(h3.MLP, "mlp (fc1+fc2)")
wrap(h3.AdalnProj, "adaLN proj")
wrap(h3.DiTBlock, "blocks total")
wrap(h3.FinalLayer, "final layer")

real_forward = h3.MiniMaxH3Model._forward
STEP = {"n": 0}


def forward(self, *args, **kwargs):
    ACC.clear()
    torch.cuda.synchronize()
    t = time.perf_counter()
    r = real_forward(self, *args, **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    STEP["n"] += 1
    blocks = ACC.get("blocks total", 0.0)
    attn = ACC.get("attention total", 0.0)
    core = ACC.get("  attention core (S^2)", 0.0)
    mlp = ACC.get("mlp (fc1+fc2)", 0.0)
    adaln = ACC.get("adaLN proj", 0.0)
    rows = [
        ("attention core (S^2)", core),
        ("attention qkv/out proj", attn - core),
        ("mlp fc1+fc2 (INT8)", mlp),
        ("adaLN proj", adaln),
        ("norms + mod (in-place)", blocks - attn - mlp - adaln),
        ("final layer", ACC.get("final layer", 0.0)),
        ("pre/post (patchify, rope, embed)",
         dt * 1000 - blocks - ACC.get("final layer", 0.0)),
    ]
    print(f"\n--- step {STEP['n']}   forward {dt:.2f}s ---", flush=True)
    for n, ms in rows:
        print(f"  {n:34} {ms/1000:7.2f}s  {100*ms/(dt*1000):5.1f}%", flush=True)
    return r


h3.MiniMaxH3Model._forward = forward

model = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", a.unet))
cond = torch.load(a.cond, weights_only=False)
latent, frames = _empty_av_latent(a.width, a.height, a.length)
print(f"{a.width}x{a.height} length={a.length} -> {frames} frames  "
      f"latent {tuple(latent['samples'].unbind()[0].shape)}", flush=True)
nodes.common_ksampler(model, 1234, a.steps, 1.0, "euler", "simple",
                      cond, cond, latent, denoise=1.0)

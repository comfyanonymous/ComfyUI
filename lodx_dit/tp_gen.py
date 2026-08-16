"""Generate a real clip twice -- one GPU and tensor-parallel -- and write both.

``tp_run.py`` compares latents; this one decodes and muxes, because whether a
2.2e-2 difference in the latent matters is a question about pixels, not norms.

    CUDA_VISIBLE_DEVICES=0,1 COMFYUI_ENABLE_MIOPEN=1 \
        python lodx_dit/tp_gen.py --width 640 --height 640 --length 72 --steps 20

Writes ``lodx_dit/tp_out/{ref,tp}.mp4``.  Watch both.
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

# TP and DynamicVRAM do not mix: the streamer owns the weights, so replacing
# the four linears with shards ADDS 9.6 GiB instead of saving any, and the
# model's own copies cannot be freed ("HostBuffer.truncate failed" if you try).
# 32 GiB model + 9.6 GiB shards = the 41.6 GiB OOM, at every shape.
# Without it the load is slow but the weights are ordinary tensors we can drop.
_NO_DVRAM = "--no-dynamic-vram" in _A
if _NO_DVRAM:
    _A = [x for x in _A if x != "--no-dynamic-vram"]
else:
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

from lodx_dit.ab_h3 import decode, write_video

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=640)
ap.add_argument("--length", type=int, default=72, help="frames; 72 = 3 s at 24 fps")
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--gpus", type=int, default=2)
ap.add_argument("--mode", default="bf16", choices=["bf16", "int8", "fp8"])
ap.add_argument("--exact-rows", action="store_true")
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--skip-ref", action="store_true")
ap.add_argument("--vram-headroom", type=float, default=20.0,
                help="GiB to keep free per GPU. DynamicVRAM otherwise parks the "
                     "whole 32 GiB model resident and the shards have nowhere "
                     "to go; with headroom it evicts the four linears we never "
                     "call and streams what it still needs.")
ap.add_argument("--ref-only", action="store_true",
                help="no TP at all; run each side in its own process so the "
                     "34 GB model and the shards never have to coexist")
ap.add_argument("--unet", default="minimax_h3_fl2va_int8_convrot.safetensors")
ap.add_argument("--cond", default=os.path.join(_ROOT, "lodx_dit/probe_out/cond.pt"))
a = ap.parse_args(_A)

OUT = os.path.join(_ROOT, "lodx_dit/tp_out")
os.makedirs(OUT, exist_ok=True)

if not _NO_DVRAM:
    _hr = int(a.vram_headroom * 1024 ** 3)
    comfy_aimdo.control.init_devices(
        (d.index, _hr) for d in comfy.model_management.get_all_torch_devices())
    print(f"DynamicVRAM on, {a.vram_headroom:.0f} GiB headroom per GPU", flush=True)
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
else:
    print("DynamicVRAM off: slow load, but the shards can replace the weights",
          flush=True)

model = comfy.sd.load_diffusion_model(
    folder_paths.get_full_path_or_raise("diffusion_models", a.unet))
cond = torch.load(a.cond, weights_only=False)
latent, frames = _empty_av_latent(a.width, a.height, a.length)
print(f"{a.width}x{a.height} length={a.length} -> {frames} frames, {a.steps} steps",
      flush=True)


def sample(tag):
    torch.manual_seed(a.seed)
    t = time.perf_counter()
    out = nodes.common_ksampler(model, a.seed, a.steps, 1.0, "euler", "simple",
                                cond, cond, latent, denoise=1.0)[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    print(f"[{tag}] sampled in {dt:.1f}s  ({dt/a.steps:.2f} s/step)", flush=True)
    return out, dt


results = {}
if a.ref_only:
    results["ref"], _ = sample("1 GPU")
    torch.save(results["ref"], os.path.join(OUT, "ref_latent.pt"))
    import lodx_dit.tp as tp
else:
    if not a.skip_ref:
        results["ref"], t_ref = sample("1 GPU")
        torch.save(results["ref"], os.path.join(OUT, "ref_latent.pt"))
    from lodx_dit import tp
    tp.install(a.gpus, mode=a.mode, exact_rows=a.exact_rows,
               release=_NO_DVRAM)
    results["tp"], t_tp = sample(f"{a.gpus} GPU TP")
    torch.save(results["tp"], os.path.join(OUT, "tp_latent.pt"))
    if not a.skip_ref:
        print(f"\nspeedup: {t_ref / t_tp:.2f}x", flush=True)

# the VAE needs the room the shards are sitting in
tp.free()
comfy.model_management.unload_all_models()
comfy.model_management.soft_empty_cache()

for tag, lat in results.items():
    img, aud = decode(lat, a.width, a.height, a.length)
    write_video(img, aud, os.path.join(OUT, f"{tag}.mp4"))
    torch.save({"images": img}, os.path.join(OUT, f"{tag}_frames.pt"))

if len(results) == 2:
    from lodx_dit.comfy_node import CompareImageBatches
    r = CompareImageBatches.execute(
        torch.load(os.path.join(OUT, "ref_frames.pt"))["images"],
        torch.load(os.path.join(OUT, "tp_frames.pt"))["images"], 5).result[0]
    print("\n" + r, flush=True)

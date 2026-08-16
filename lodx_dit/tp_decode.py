"""Decode tp_gen.py's latents, working around two separate memory problems.

1. The 34 GB DiT stays resident under DynamicVRAM even after
   ``unload_all_models``, so decoding cannot share a process with sampling.
2. Autograd. ``comfy.sd.VAE.decode`` does not disable it -- the only place
   ComfyUI does is ``execution.py:751``, where the prompt executor wraps the
   whole run in ``torch.inference_mode()``. Call the VAE from a script and the
   graph is built and kept: ~9 GiB of saved activations per call, invisible to
   ``gc`` (they live in C++ autograd nodes) and untouched by ``empty_cache``.
   Measured allocated over successive single-frame decodes:
   13.95 -> 22.98 -> 32.02 -> 41.06 GiB, and ``.detach()`` does not help.
   Under ``inference_mode`` it stays flat at the weights' 4.89 GiB.

    python lodx_dit/tp_decode.py --tag tp
"""
from __future__ import annotations

import argparse, glob, os, sys

_A = sys.argv[1:]
sys.argv = [sys.argv[0]]
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="tp")
ap.add_argument("--frames", help="latent frame range i:j")
ap.add_argument("--assemble", action="store_true")
ap.add_argument("--fps", type=int, default=24)
a = ap.parse_args(_A)

OUT = os.path.join(_ROOT, "lodx_dit/tp_out")
PARTS = os.path.join(OUT, "parts")
os.makedirs(PARTS, exist_ok=True)
LAT = os.path.join(OUT, f"{a.tag}_latent.pt")


def _vae(name):
    import comfy.sd, comfy.utils, folder_paths
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        folder_paths.get_full_path_or_raise("vae", name)))


if a.frames:
    i, j = (int(x) for x in a.frames.split(":"))
    v = torch.load(LAT, weights_only=False)["samples"].unbind()[0]
    with torch.inference_mode():                 # what execution.py:751 does
        out = _vae("minimax_h3_video_vae_fp16.safetensors").decode(v[:, :, i:j])
        if out.dim() == 5:
            out = out.reshape(-1, *out.shape[-3:])
        torch.save(out.clone().cpu(), os.path.join(PARTS, f"{a.tag}_{i:03d}.pt"))
    print(f"decoded {i}:{j} -> {out.shape[0]} images", flush=True)

elif a.assemble:
    from lodx_dit.ab_h3 import write_video
    files = sorted(glob.glob(os.path.join(PARTS, f"{a.tag}_*.pt")))
    if not files:
        raise SystemExit("no parts; run the workers first")
    images = torch.cat([torch.load(f, weights_only=False) for f in files])
    al = torch.load(LAT, weights_only=False)["samples"].unbind()[-1]
    with torch.inference_mode():
        avae = _vae("minimax_h3_audio_vae_fp32.safetensors")
        audio = avae.decode(al).movedim(-1, 1).clone().cpu()
        rate = int(avae.first_stage_model.output_sample_rate)
    write_video(images, {"waveform": audio, "sample_rate": rate},
                os.path.join(OUT, f"{a.tag}.mp4"), fps=a.fps)
    torch.save({"images": images}, os.path.join(OUT, f"{a.tag}_frames.pt"))
    print(f"{a.tag}: {images.shape[0]} frames from {len(files)} parts", flush=True)
else:
    raise SystemExit("pass --frames i:j or --assemble")

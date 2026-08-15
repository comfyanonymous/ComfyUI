"""Run the same MiniMax H3 generation dense and with the LoD read, then compare.

One process, one model load, one seed, two samplings.  The only difference
between the branches is whether the attention wrapper has a config to act on --
without one it is a bit-identical pass-through, which is what makes the
comparison an ablation rather than two similar runs.

    python lodx_dit/ab_h3.py --steps 20 --top-pages 128

Writes to ``lodx_dit/ab_out``: an mp4 per branch (audio muxed), the decoded
frames as .pt so the comparison can be redone without resampling, and a report.

The numbers it prints narrow down where to look; they do not decide whether the
result is acceptable.  Watch both files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from fractions import Fraction

_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DynamicVRAM streams weights during the forward instead of transferring the
# whole checkpoint up front, and main.py gates it on ROCm >= 7.14 -- so on an
# older runtime it is off and a 27 GB quantized load blocks for minutes.  The
# allocator has to be hooked before torch touches the GPU, hence this sits
# above the import rather than in main().
_DYNAMIC_VRAM = "--dynamic-vram" in _ARGV
if _DYNAMIC_VRAM:
    _ARGV = [x for x in _ARGV if x != "--dynamic-vram"]
    import comfy_aimdo.control
    comfy_aimdo.control.init()

import torch


def _enable_dynamic_vram():
    """The second half of main.py's setup, once model_management knows the GPUs."""
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    ok = comfy_aimdo.control.init_devices(
        (d.index, 0) for d in comfy.model_management.get_all_torch_devices())
    if not ok:
        raise RuntimeError("comfy-aimdo init_devices failed")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_out")
PROMPT = ("A calico cat naps on a sunlit windowsill, tail flicking, while "
          "leaves move outside; soft ambient room tone.")


def log(msg, t0=[None]):
    if t0[0] is None:
        t0[0] = time.time()
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def encode_prompt(path):
    """Text encoding is its own phase so the 27 GB encoder never coexists with
    the 34 GB DiT, and so a re-run can skip it."""
    import comfy.model_management
    import comfy.sd
    import folder_paths

    if os.path.exists(path):
        log(f"conditioning cached: {path}")
        return torch.load(path, weights_only=False)
    te = folder_paths.get_full_path_or_raise(
        "text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
    log(f"loading text encoder ({os.path.getsize(te)/1e9:.1f} GB)")
    clip = comfy.sd.load_clip([te], clip_type=comfy.sd.CLIPType.MINIMAX)
    log("encoding prompt")
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    payload = [[c[0].cpu(), {k: (v.cpu() if torch.is_tensor(v) else v)
                             for k, v in c[1].items()}] for c in cond]
    os.makedirs(OUT, exist_ok=True)
    torch.save(payload, path)
    del clip, cond
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    log("text encoder released")
    return payload


def _contiguous_attention():
    """Hand SDPA contiguous q/k/v.

    ``comfy/ldm/minimax/model.py:181`` builds them as
    ``q.transpose(0, 1).unsqueeze(0)`` from an (s, heads, dim) buffer, which is
    not contiguous.  Measured at H3's shape on a W7900: 722.8 ms contiguous,
    2145.4 ms as handed over, 764.9 ms after copying -- so the copy pays for
    itself four times over.  Applied to both branches so the ablation stays
    about the read, not about this.
    """
    import comfy.ldm.minimax.model as h3
    from comfy.ldm.modules.attention import AttentionTensorContainer

    real = h3.optimized_attention

    def attn(q, k, v, heads, *args, **kwargs):
        q, k, v = (AttentionTensorContainer(t.take().contiguous())
                   for t in (q, k, v))
        return real(q, k, v, heads, *args, **kwargs)

    h3.optimized_attention = attn


def sample(model, cond, latent, seed, steps, sampler, scheduler):
    import nodes
    return nodes.common_ksampler(model, seed, steps, 1.0, sampler, scheduler,
                                 cond, cond, latent, denoise=1.0)[0]


def decode(latent, width, height, length):
    """Video through the video VAE, audio through the audio VAE."""
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import folder_paths

    samples = latent["samples"]
    video_latent, audio_latent = samples.unbind()[0], samples.unbind()[-1]

    vae_path = folder_paths.get_full_path_or_raise(
        "vae", "minimax_h3_video_vae_fp16.safetensors")
    log(f"loading video vae ({os.path.getsize(vae_path)/1e9:.1f} GB)")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    images = vae.decode(video_latent)
    if images.dim() == 5:
        images = images.reshape(-1, *images.shape[-3:])
    del vae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    apath = folder_paths.get_full_path_or_raise(
        "vae", "minimax_h3_audio_vae_fp32.safetensors")
    log("loading audio vae")
    avae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(apath))
    audio = avae.decode(audio_latent).movedim(-1, 1).to(audio_latent.device)
    rate = int(avae.first_stage_model.output_sample_rate)
    del avae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    return images.cpu(), {"waveform": audio.cpu(), "sample_rate": rate}


def write_video(images, audio, path, fps=24):
    from comfy_api.latest._input_impl.video_types import VideoFromComponents
    from comfy_api.latest._util.video_types import VideoComponents
    VideoFromComponents(VideoComponents(
        images=images, frame_rate=Fraction(fps), audio=audio)).save_to(path)
    log(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=124, help="frames at 24 fps")
    ap.add_argument("--top-pages", type=int, default=128)
    ap.add_argument("--select-block", type=int, default=64)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--local-radius", type=int, default=0)
    ap.add_argument("--start-percent", type=float, default=0.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-contiguous-fix", action="store_true",
                    help="leave the non-contiguous SDPA inputs as the model "
                         "builds them (3.5x slower per step here)")
    ap.add_argument("--skip-dense", action="store_true",
                    help="reuse a previous dense run's frames")
    a = ap.parse_args(_ARGV)
    os.makedirs(OUT, exist_ok=True)
    tag = a.tag or f"kp{a.top_pages}_sb{a.select_block}_s{a.steps}"

    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent
    from lodx_dit.comfy_node import CompareImageBatches, MiniMaxH3LoDAttention

    if _DYNAMIC_VRAM:
        _enable_dynamic_vram()
        log("DynamicVRAM enabled (main.py gates this off below ROCm 7.14)")
    if not a.no_contiguous_fix:
        _contiguous_attention()
        log("contiguous-attention fix applied to BOTH branches")

    log(f"A/B  {a.width}x{a.height} {a.length}f  steps={a.steps} "
        f"seed={a.seed}  top_pages={a.top_pages}")
    cond = encode_prompt(os.path.join(OUT, "cond.pt"))

    unet = folder_paths.get_full_path_or_raise(
        "diffusion_models", "minimax_h3_fl2va_int8_convrot.safetensors")
    log(f"loading DiT ({os.path.getsize(unet)/1e9:.1f} GB)")
    t = time.time()
    model = comfy.sd.load_diffusion_model(unet)
    log(f"DiT loaded in {time.time()-t:.0f}s")

    frames = {}
    audios = {}
    times = {}
    dense_pt = os.path.join(
        OUT, f"dense_{a.width}x{a.height}_{a.length}f_s{a.steps}_{a.seed}.pt")
    for branch in ("dense", "lod"):
        if branch == "dense" and a.skip_dense and os.path.exists(dense_pt):
            blob = torch.load(dense_pt, weights_only=False)
            frames["dense"], audios["dense"] = blob["images"], blob["audio"]
            log("reused cached dense frames")
            continue
        m = model
        if branch == "lod":
            m = MiniMaxH3LoDAttention.execute(
                model, a.top_pages, a.select_block, a.page_size,
                a.local_radius, True, a.start_percent, 1.0).result[0]
        latent, _ = _empty_av_latent(a.width, a.height, a.length)
        log(f"sampling [{branch}] ...")
        t = time.time()
        out = sample(m, cond, latent, a.seed, a.steps, a.sampler, a.scheduler)
        times[branch] = time.time() - t
        log(f"[{branch}] {times[branch]:.0f}s "
            f"({times[branch]/a.steps:.1f}s/step)")
        img, aud = decode(out, a.width, a.height, a.length)
        frames[branch], audios[branch] = img, aud
        if branch == "dense":
            torch.save({"images": img, "audio": aud}, dense_pt)
        write_video(img, aud, os.path.join(
            OUT, f"{branch}{'' if branch == 'dense' else '_' + tag}.mp4"))

    report = CompareImageBatches.execute(frames["dense"], frames["lod"], 5).result[0]
    speed = (f"dense {times['dense']:.0f}s / lod {times['lod']:.0f}s = "
             f"{times['dense']/times['lod']:.2f}x"
             if len(times) == 2 else "dense reused, no timing")
    text = (f"{a.width}x{a.height} {a.length}f steps={a.steps} seed={a.seed}\n"
            f"top_pages={a.top_pages} select_block={a.select_block} "
            f"page_size={a.page_size} local_radius={a.local_radius} "
            f"start_percent={a.start_percent}\n{speed}\n\n{report}\n")
    print("\n" + text, flush=True)
    with open(os.path.join(OUT, f"report_{tag}.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, f"report_{tag}.json"), "w") as f:
        json.dump({"config": vars(a), "times": times, "report": report}, f,
                  indent=1)


if __name__ == "__main__":
    main()

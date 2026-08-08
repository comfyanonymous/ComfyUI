# Render notes — AMD RX 6800 XT + DirectML (ComfyUI)

Verified end-to-end on 2026-08-08: `output/AnimateDiff_00001.mp4` (256x256,
8 frames, 12 steps, euler) rendered successfully via `workflows/_test_render.py`.

## Hard blocker discovered
`torch-directml` exposes only **1024 MB VRAM** to PyTorch on this card, even
though the RX 6800 XT has 16 GB. Confirmed in ComfyUI's own boot log:
`Total VRAM 1024 MB`. A 512x512 x16-frame AnimateDiff batch OOMs the sampler:

```
RuntimeError: Could not allocate tensor with 268435456 bytes.
There is not enough GPU video memory available!
```

So under DirectML you MUST run small: 256x256, 8 frames, ~12 steps. This is a
backend limitation, not a setup bug.

## Pitfalls fixed (so you don't hit them again)
1. **Boot crash:** `tokenizers>=0.22.0,<=0.23.0 required but found 0.23.1`.
   Fix: pin `tokenizers==0.22.2` in the ComfyUI venv. (0.23.x removed a kwarg
   the installed transformers needs.) Also clear `PYTHONPATH` at launch or an
   inherited venv shadows the local one.
2. **API prompt validation** (`_test_render.py` builds the prompt from
   `/object_info` + the UI JSON): required slots `loop_count` (VHS_VideoCombine)
   and `seed_gen` (ADE_AnimateDiffSamplingSettings) have `None` API defaults and
   must be set explicitly.
3. **Legacy AnimateDiff loader:** `ADE_AnimateDiffLoaderWithContext` resolves to a
   legacy class whose `load_mm_and_inject_params()` rejects the
   `deprecation_warning` key (even as `None`). Drop it from the prompt inputs.

## Recommendation: switch to ZLUDA
DirectML is the wrong backend for AMD video. patientx/ComfyUI-Zluda is a
Windows fork that auto-detects the GPU and installs ROCm + PyTorch + triton +
sage-attention, exposing full VRAM. For the 6800 (Navi 21 / RDNA2) it requires
AMD drivers **above 25.5.1** — this machine is on 32.0.21043.19003, which is
eligible.

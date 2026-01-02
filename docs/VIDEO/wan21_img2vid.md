# Wan 2.1 Video Generation

[Back to Index](INDEX.md)

## Overview

Wan 2.1 is a state-of-the-art video generation model by Alibaba. Multiple variants are available for different use cases.

## Installed Models

### Phantom 1.3B (Subject-to-Video)

| Parameter | Value |
|-----------|-------|
| **Model** | Phantom-Wan-1_3B_fp16.safetensors |
| **Location** | `models/diffusion_models/wan/` |
| **Size** | 2.87 GB |
| **Source** | [Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy) |
| **Type** | Subject-to-Video (reference image animation) |
| **VRAM** | ~10-12 GB |

**Note:** Phantom 1.3B is designed for subject animation - it takes a reference image and animates it based on text prompts. It's optimized for lower VRAM usage.

## Required Components

| Component | File | Location | Size |
|-----------|------|----------|------|
| **Diffusion Model** | Phantom-Wan-1_3B_fp16.safetensors | `models/diffusion_models/wan/` | 2.87 GB |
| **Text Encoder** | umt5-xxl-enc-fp8.safetensors | `models/clip/umt5/` | 2.87 GB |
| **VAE** | Wan2_1_VAE_bf16.safetensors | `models/vae/` | 242 MB |

## Custom Nodes Required

| Node | Repository | Purpose |
|------|------------|---------|
| ComfyUI-WanVideoWrapper | [Kijai/ComfyUI-WanVideoWrapper](https://github.com/Kijai/ComfyUI-WanVideoWrapper) | Wan 2.1 integration |
| ComfyUI-VideoHelperSuite | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | Video export |

## Capabilities

- **Input:** Single image (e.g., Flux-generated portrait)
- **Output:** 33 frames @ 8fps = ~4 seconds video
- **Resolution:** 480p (480x832 portrait, 832x480 landscape)
- **Motion:** Realistic human movement, hair, clothing
- **Style:** Photorealistic, matches input image style

## Recommended Settings

| Setting | Value |
|---------|-------|
| **Resolution** | 480x832 (portrait) or 832x480 (landscape) |
| **Frames** | 33 (4 seconds @ 8fps) |
| **Steps** | 25 |
| **CFG** | 5.0 |
| **Sampler** | euler |
| **Scheduler** | simple |

## Sample Prompts

**Head turn with smile:**
```
A beautiful woman slowly turning her head and smiling, natural movement, realistic skin, cinematic lighting
```

**Hair blowing:**
```
Woman with long hair blowing in the wind, gentle breeze, natural movement, outdoor lighting
```

**Looking at camera:**
```
Woman looking directly at camera, subtle breathing, natural micro-movements, professional lighting
```

## VRAM Usage

| Phase | VRAM |
|-------|------|
| Model loading | ~8 GB |
| Generation (480p, 33 frames) | ~10-11 GB |
| Peak | ~11 GB |

With 12GB VRAM, the model uses offloading to fit in memory.

## Generation Time

| Resolution | Frames | Time (RTX 3000 series) |
|------------|--------|------------------------|
| 480p | 33 | 2-5 minutes |

## Workflow

Tutorial workflow: `LEARN_Wan21_Img2Vid.json`

**Nodes used:**
1. LoadImage - source image
2. DownloadAndLoadWanModel - load 1.3B model
3. DownloadAndLoadWanVAE - load VAE
4. DownloadAndLoadWanTextEncoder - load UMT5
5. WanTextEncode - encode motion prompt
6. WanImageToVideo - generate video latents
7. WanVAEDecode - decode to frames
8. VHS_VideoCombine - export to MP4

## Tips

- Use high-quality source images (Flux works great)
- Keep prompts focused on motion, not appearance
- Start with 25 steps, increase for quality
- 33 frames is good balance of length and speed
- Source image defines appearance, prompt defines motion

## Alternatives

| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| **Wan 2.1 1.3B** | 12GB | Good | Medium |
| Wan 2.1 14B | 24GB+ | Best | Slow |
| LTX Video | 8GB | Medium | Fast |
| CogVideoX | 16GB+ | Good | Slow |

## Notes

- Phantom variant is specifically optimized for lower VRAM
- FP16 precision balances quality and memory
- UMT5 text encoder is different from Flux CLIP
- First generation may be slower (model caching)

---

**Category:** Video Generation
**Installed:** 2026-01-02

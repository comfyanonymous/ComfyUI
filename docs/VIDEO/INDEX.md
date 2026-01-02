# Video Generation

[Back to Main Index](../INDEX.md)

Video generation models and workflows for creating short videos from images or text.

---

## Installed Models

| Model | Type | Resolution | VRAM | Status |
|-------|------|------------|------|--------|
| [Wan 2.1 Phantom 1.3B](wan21_img2vid.md) | Subject-to-Video | 480p | 10-12GB | Installed |

---

## Required Components

| Component | File | Location | Status |
|-----------|------|----------|--------|
| Diffusion Model | `Phantom-Wan-1_3B_fp16.safetensors` | `models/diffusion_models/wan/` | Installed |
| Text Encoder | `umt5-xxl-enc-fp8.safetensors` | `models/clip/umt5/` | Installed |
| VAE | `Wan2_1_VAE_bf16.safetensors` | `models/vae/` | Installed |
| CLIP Vision | `clip_vision_h.safetensors` | `models/clip_vision/` | Downloading |

## Custom Nodes

| Node | Purpose | Status |
|------|---------|--------|
| ComfyUI-WanVideoWrapper | Wan 2.1 video generation | Installed |
| ComfyUI-VideoHelperSuite | Video export (MP4, GIF) | Installed |
| ComfyUI-KJNodes | Image resize utilities | Installed |

---

## Workflows

| Workflow | Purpose | Status |
|----------|---------|--------|
| `Wan21_Phantom_Subject2Vid.json` | Phantom 1.3B: Animate subject from reference image | Ready |
| `Wan21_I2V_OFFICIAL.json` | I2V example (adapted for 1.3B, needs CLIP Vision) | Downloading |

**Note:** Both workflows have been configured for the installed Phantom 1.3B model. The I2V workflow requires CLIP Vision model (downloading ~1.26 GB).

---

## Quick Start (Phantom 1.3B)

1. Load workflow: `Wan21_Phantom_Subject2Vid.json`
2. Load source image (e.g., Flux-generated portrait)
3. Write motion prompt: "woman slowly turning head and smiling"
4. Set model to `wan\Phantom-Wan-1_3B_fp16.safetensors`
5. Queue prompt (generation takes 2-5 minutes)
6. Output: MP4 video file

---

## Model Types Explained

| Type | Description | Model Needed |
|------|-------------|--------------|
| **T2V** | Text-to-Video - generate video from text only | T2V models |
| **I2V** | Image-to-Video - animate a single image | I2V models (14B) |
| **Subject2Vid** | Animate subject from reference image | Phantom 1.3B |

---

*Last updated: 2026-01-02*

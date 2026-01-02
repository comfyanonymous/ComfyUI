# 15. Upscalers

[← Back to Index](INDEX.md)

Models for upscaling/enlarging images. Located in `models/upscale_models/`.

## Table of Contents

- [4x Foolhardy Remacri](#4x-foolhardy-remacri) - **RECOMMENDED** Best for people/NSFW
- [4x UltraSharp](#4x-ultrasharp) - Very sharp, general purpose
- [RealESRGAN x4plus](#realesrgan-x4plus) - Standard balanced upscaler

---

## 4x Foolhardy Remacri

> **RECOMMENDED UPSCALER** - Best for people, faces, and NSFW content

| Parameter | Value |
|-----------|-------|
| **File** | `4x_foolhardy_Remacri.safetensors` |
| **Scale** | 4x |
| **License** | CC-BY-NC-SA-4.0 |
| **Creator** | FoolhardyVEVO |

### Description
High-quality 4x upscaler optimized for realistic images, especially people and faces. Preserves natural skin textures without creating plastic-looking artifacts. Very popular in NSFW community for its ability to maintain realistic details during upscaling.

### Best for
- Human subjects (portraits, full body)
- NSFW content
- Realistic photography
- Skin textures and details
- Faces

### Usage
Works with:
- Pony Diffusion
- Illustrious XL
- SDXL
- Flux (in img2img/hires workflows)

### Workflow integration
Used in img2img-upscale and img2img-hires workflows:
- Typical upscale: 1.5x - 2x
- Resolution: 832x1216 → 2688x3840

### Notes
- Less "plastic" effect than other upscalers
- Maintains natural textures
- Good balance of sharpness and smoothness
- Works well with all base models

---

## 4x UltraSharp

| Parameter | Value |
|-----------|-------|
| **File** | `4x-UltraSharp.pth` |
| **Scale** | 4x |

### Description
Very sharp upscaler for general purpose use. Can sometimes over-sharpen, creating slightly artificial look.

### Best for
- General images
- Landscapes
- Objects
- When maximum sharpness is desired

### Notes
- May over-sharpen skin textures
- Good for non-human subjects
- Creates crisp edges

---

## RealESRGAN x4plus

| Parameter | Value |
|-----------|-------|
| **File** | `RealESRGAN_x4plus.pth` |
| **Scale** | 4x |

### Description
Standard ESRGAN upscaler with good balance. Default choice for many workflows.

### Best for
- General purpose
- When unsure which to use
- Balanced results

### Notes
- Safe default choice
- Well-balanced sharpness
- Works with everything

---

## Comparison

| Upscaler | People/NSFW | Sharpness | Artifacts | Recommended Use |
|----------|-------------|-----------|-----------|-----------------|
| **Remacri** | Excellent | Good | Low | People, NSFW |
| **UltraSharp** | Good | Excellent | Medium | Objects, landscapes |
| **RealESRGAN** | Good | Good | Low | General, default |

---

## Installation

Place `.safetensors` or `.pth` files in:
```
ComfyUI/models/upscale_models/
```

---

[← Back to Index](INDEX.md)

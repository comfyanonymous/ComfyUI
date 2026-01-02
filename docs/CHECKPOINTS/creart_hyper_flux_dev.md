# 8 Steps CreArt-Hyper-Flux-Dev

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Ultimate Hyper Flux Dev merged with ByteDance Hyper 8 steps LoRA for fast, high-quality generation.

## File Information

- **Filename:** 8StepsCreartHyperFlux_v26HyperDevFp8Unet.safetensors
- **Location:** models\diffusion_models\
- **Format:** Unet FP8
- **Version:** v2.6

## Statistics

- **Downloads:** 16,577
- **Rating:** 820
- **Tips:** 8,650
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/699688

## Model Type

Diffusion Model (Unet) in FP8 format - requires separate CLIP L, T5XXL, and VAE.

## Description

The Ultimate Hyper Flux Dev model merged with ByteDance Hyper 8 steps LoRA, creating a fast-generation powerhouse. This model includes merged LoRAs for MoreFace, SkinDetails, and Real-lora, providing enhanced facial features and realistic skin rendering.

Optimized for 8-10 step generation while maintaining high quality, making it ideal for workflows requiring speed without sacrificing detail.

## Key Features

- Merged with ByteDance Hyper 8 steps LoRA
- Integrated MoreFace LoRA for better facial features
- SkinDetails LoRA for realistic skin texture
- Real-lora for enhanced realism
- Fast generation (8-10 steps)
- FP8 precision for VRAM efficiency

## Integrated LoRAs

1. **ByteDance Hyper 8 Steps:** Fast generation capability
2. **MoreFace:** Enhanced facial features and expressions
3. **SkinDetails:** Realistic skin texture and pores
4. **Real-lora:** Overall realism improvement

## Recommended Settings

- **Steps:** 8-10
- **Guidance (CFG):** 3-3.5
- **Sampler:** Euler
- **Scheduler:** Beta

## Technical Details

- **Format:** FP8 Unet (8-bit floating point)
- **Requirements:**
  - CLIP L model
  - T5XXL text encoder
  - FLUX VAE
- **Precision:** FP8 reduces VRAM usage vs FP16

## Use Cases

- Fast high-quality portrait generation
- Realistic skin rendering
- Facial detail work
- Production workflows requiring speed
- VRAM-constrained systems needing quality

## Performance

- **Generation Time:** Very fast (8-10 steps)
- **Quality:** High, thanks to merged LoRAs
- **VRAM:** Reduced compared to FP16 models

## Workflow Integration

As a Unet-only model, use with:
1. FLUX CLIP L text encoder
2. T5XXL text encoder
3. FLUX VAE (ae.safetensors)

## Strengths

- Excellent facial detail and features
- Realistic skin texture with visible pores
- Fast generation without quality loss
- Good balance of speed and detail
- Lower VRAM than full FP16 models

## Notes

- Hyper models require specific sampler settings
- Beta scheduler recommended for best results
- Lower guidance (3-3.5) produces better results
- FP8 format provides good quality with efficiency
- Version 2.6 indicates ongoing refinement

---

**Category:** Fast Generation, Photorealistic
**Last Updated:** 2025-12-31

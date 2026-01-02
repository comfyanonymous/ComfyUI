# Flux1-DedistilledMixTuned

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Pure base model integrating realism from SRPO and artistry from Krea with excellent texture and LoRA compatibility.

## File Information

- **Filename:** flux1_v40Fp8.safetensors
- **Location:** models\diffusion_models\
- **Format:** Diffusion Model (FP8)
- **Version:** V4.0

## Statistics

- **Downloads:** 1,767
- **Rating:** 184
- **Tips:** 0
- **Score:** ⭐

## Links

- **Civitai:** https://civitai.com/models/941929/flux1-dedistilledmixtuned

## Model Type

De-distilled and mixed diffusion model in FP8 format.

## Description

Version 4.0 Pure base model that integrates realism from SRPO training with artistry from Krea. This de-distilled model features excellent texture rendering and superior LoRA compatibility, making it a versatile choice for both realistic and artistic generation.

The de-distillation process reverses the distillation shortcuts, resulting in a model with fuller diffusion capabilities and better quality at the cost of slightly slower generation.

## Key Features

- Integrates SRPO realism training
- Incorporates Krea artistic capabilities
- Excellent texture rendering
- Superior LoRA compatibility
- De-distilled for better quality
- FP8 format for efficiency

## Technical Details

- **Format:** FP8 Diffusion Model
- **Process:** De-distilled (reversed from distilled models)
- **Training:** Mixed with SRPO and Krea
- **Requirements:** CLIP L, T5XXL, VAE

## De-Distillation Explained

De-distillation reverses the shortcuts taken during model distillation:
- Restores full diffusion process
- Improves quality and detail
- Better texture rendering
- Enhanced prompt following
- Slightly slower than distilled models

## LoRA Compatibility

Excellent compatibility with various LoRA types:
- Style LoRAs
- Character LoRAs
- Concept LoRAs
- Detail enhancement LoRAs

## Use Cases

- Realistic photography
- Artistic generation
- Texture-rich scenes
- LoRA experimentation
- Hybrid realistic/artistic styles

## Recommended Settings

Standard FLUX settings work well:
- **Steps:** 20-30
- **CFG Scale:** 3.5-7
- **Sampler:** Euler, DPM++ 2M
- **Scheduler:** Normal, Simple

## Version History

- **V4.0 Pure Base:** Current version with integrated SRPO and Krea

## Strengths

- Balanced realism and artistry
- Excellent texture quality
- Strong LoRA compatibility
- Versatile across styles
- FP8 efficiency

## Notes

- De-distilled models require more steps than distilled
- FP8 format reduces VRAM needs
- Requires external CLIP and VAE
- Good for users wanting quality over speed
- Lower download count but high quality

---

**Category:** Standard Base Models, Photorealistic
**Last Updated:** 2025-12-31

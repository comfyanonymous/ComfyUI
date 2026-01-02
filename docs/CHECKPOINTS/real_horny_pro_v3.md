# Real Horny Pro V3

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Specialized NSFW model with improved realism and excellent LoRA compatibility.

## File Information

- **Filename:** realHornyProV3_realHornyProV2NF4.safetensors
- **Location:** models\unet\
- **Format:** Unet (NF4)
- **Version:** V3 (file is V2 NF4)

## Statistics

- **Downloads:** 27,142
- **Rating:** 1,027
- **Tips:** 3,090
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/684924/real-horny-pro-v3

## Model Type

Specialized NSFW Unet model in NF4 format - requires separate VAE and CLIP models.

## Description

Real Horny Pro V3 is a specialized NSFW model featuring improved realism and excellent LoRA compatibility. The model comes in an "Asian Cuties" version that has a preference for generating Asian women.

As a Unet-only model in NF4 format, it requires external VAE and CLIP models for full functionality, which helps reduce VRAM usage while maintaining quality.

## Key Features

- Improved realism over previous versions
- Excellent LoRA compatibility
- NF4 quantization for lower VRAM usage
- Asian Cuties variant available
- Specialized for NSFW content

## Variants

- **Standard:** General NSFW content
- **Asian Cuties:** Preference for Asian female subjects

## Technical Details

- **Format:** NF4 (4-bit NormalFloat quantization)
- **Type:** Unet only (not full checkpoint)
- **Requirements:** Separate VAE and CLIP models needed

## Use Cases

- High-quality NSFW generation
- LoRA experimentation and stacking
- Asian female subject generation (Asian Cuties version)
- VRAM-constrained systems

## Recommended Settings

Use standard FLUX settings with adjustments for NF4:
- Steps: 20-30
- CFG Scale: 3-7
- Sampler: Euler, DPM++ 2M
- Scheduler: Normal, Simple

## LoRA Compatibility

This model is specifically noted for excellent LoRA compatibility, making it ideal for:
- Stacking multiple LoRAs
- Character-specific LoRAs
- Style LoRAs
- Pose and composition LoRAs

## Notes

- NF4 format reduces VRAM requirements significantly
- Requires compatible VAE (recommend FLUX VAE)
- Requires compatible CLIP models (CLIP L and T5XXL)
- Specialized for NSFW content generation

---

**Category:** NSFW Specialized
**Last Updated:** 2025-12-31

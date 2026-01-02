# FLUX1 Dev GGUF F16

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Standard FLUX Dev model in GGUF format with F16 precision.

## File Information

- **Filename:** flux1-dev-F16.gguf
- **Location:** models\unet\
- **Size:** 22.1 GB
- **Format:** GGUF
- **Precision:** F16 (16-bit floating point)

## Model Type

Standard FLUX Dev GGUF - baseline model for high-quality image generation.

## Description

This is the standard FLUX Dev model in GGUF format with F16 precision. GGUF (GPT-Generated Unified Format) is optimized for efficient loading and inference. The F16 precision maintains high quality while being more memory-efficient than F32.

## Use Cases

- High-quality general-purpose image generation
- Base model for workflows requiring FLUX Dev
- Suitable for systems with adequate VRAM (22+ GB)

## Technical Details

- Full precision F16 maintains excellent quality
- GGUF format for optimized loading
- Requires sufficient VRAM for full model

## Recommended Settings

Use standard FLUX Dev settings:
- Steps: 20-30
- CFG Scale: 3.5-7
- Sampler: Euler, DPM++ 2M
- Scheduler: Simple, Normal

## Notes

- This is a base FLUX Dev model without specialized training
- Compatible with standard FLUX workflows
- Requires VAE and CLIP models for full functionality

---

**Category:** Standard Base Models
**Last Updated:** 2025-12-31

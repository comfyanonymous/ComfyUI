# GGUF FLUX.1 Dev Q8

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Q8 quantized FLUX Dev model optimized for the InfiniteYOU Workflow.

## File Information

- **Filename:** ggufFLUX1DevQ8ModelUsed_v10.gguf
- **Location:** models\unet\
- **Format:** GGUF
- **Quantization:** Q8 (8-bit)

## Links

- **Civitai:** https://civitai.com/models/1452850?modelVersionId=1642703

## Model Type

Quantized FLUX Dev model in GGUF format.

## Description

Q8 quantized version of FLUX.1 Dev specifically designed for use with the InfiniteYOU Workflow. Quantization reduces model size and VRAM requirements while maintaining good quality.

## Use Cases

- InfiniteYOU Workflow implementations
- Systems with limited VRAM
- Faster inference with acceptable quality trade-off

## Technical Details

- Q8 quantization (8-bit) reduces size significantly
- GGUF format for efficient loading
- Optimized for specific workflow compatibility
- Lower VRAM requirements than F16 version

## Advantages

- Smaller file size than F16
- Faster loading times
- Reduced VRAM usage
- Good quality retention despite quantization

## Recommended Settings

Standard FLUX Dev settings work well:
- Steps: 20-30
- CFG Scale: 3.5-7
- Sampler: Euler, DPM++ 2M

## Notes

- Specifically mentioned for InfiniteYOU Workflow
- Q8 provides good balance between quality and efficiency
- Some minimal quality loss compared to F16, but often imperceptible

---

**Category:** Standard Base Models
**Last Updated:** 2025-12-31

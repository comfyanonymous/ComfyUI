# FluxUnchained NF4

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Optimized for lower VRAM GPUs with NF4 quantization for good inference speed.

## File Information

- **Filename:** fluxunchainedNF4_fluxunchainedV11NF4.safetensors
- **Location:** models\unet\
- **Format:** Diffusion Model (NF4)
- **Version:** V1.1

## Statistics

- **Downloads:** 6,361
- **Rating:** 264
- **Tips:** 160
- **Score:** ⭐⭐

## Links

- **Civitai:** https://civitai.com/models/663307/fluxunchained-nf4

## Model Type

NF4 quantized diffusion model optimized for lower VRAM systems.

## Description

FluxUnchained NF4 is specifically optimized for lower VRAM GPUs, successfully running on RTX 3080 10GB cards. The NF4 (4-bit NormalFloat) quantization provides good inference speed while maintaining acceptable quality, making it an excellent choice for users with VRAM constraints.

## Key Features

- Optimized for lower VRAM GPUs
- Works on RTX 3080 10GB
- NF4 quantization for efficiency
- Good inference speed
- Maintains acceptable quality
- Version 1.1 refinements

## VRAM Requirements

- **Minimum:** 10 GB (RTX 3080 tested)
- **Recommended:** 12+ GB for comfortable use
- **Format Advantage:** NF4 uses ~4x less VRAM than FP16

## Performance Characteristics

- **Speed:** Good inference speed for quantized model
- **Quality:** Acceptable quality loss from quantization
- **Efficiency:** Excellent VRAM usage
- **Compatibility:** Works on mid-range GPUs

## NF4 Quantization

NF4 (4-bit NormalFloat) advantages:
- Dramatic VRAM reduction
- Faster inference than FP16
- Maintains reasonable quality
- Good for batch generation

## Use Cases

- RTX 3060/3070/3080 8-12GB GPUs
- Systems with VRAM constraints
- Users wanting FLUX on lower-end hardware
- Batch generation on limited VRAM
- Budget-conscious setups

## Recommended Settings

Standard FLUX settings work:
- **Steps:** 20-30
- **CFG Scale:** 3.5-7
- **Sampler:** Euler, DPM++ 2M
- **Scheduler:** Normal, Simple

## Technical Details

- **Format:** NF4 (4-bit NormalFloat)
- **Type:** Unet (requires VAE and CLIP)
- **Optimization:** VRAM-focused
- **Compatibility:** Mid-range NVIDIA GPUs

## Quality Considerations

NF4 quantization trade-offs:
- **Pros:** Much lower VRAM, faster generation
- **Cons:** Some quality loss vs FP16
- **Result:** Good balance for hardware-limited users

## Comparison with Other Formats

| Format | VRAM | Quality | Speed | File Size |
|--------|------|---------|-------|-----------|
| FP16 | 20+ GB | Highest | Slow | ~22 GB |
| FP8 | 12-16 GB | Very Good | Medium | ~11 GB |
| NF4 | 8-12 GB | Good | Fast | ~5-6 GB |

## Version History

- **V1.1:** Current version with refinements

## Strengths

- Enables FLUX on mid-range GPUs
- Good speed/quality balance
- Proven on RTX 3080 10GB
- Lower VRAM than alternatives

## Notes

- Best choice for 10-12 GB VRAM systems
- "Unchained" suggests fewer restrictions/censorship
- NF4 is currently the best low-VRAM option
- Requires external VAE and CLIP models

---

**Category:** Low VRAM
**Last Updated:** 2025-12-31

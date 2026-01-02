# Fluxed Up NSFW Nunchaku

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Nunchaku INT4 quantized NSFW model - very fast but with occasional quality issues like weird legs.

## File Information

- **Filename:** fluxedUpFluxNSFWCheckpoint_51INT4.safetensors
- **Location:** models\unet\
- **Format:** Nunchaku NVFP4 (INT4)
- **Version:** 5.1

## Statistics

- **Downloads:** 2,185
- **Rating:** 56
- **Tips:** 10
- **Score:** ⭐

## Links

- **Civitai:** https://civitai.com/models/1967499/fluxed-up-flux-nsfw-checkpoint-nunchaku-int4fp4

## Model Type

NSFW Unet model in Nunchaku INT4/FP4 format.

## Description

Fluxed Up NSFW uses Nunchaku's INT4 quantization format for extremely fast generation on compatible hardware. While speed is excellent, the model has some noted quality issues, particularly with leg anatomy which can sometimes appear weird or distorted.

The Nunchaku format is NVIDIA's experimental quantization scheme providing extreme compression.

## Key Features

- Nunchaku INT4/FP4 format
- Very fast generation
- NSFW-focused training
- Extremely low VRAM usage
- Compatible with Turbo LoRA

## Nunchaku Format

Nunchaku INT4 characteristics:
- Experimental NVIDIA quantization
- Extreme compression (4-bit)
- Very fast inference
- Lower quality than standard formats
- Requires compatible hardware/software

## Known Issues

- **Legs:** Can appear weird or distorted
- **Anatomy:** Occasional issues with proportions
- **Consistency:** Variable quality across generations
- **Detail Loss:** More than standard quantization

## Recommended Settings

### With Turbo LoRA
- **Steps:** 8
- **Sampler:** Euler, DPM++ 2M
- **Scheduler:** Beta

### Without Turbo LoRA
- **Steps:** 30
- **Sampler:** DPM++ 2M
- **Scheduler:** Beta

## Performance

- **Speed:** Very fast (especially with Turbo LoRA)
- **VRAM:** Extremely low
- **Quality:** Variable, some issues
- **Consistency:** Lower than standard models

## Use Cases

- Rapid NSFW generation
- Extreme VRAM constraints
- Testing and iteration
- Systems with <8GB VRAM
- When speed is prioritized over quality

## Not Recommended For

- High-quality final outputs
- Professional work
- Leg-focused compositions
- Full-body shots requiring accuracy
- Critical anatomy accuracy

## Technical Details

- **Format:** Nunchaku NVFP4/INT4
- **Compression:** 4-bit quantization
- **Type:** Unet (requires VAE and CLIP)
- **Optimization:** Speed over quality

## Quality Trade-offs

| Aspect | 👍 | Notes |
|--------|--------|-------|
| Speed | Excellent | Very fast generation |
| VRAM Usage | Excellent | Minimal requirements |
| Upper Body | Good | Generally acceptable |
| Lower Body/Legs | Poor | Known issue area |
| Overall Detail | Fair | Loss from quantization |
| Consistency | Fair | Variable results |

## Turbo LoRA Integration

Works well with Turbo LoRA:
- Reduces steps to 8
- Maintains speed advantage
- Some quality improvement
- Good for rapid iteration

## Comparison with Alternatives

For speed:
- **Flux ArtFusion 4-steps:** Better quality, still fast
- **CreArt Hyper 8-steps:** Better quality, similar speed

For NSFW:
- **Real Horny Pro V3:** Much better quality
- **getphat FLUX Reality:** Better anatomy

## Version History

- **V5.1:** Current version

## Notes

- Nunchaku format is experimental
- Quality issues make it niche
- Best for speed-critical workflows
- Lower ratings reflect quality concerns
- Consider alternatives unless speed is critical
- Leg issues are a significant drawback
- INT4 is bleeding-edge compression

---

**Category:** Fast Generation, NSFW Specialized
**Last Updated:** 2025-12-31

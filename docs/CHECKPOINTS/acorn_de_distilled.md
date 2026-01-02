# Acorn De-Distilled V1.5

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

De-Distilled version of Acorn for maximum photorealism with full diffusion process.

## File Information

- **Filename:** acornIsSpinningFLUX_aisfDeDistilledV15.safetensors
- **Location:** models\unet\
- **Size:** 11.9 GB
- **Version:** V1.5 De-Distilled

## Statistics

- **Downloads:** Not specified
- **Rating:** Not specified
- **Tips:** Not specified
- **Score:** ⭐⭐

## Links

- **Civitai:** https://civitai.com/models/673188?modelVersionId=1450688

## Model Type

De-distilled photorealistic FLUX model for maximum quality.

## Description

Acorn De-Distilled V1.5 is the ultimate quality version of Acorn Is Spinning FLUX, utilizing a full diffusion process rather than distilled shortcuts. This version prioritizes maximum photorealism over generation speed, making it ideal for final high-quality outputs.

The de-distillation process restores the complete diffusion steps that were compressed in distilled models, resulting in superior detail and realism.

## Key Features

- Maximum photorealism
- Full diffusion process (de-distilled)
- Superior detail and quality
- Restored complete diffusion steps
- Smaller file size (11.9 GB vs standard)
- Ultimate quality version of Acorn

## De-Distillation Process

De-distillation reverses distillation:
- **Distilled Models:** Compressed steps for speed
- **De-Distilled Models:** Full diffusion for quality
- **Result:** Maximum quality, slower generation
- **Benefit:** Enhanced realism and detail

## Critical Settings

### Distilled CFG
- **MUST be set to 0**
- This disables the distilled guidance
- Required for de-distilled models

### Actual CFG
- **Range:** 3.5-8
- **Recommended:** 4-6
- **Purpose:** Standard guidance control

### Other Settings
- **Steps:** 25-30
- **Sampler:** DPM2, Euler
- **Scheduler:** Normal, Simple

## Settings Configuration

```
Distilled CFG: 0 (CRITICAL - must be 0)
CFG Scale: 3.5-8 (actual guidance)
Steps: 25-30
Sampler: DPM2 or Euler
Scheduler: Normal or Simple
```

## Use Cases

- Final high-quality outputs
- Maximum photorealism requirements
- Professional work
- Print-quality images
- Portfolio pieces
- When quality trumps speed

## Not Recommended For

- Rapid iteration
- Testing prompts
- Batch generation
- Time-sensitive work
- Quick previews

## Workflow Integration

Best workflow approach:
1. Use Hyper 8-Step variant for iteration
2. Test and refine prompts quickly
3. Switch to De-Distilled for final output
4. Generate final high-quality version

## Quality Comparison

| Variant | Quality | Speed | Best For |
|---------|---------|-------|----------|
| De-Distilled | Maximum | Slow | Finals |
| V1.69 | Very High | Medium | General |
| Hyper 8-Step | High | Fast | Iteration |
| Schnell | Good | Very Fast | Testing |

## Technical Details

- **Size:** 11.9 GB (smaller than typical FP16)
- **Format:** Unet (requires VAE and CLIP)
- **Process:** Full de-distilled diffusion
- **Optimization:** Quality over speed

## Strengths

- Absolute maximum photorealism
- Superior detail rendering
- Enhanced texture quality
- Professional-grade output
- Best-in-class realism

## Considerations

- Slower generation (25-30 steps minimum)
- Requires understanding of CFG settings
- Distilled CFG must be 0 (critical)
- Not for rapid workflows
- Best for final outputs only

## Common Mistakes to Avoid

1. **Forgetting Distilled CFG = 0:** Will produce poor results
2. **Too Few Steps:** Need 25-30 minimum
3. **Using for Iteration:** Too slow, use Hyper variant
4. **Wrong Sampler:** DPM2 or Euler recommended

## Prompting Tips

Since quality is maximum:
- Use detailed, descriptive prompts
- Specify fine details
- Include lighting nuances
- Describe materials precisely
- Reference professional photography
- Take advantage of enhanced capability

## Comparison with Standard Acorn

- **De-Distilled V1.5:** Maximum quality, full process
- **Standard V1.69:** Balanced quality/speed
- **Quality Difference:** Noticeable in fine details
- **Speed Difference:** 2-3x slower
- **Use Case:** Finals vs general use

## File Size Note

At 11.9 GB, this is smaller than typical FP16 models (~22 GB):
- Likely optimized format
- Efficient storage
- Maintains quality despite smaller size
- Good VRAM efficiency

## Notes

- Part of Acorn Is Spinning FLUX family
- Ultimate quality variant
- Requires specific CFG configuration
- Distilled CFG = 0 is non-negotiable
- Best saved for final generation
- Slower but worth it for quality
- Version 1.5 suggests refinement from earlier de-distilled versions

---

**Category:** Photorealistic
**Last Updated:** 2025-12-31

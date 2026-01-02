# Jib Mix Flux

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

FLUX Dev trained on SDXL dataset with merged LoRAs, correcting anatomy censorship and excessive bokeh.

## File Information

- **Primary File:** jibMixFlux_v8Accentueight.safetensors
- **Alternative:** jibMixFlux_v85Consisteight.safetensors
- **Location:** models\unet\

## Statistics

- **Downloads:** 55,100
- **Rating:** 1,600
- **Tips:** 470,700
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/686814/jib-mix-flux

## Model Type

FLUX Dev Unet trained on SDXL dataset with merged LoRAs.

## Description

Jib Mix Flux is a FLUX Dev model trained on SDXL dataset with merged LoRAs, specifically designed to correct anatomy censorship and reduce excessive bokeh/blurred backgrounds that plague many FLUX models.

The model offers multiple versions optimized for different use cases, from cleaner outputs to consistent generation to superior skin texture.

## Key Features

- Corrects anatomy censorship issues
- Reduces excessive bokeh and blurred backgrounds
- Multiple versions for different needs
- Trained on SDXL dataset
- Merged LoRAs for enhanced capabilities
- Excellent skin texture (v8)

## Available Versions

- **v12:** Cleaner output, refined results
- **v8.5 (Consisteight):** Most consistent generation
- **v8 (Accentueight):** Best skin texture quality
- **v8-Flash SVDQuant-4bit:** Super fast quantized version

## Recommended Settings

- **Guidance (CFG):** 2.5-3.5
- **Sampler:** dpmpp_2m
- **Steps:** 8-14 (when using with Hyper LoRA)
- **Scheduler:** sgm_uniform (recommended)

## Sample Prompts and Combinations

The model works well with detailed prompts. Here are some effective patterns:

### Portrait Photography
```
Professional portrait, natural lighting, sharp focus, detailed skin texture,
[subject description], clean background
```

### Environmental Shots
```
Full scene composition, balanced depth of field, environmental context,
[subject and setting], natural atmosphere
```

### Detailed Realism
```
Photorealistic, high detail, proper anatomy, natural proportions,
[specific details], professional photography
```

## Use Cases

- Portrait photography with correct anatomy
- Scenes requiring controlled depth of field
- Photorealistic generation
- Professional photography styles
- NSFW content (uncensored anatomy)

## Technical Details

- Based on FLUX Dev architecture
- SDXL dataset training
- Merged LoRAs integrated into base model
- Unet format (requires VAE and CLIP)

## Version Comparison

| Version | Strength | Best For |
|---------|----------|----------|
| v12 | Clean output | Professional, refined results |
| v8.5 | Consistency | Batch generation, uniform style |
| v8 | Skin texture | Portraits, close-ups |
| v8-Flash | Speed | Rapid generation, testing |

## Notes

- Particularly effective at avoiding blurred backgrounds
- Anatomy rendering is uncensored and accurate
- Works well with Hyper LoRA for fast generation
- Popular model with high tip count (470K+)
- Guidance scale lower than standard FLUX (2.5-3.5 vs 3.5-7)

---

**Category:** Photorealistic, NSFW Specialized
**Last Updated:** 2025-12-31

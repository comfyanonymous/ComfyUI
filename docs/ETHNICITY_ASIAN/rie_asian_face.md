# Rie: Asian Face Flux LoRA

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,506 |
| **👍** | 83 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Rie_Asian_Face.safetensors` |
| **Original filename** | `varoriya.rie1.1.safetensors` |
| **Civitai** | https://civitai.com/models/644945 |
| **Trigger word** | `Girl` or `Women` |
| **Strength** | 0.4-0.7 |
| **Type** | CHARACTER |

## Description

Character LoRA for generating Asian female faces with high detail. Version 1.1 was trained on 1024px images (upgraded from 768px in v1.0) for improved facial detail and quality. Works with FLUX.1-dev model.

The LoRA specializes in creating realistic Asian facial features with good skin texture and natural proportions.

## Sample Prompts

**Prompt 1 (Basic portrait):**
```
Girl, portrait of a beautiful asian woman, soft natural lighting, detailed face, looking at camera, black hair, brown eyes
```
Settings: Steps 30, CFG 3.5, Euler, Strength 0.6

**Prompt 2 (Fashion photo):**
```
Women, professional fashion photography, asian model, elegant pose, studio lighting, detailed skin texture, high resolution
```
Settings: Steps 30, CFG 3.5, Euler, Strength 0.5

**Prompt 3 (Casual style):**
```
Girl, candid photo of young asian woman, natural makeup, casual clothing, outdoor setting, soft bokeh background
```
Settings: Steps 30, CFG 3.5, Euler, Strength 0.6

## Keywords

- `Girl` (trigger word)
- `Women` (trigger word)
- `asian woman`, `asian model`
- `portrait`, `detailed face`
- `natural lighting`, `soft lighting`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 25-35 |
| **CFG** | 3-4 |
| **Sampler** | Euler |
| **Size** | 1024x1024 or higher |
| **Strength** | 0.4-0.7 |

## Combinations

Works well with:
- **Detail Enhancer** (0.7) - For maximum facial detail
- **Flux Realism LoRA** (0.5) - For photorealistic results
- **Film/Photography LoRAs** - For specific aesthetic styles

## Notes

- Use lower strength (0.4-0.5) for subtle Asian features
- Use higher strength (0.6-0.7) for stronger effect
- Trigger words `Girl` or `Women` recommended but may work without
- v1.1 trained on 1024px images for better detail than v1.0
- Good base for combining with style LoRAs


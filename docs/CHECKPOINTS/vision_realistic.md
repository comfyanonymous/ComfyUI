# Vision Realistic V2 - Flux Dev FP8

[← Back to CHECKPOINTS Index](INDEX.md)

## Info
- **File:** `Vision_Realistic_V2_Flux_Dev_FP8.safetensors`
- **Original filename:** `visionRealistic_v2FluxDevFp8.safetensors`
- **Civitai:** https://civitai.com/models/619656/vision-realistic
- **Trigger:** None
- **Type:** CHECKPOINT (BASE MODEL)

## Description
Fine-tuned Flux Dev FP8 model optimized for photorealism. Addresses common Flux issues like occasional blurry images and incorrect skin tones. Features improved NSFW handling, brighter images, and fewer blur issues. Has CLIP and VAE baked in - no separate versions needed.

Created by training LoRA models and merging them with Flux dev fp8, with additional optimizations.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| VAE | Baked in (not required) |
| CLIP | Baked in (not required) |
| Sampler | Euler |
| Scheduler | Simple |
| Steps | 20 |
| CFG | 1 |

## Installation
Place in: `ComfyUI\models\unet\`

Load using "Load Diffusion Model" node in ComfyUI.

## Key Features
- CLIP and VAE baked in
- Better photorealism than base Flux in some cases
- Improved skin tones
- Brighter images
- Fewer blur issues
- Better NSFW content handling

## vs Original Flux
Not necessarily "better" than original Flux, but performs better for:
- Photorealism
- Consistent image quality (fewer random blurs)
- Accurate skin tones
- NSFW generation

## Best For
- Photorealistic portraits
- Realistic skin textures
- NSFW content
- When base Flux produces blurry results

## Notes
- Tested primarily on ComfyUI
- No trigger words needed
- Fine-tuned on 100k+ training steps (SD3M base version)
- Training data: 5k stock photos + custom dataset
- For FLUX version: Use simple workflow with Euler sampler

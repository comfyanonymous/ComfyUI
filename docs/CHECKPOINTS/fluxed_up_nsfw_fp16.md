# Fluxed Up NSFW v7.1 FP16

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

Full FP16 NSFW checkpoint - the primary recommended version of Fluxed Up. Heavily biased toward nude/NSFW female imagery with excellent quality and stability.

## File Information

- **Filename:** fluxedUpFluxNSFW_71FP16.safetensors
- **Location:** models\unet\
- **Format:** SafeTensors FP16
- **Version:** 7.1
- **Size:** ~22.2 GB
- **Base Model:** Flux.1 Dev

## Statistics

- **Downloads:** 82,582 (total model), 3,547 (v7.1)
- **Rating:** 3,508
- **Tips:** 8,022,894
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/847101/fluxed-up-flux-nsfw-checkpoint

## Model Type

NSFW Unet model in full FP16 format. Requires separate VAE and CLIP (not baked in).

## Description

Fluxed Up is a nude/NSFW capable checkpoint emphasizing female primary subjects. Version 7.1 is the latest iteration with improved stability and reduced artifacts compared to earlier versions. The FP16 format provides full quality without quantization loss.

Unlike the INT4/Nunchaku version (which has compatibility issues with standard UNETLoader), this FP16 version works perfectly with ComfyUI's built-in UNETLoader node.

## Key Features

- Full FP16 precision - no quality loss from quantization
- Strong NSFW capability out of the box
- Improved stability in v7.1
- Reduced artifacts vs earlier versions
- Works with standard UNETLoader (no special nodes needed)

## Recommended Settings

- **Sampler:** DPM++ 2M
- **Scheduler:** Beta
- **Steps:** 20-30
- **CFG:** 1.0 (FLUX standard)
- **VAE:** Required (not baked in)
- **CLIP:** Required (not baked in)

## Performance

- **VRAM:** ~22 GB (FP16 full precision)
- **Quality:** Excellent
- **Consistency:** High
- **LoRA Compatibility:** Good

## Use Cases

- High-quality NSFW generation
- Female-focused imagery
- Photorealistic nude photography
- Works well with quality enhancement LoRAs

## Sample prompts

**Prompt 1 (K-beauty coffee shop):**
```
A professional realistic photo of a woman The woman has K-beauty straight brows, Rounded (diamond:0.25) face, (puffy cheek:0.3), happy, auburn Illusion Ponytail hair, red eyes, overlined lips with subtle outline, portrait shot, upper body. The woman is happy and smiling. A (solo:1.4) woman slouching, wrapping her arms around her legs, surprised, legs together, pussy peek, holding one leg. Viewed from above, warm coffee shop window seat
```
Settings: Steps: 30, Sampler: DPM++ 2M, Seed: 705103279066972

**Prompt 2 (Ahegao cave selfie):**
```
A professional realistic photo of a woman The woman has 90s ultra-thin brows, Flat oval face, low cheekbone definition, broad forehead, soft jaw, wide nose bridge, (puffy cheek:0.3), happy, red Airy Layers hair, Green eyes, Commissure Lips, portrait shot, upper body. The woman is happy and smiling. amateur photo of a woman making ahegao face with her tongue sticking out and her eyes crossed solo taking a selfie holding a camera leaning against a wall, long hair, pink blouse, bare shoulders, very long hair, standing, pink hair, denim shorts, pointy ears, short shorts, drawn on brown freckles, wooden floor, fishnet pantyhose, cutoffs, wearing leather bdsm straps underneath her blouse, , underground cave in an adventure game, with glowing crystals scattered on the ground
```
Settings: Steps: 30, Sampler: DPM++ 2M, Seed: 984547453268131

**Prompt 3 (Emily Feld doggy - palm trees sunset):**
```
absolutely gorgeous 23-year-old Emily_Feld, brunette, long dark brown hair, detailed brown eyes, Her breasts are small with erect pink nipples, natural blended edge areolas, eye liner on the upper eyelid, natural color makeup, thin and fit, 5.9 ft tall, she weighs 125lb, gorgeous narrow face, wide hips, in the doggy position, back arched downward, hair and skin wet from sweating, completely naked, dark sun tanned skin with blended tan lines, showing her pussy, 4k hd image, sharper image, 4k quality images, no makeup, very small tattoo of a heart on her left hip, spreading her legs showing her perfect pussy, perfect eyes, balanced well shaped eyes, detailed eyes, full color tones, deep colors, natural colors, in the shade under a group of palm trees at sunset, hd image, flawless face, masterpiece, perfect face, ((NO BIG BREASTS:1.1))
```
Settings: Steps: 30, Sampler: Euler, Seed: 618270597046206, Size: 1024x1024

## Comparison with INT4 Version

| Aspect | FP16 (this) | INT4 (Nunchaku) |
|--------|-------------|-----------------|
| Size | 22.2 GB | 6.3 GB |
| Quality | Excellent | Fair |
| Speed | Standard | Very fast |
| VRAM | ~22 GB | ~6 GB |
| UNETLoader | Works | Does NOT work |
| Anatomy | Good | Issues with legs |

## Version History

- **V7.1 FP16:** Current version (Jan 2026) - improved stability
- **V5.1 INT4:** Nunchaku version - fast but compatibility issues

## Notes

- Replaces the problematic INT4 version as the recommended Fluxed Up checkpoint
- No VAE or CLIP baked in - must load separately
- Sample images on Civitai include embedded workflows
- Available in multiple formats: FP16, Q8_GGUF, Q4_GGUF

---

**Category:** NSFW Specialized, Photorealistic
**Last Updated:** 2026-02-11

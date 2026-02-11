# getphat FLUX Reality NSFW

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

High quality FP8 NSFW checkpoint (11.08 GB) with extensive trained sexual concepts including full hardcore content, danbooru tag support, and natural language prompting. V10 is the full (non-softcore) version with all trained concepts enabled.

## File Information

- **Filename:** getphatFLUXReality_v10FP8.safetensors
- **Location:** models\unet\
- **Format:** SafeTensors FP8
- **Size:** 11.08 GB
- **Version:** V10 (Full)
- **Base Model:** Flux.1 Dev

## Statistics

- **Downloads:** 65,025 (total all versions)
- **Rating:** 4,031
- **Tips:** 5,165
- **Comments:** 152
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/861840/getphat-flux-reality-nsfw

## Model Type

NSFW-specialized FP8 Unet model with extensive sexual concept training (full/hardcore). Supports both danbooru tags and natural language prompting.

## Description

getphat FLUX Reality NSFW V10 is the full (hardcore) version with all trained sexual concepts enabled including positions, oral, cum effects, and more. The model uses danbooru tags (with 'woman' replacing '1girl') but also supports natural language and multi-language prompting (French, Italian tested).

At 11.08 GB this is an FP8 quantized model optimized for 12 GB VRAM GPUs. FP16 version (23.2 GB) also available for higher VRAM systems.

**Note:** Replaced V11 Softcore FP16 (22.17 GB) - V10 FP8 provides full content in VRAM-friendly format. New updates posted to Patreon first.

## Key Features

- FP8 format (11.08 GB) - fits 12 GB VRAM
- Full hardcore content (all trained concepts enabled)
- Extensive trained sexual concepts with specific danbooru tags
- Dual prompting: danbooru tags + natural language
- Multi-language support (English, French, Italian confirmed)
- Photography style control ("realistic" for DSLR, "amateur photo" for casual)

## LoRA Compatibility

**⚠️ CRITICAL WARNING:** LoRAs trained on Flux Dev are NOT compatible - they will burn/destroy the image!

**Compatible LoRAs:**
- LoRAs trained specifically on Flux Reality checkpoint
- LoRAs trained on other Flux Dreambooth fine-tune checkpoints
- Previous version LoRAs work but retraining recommended
- Softcore LoRAs work better with softcore versions

**Patreon LoRAs:**
- Amateur Photography LoRA (enhances amateur photo style)

## Trained Concepts (with danbooru tags)

### Body Features
- **Big ass** - ass, big_ass
- **Innie pussy** - pussy, innie
- **Pubic hair** - pubic_hair, landing_strip, excessive_pubic_hair (female and male)

### Clothing/Lingerie
- **G-strings** - g-string, thong

### Sexual Positions (POV)
- **POV Cowgirl** - vaginal_sex, cowgirl_position, breasts, spread_legs, solo_focus
- **POV Reverse Cowgirl** - vaginal_sex, reverse_cowgirl_position, ass, pussy, anus
- **Missionary** - vaginal_sex, missionary, male_pov, spread_legs, pussy
- **Doggystyle** - vaginal_sex, doggystyle, male_pov, pussy, anus, penis

### Oral
- **POV Blowjob/Paizuri** - paizuri, breasts_squeezed_together, penis, solo_focus
- **Blowjob from side** - fellatio, deepthroat, penis, tongue_out
- **Ahegao face** - ahegao, tongue_out, rolling_eyes

### Cum Effects
- **Facial** - open_mouth, tongue, cum_on_face, solo_focus
- **Cum placement** - cum_on_body, cum_on_breasts, cum_in_mouth (various options)

### Style
- **Anime** - anime style rendering option

## Prompt Style Support

### Danbooru Tags (primary)
```
woman, big_breasts, g-string, pov, cowgirl_position, bedroom, realistic
```

### Natural Language
```
Beautiful woman with large breasts wearing a g-string,
POV angle, cowgirl position, bedroom setting, realistic
```

### Photography Style Control
```
realistic          → DSLR photograph quality
amateur photo      → casual amateur photography look
```

Both prompting styles work effectively. Model uses danbooru tags internally (with 'woman' instead of '1girl').

## Recommended Settings

| Setting | Value |
|---------|-------|
| **Sampler** | dpmpp_2m |
| **Scheduler** | sgm_uniform |
| **Steps** | 35 |
| **CFG** | 4 |
| **VAE** | Required (not baked in) |
| **CLIP** | Required (not baked in) |

## Version History

| Version | Type | Size | Downloads | Rating | Date |
|---------|------|-----:|----------:|-------:|------|
| **V11 Softcore** | FP16 | 23.2 GB | 7,797 | 668 | 2025-07 |
| V11 Softcore FP8 | FP8 | 11.6 GB | 2,601 | 95 | 2025-07 |
| V10 | FP16 | 23.2 GB | 2,883 | 301 | 2025-07 |
| V10 FP8 | FP8 | 11.6 GB | 2,576 | 65 | 2025-07 |
| V9 Softcore | FP16 | 23.2 GB | 1,069 | 233 | 2025-06 |
| V9 Softcore FP8 | FP8 | 11.6 GB | 777 | 50 | 2025-07 |
| V8 | FP16 | 23.2 GB | 2,273 | 295 | 2025-06 |
| V8 FP8 | FP8 | 11.6 GB | 1,683 | 58 | 2025-06 |
| V7 | FP16 | 23.2 GB | 5,935 | 634 | 2025-05 |
| V7 FP8 | FP8 | 11.6 GB | 2,725 | 114 | 2025-05 |
| V6 | FP16 | 23.2 GB | 6,078 | 700 | 2025-04 |

**Version notes:**
- Odd versions (V9, V11) = Softcore only
- Even versions (V8, V10) = Full version
- V11 = Major breast/nipple rendering upgrade
- V7 = Most downloaded single version (5.9K)

## VRAM Requirements

| Format | VRAM | File Size | Installed |
|--------|:----:|:---------:|:---------:|
| **FP8** | 12 GB | 11.08 GB | ✅ V10 |
| FP16 | 20-24 GB | 23.2 GB | - |

## Use Cases

- POV sexual position generation
- Softcore adult photography
- Breast-focused content (V11 upgrade)
- Lingerie and clothing
- Multi-language prompting
- Amateur photography style

## Comparison with Alternatives

| Aspect | getphat Reality | Fluxed Up v7.1 | Fux Capacity 5.1 |
|--------|:--------------:|:--------------:|:-----------------:|
| Size | 11.08 GB (FP8) | 22.2 GB (FP16) | ~22 GB (FP16) |
| Downloads | 65K | 82K | 48K |
| NSFW Level | Full (V10) | Full | Full |
| Trained Concepts | Extensive | General | General |
| LoRA Compat | ⚠️ Limited | Standard | Standard |
| Steps | 35 | 20-30 | 32 |
| Best For | POV, positions | Female NSFW | Film grain, ethnicity |

## Strengths

- 65K+ downloads with 4K+ ratings - strong community validation
- Full hardcore content in V10 (all concepts enabled)
- Extensive trained sexual concepts with specific danbooru tags
- Dual prompting (tags + natural language + multi-language)
- FP8 format fits 12 GB VRAM perfectly
- Photography style control (DSLR vs amateur)
- Active development with regular version updates

## Limitations

- **LoRA compatibility:** Flux Dev LoRAs WILL BURN images - only use Reality-trained LoRAs
- **35 steps** required - higher than most FLUX models
- **sgm_uniform** scheduler (less common than Beta/Euler)
- Patreon-first updates

## Notes

- Total 65K+ downloads across all versions indicates strong community
- V11 specifically improved breast variety and nipple rendering
- Photography style prompts ("realistic", "amateur photo") control output aesthetic
- Multi-language support is unusual for FLUX models
- LoRA warning is critical - standard Flux Dev LoRAs are incompatible
- FP8 version available for lower VRAM (11.6 GB)
- Patreon has exclusive LoRAs and early access to updates

---

**Category:** NSFW Specialized
**Last Updated:** 2026-02-11

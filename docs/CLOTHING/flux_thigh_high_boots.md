# FLUX Thigh High Boots

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 196 |
| **👍** | 16 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_Thigh_High_Boots.safetensors` |
| **Civitai** | https://civitai.com/models/1216257/flux-thigh-high-boots |
| **Trigger word** | `Thigh-High Boots` |
| **Strength** | 1.0-1.1 |
| **Type** | CLOTHING / Footwear |

## Description

LoRA for generating thigh-high boots in FLUX. Created because base FLUX is bad at generating thigh-high boots. Works with any color or pattern of boots.

### Key Features
- Thigh-high boots generation
- Any color/pattern (black, white, latex, leather, etc.)
- Platform heels option
- Works with various outfits
- Fixes FLUX's weakness with this clothing item

## Sample Prompts

**Prompt 1 (Basic - black boots):**
```
a young brunette woman, with curly hair is standin outside of a crab restaurant at night, she is wearing black Thigh-High Boots, white leather pants and a black angora sweater
```
Settings: Steps 4, CFG 1, Euler, 544x968

**Prompt 2 (White boots):**
```
a young brunette woman, with curly hair is standin outside of a restaurant at night, she is wearing white Thigh-High Boots, black leather pants and a black angora sweater
```
Settings: Steps 4, CFG 1, Euler, 776x968

**Prompt 3 (Pakistani nightclub - NSFW detailed):**
```
highly detailed head-to-foot wide-angle photo of a brown-skinned pakistani woman standing, wearing a tight-fitting see-through lycra micro-minidress and (Thigh-High high-heled black latex platform Boots:1.1) <lora:FLUX_Thigh_High_Boots:1.1>. She has a petite diminutive body with slim waist and wide hips. She has thick lips and almond-shaped eyes. posing in the street, at night, outside a night club in london. The scene is lit up by a camera flash. her skin is sweaty and glistening and her makeup is smeared and her hair is greasy and clumped and dishevelled after a night of partying. her perky breasts, dark nipples, navel and darkened vulva are clearly visible under the see-through transparent fabric of her dress. she has a wide drunken smile and her eyes are droopy in a drunken stupor. A natural and unposed amateur candid photo capturing a genuine moment, framed with slight imperfection for authenticity. <lora:Flux_Skin_Detailer:0.7>, <lora:MeltingPot05:0.5>, indian woman
```
Settings: Steps 45, CFG 1, DPM++ 2M, 768x1024, Dist.CFG 2, Beta scheduler (0.6/0.6)
Resources: Flux Skin Detailer 0.7, MeltingPot 0.5, Thigh High Boots 1.1

## Keywords

- `Thigh-High Boots` - **TRIGGER WORD** (capitalize hyphenated)
- `black Thigh-High Boots` - color variant
- `white Thigh-High Boots` - color variant
- `black latex platform Boots` - material + style
- `high-heeled` - heel type
- `platform Boots` - platform style

## Boot Variations

| Variation | Prompt Addition |
|-----------|-----------------|
| **Black** | `black Thigh-High Boots` |
| **White** | `white Thigh-High Boots` |
| **Latex** | `black latex Thigh-High Boots` |
| **Platform** | `Thigh-High platform Boots` |
| **High-heeled** | `Thigh-High high-heeled Boots` |
| **Combined** | `(Thigh-High high-heeled black latex platform Boots:1.1)` |

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 4-45 |
| **CFG** | 1 |
| **Distilled CFG** | 2 |
| **Sampler** | Euler / DPM++ 2M |
| **Size** | 544x968 / 768x1024 |
| **Strength** | 1.0-1.1 |

## Recommended Combinations

| LoRA | Strength | Purpose |
|------|----------|---------|
| Flux Skin Detailer | 0.7 | Realistic skin |
| MeltingPot | 0.5 | Ethnicity enhancement |
| Character LoRAs | 0.8-1.0 | Specific characters |

## Notes

- Trigger: `Thigh-High Boots` (capitalize, use hyphen)
- Created to fix FLUX's weakness with thigh-high boots
- Works with any color or pattern
- Can specify material (latex, leather, etc.)
- Can add heel style (platform, high-heeled)
- Use weight emphasis `(Thigh-High Boots:1.1)` for stronger effect
- Works well with nightclub/street photography scenes
- Portrait orientation recommended for full-body shots


# Asian Beauty Standard (Type C and C+)

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 588 |
| **👍** | 54 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Asian_Beauty_Standard_Type_C.safetensors` |
| **Original filename** | `beauty_standard4-000080.safetensors` |
| **Civitai** | https://civitai.com/models/785098 |
| **Trigger word** | None |
| **Strength** | 0.9-1.0 |
| **Type** | STYLE / Ethnicity |

## Description

More advanced version of Asian Beauty Standard series. Makes subjects more Asian AND more detailed. Described as "WILD" by the author - stronger effect than Type B.

- Training Set: 33 SD-generated images (NSFW)
- Training Steps: 2640 (C) / 3960 (C+)
- Two versions available: C (80 epochs) and C+ (120 epochs)

## Sample Prompts

**Prompt 1 (Portrait - black shirt):**
```
head shot portrait photo of a beautiful 20yo woman, lacey black shirt, smiling
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 1.0

**Prompt 2 (School uniform amusement park):**
```
a beautiful 30yo woman in lacey shirt, school uniform jacket, plaid skirt, in an amusement park, smiling, v hand sign
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 1.0

**Prompt 3 (Lara Croft cosplay):**
```
photo of a beautiful Lara Croft woman, long dark hair, loose ponytail, sweaty, earrings, necklaces, dirty aqua blue crop top, short denim shorts, boots, backpack, holsters on thigh straps, pierced navel, no tattoos, walking carefully across a rickety rope bridge which bridges a chasm between two cliffs with a raging river below, in a humid, steamy tropical jungle, face in sharp focus, detailed face, detailed eyes, perfect hands, High Detail, Perfect Composition, dramatic dim lighting, high contrast, viewed from the side
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.9

**Prompt 4 (School uniform NSFW):**
```
a beautiful woman in school uniform jacket, skirt, thighhighs, showing off her ass from behind, slightly tilting forward, sticking up her ass, huge breasts, gigantic thighs, grin
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 1.0

**Prompt 5 (Redhead vintage - with NSFW Master):**
```
head shot portrait of a young woman with flowing red hair and green eyes, wearing a vintage dress, set in a field of wildflowers at sunset.
```
Combined with: pytorch_lora_weights 0.6, NSFW_master 0.5
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.6

**Prompt 6 (Multiple women NSFW - with Korean Girl):**
```
This image is an illustration of four 21yo woman fully naked and is wearing no clothes crouching in a studio room.
First Girl (Far Left): She has silver hair and her hair tied up in a high ponytail with long bangs and is wearing no clothes and is fully naked. She is grabbing the second girls chest.
Second Girl (Second from Left): She has long, dark hair with bangs and is wearing no clothes and is fully naked.
Third Girl (Second from Right): She has long, wavy brown hair and is wearing no clothes and is fully naked.
Fourth Girl (Far Right): She has blonde hair and her hair in a high ponytail and is wearing no clothes and is fully naked.
Long eyelashes, ultra detailed, detailed beautiful eyes and detailed face, delicate facial features, mixed American Korean actress, natural and detailed beauty doll face, greatest glamorous body type, blood vessels, skin pores, blood vessels in sclera, detailed skin, beauty spots, film grain, skin fuzz, beautiful detailed eyes, cat eyes,wing eyeliner, subtle eye makeup,glossy red lipstick
```
Combined with: pytorch_Beautiful Korean Girl 0.6, NSFW_master 0.5
Settings: Steps 40, CFG 1, LMS, 1080x576, Distilled CFG 5, Strength 0.6, Upscale 2x (4xNMKD-Superscale)

**Prompt 7 (Cyberpunk duo - with Korean Girl):**
```
This is an image two 21yo women dressed in pastel, holographic outfits with a futuristic and sparkly aesthetic. They find themselves in a dystopian neon cyberpunk cityscape where neon lights bounce off wet streets like shards of broken dreams...
[Full detailed prompt with clothing descriptions]
Long eyelashes, ultra detailed, detailed beautiful eyes and detailed face, delicate facial features, mixed American Korean actress, natural and detailed beauty doll face, greatest glamorous body type, blood vessels, skin pores, blood vessels in sclera, detailed skin, beauty spots, film grain, skin fuzz, beautiful detailed eyes, cat eyes,wing eyeliner, subtle eye makeup,glossy red lipstick
```
Combined with: pytorch_Beautiful Korean Girl 0.6, NSFW_master 0.5
Settings: Steps 30, CFG 1, LMS, 512x768, Distilled CFG 5, Strength 0.6, Upscale 2x

## Keywords

- No trigger word required
- `asian`, `korean`, `chinese`, `japanese`
- `beautiful woman`, `20yo`, `30yo`
- `detailed face`, `detailed eyes`
- `ultra detailed`, `skin pores`, `film grain`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-40 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5-5 |
| **Sampler** | Euler / LMS |
| **Size** | 896x1152 / 1080x576 |
| **Strength** | 0.6-1.0 |

## Comparison: Type B vs Type C

| Feature | Type B | Type C |
|---------|--------|--------|
| Training Images | 11 | 33 |
| Training Steps | 1,100 | 2,640-3,960 |
| Effect Strength | Subtle | Strong ("WILD") |
| Detail Level | Standard | Enhanced |
| Recommended Strength | 0.7-0.85 | 0.9-1.0 |
| Best for | Subtle Asian features | Strong transformation |

## Combinations

Works well with:
- **Beautiful Korean Girl** (0.6) - Enhanced Korean features
- **NSFW Master** (0.5) - NSFW unlock
- **Detail enhancer LoRAs** - For maximum detail

## Notes

- Stronger effect than Type B - use when you want more pronounced Asian features
- Try switching between 0.9 and 1.0 strength if you get bad anatomy
- Change seed if anatomy issues persist
- Lower strength (0.6) when combining with other Asian LoRAs
- Works great for multiple character scenes
- Upscale with 4xNMKD-Superscale or 4xFFHQDAT for best results


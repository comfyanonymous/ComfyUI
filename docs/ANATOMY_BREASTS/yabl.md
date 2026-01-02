# YABL Boob Diffusion

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 518 |
| **👍** | 27 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `boobDiffusion.safetensors` |
| **Civitai** | https://civitai.com/models/1711906/yabl-yet-another-boob-lora-fluxd |
| **Trigger word** | `b00bs` |
| **Strength** | 0.8-1.0 |
| **Type** | Anatomy / Breasts |
| **Training** | ~200 images above 1MP |

## Description

Yet Another Boob Lora (YABL) for FLUX Dev. High-quality breast enhancement LoRA trained on approximately 200 high-resolution images (all above 1MP). Uses l33t trigger word for NSFW bypass.

## Sample prompts

**Prompt 1 (Scandinavian portrait):**
```
b00bs, a portrait of a beautiful scandinavian woman with big breast and slim curvy body, she is naked and doing a sexy pose for the camera <lora:boobDiffusion:1>
```
Settings: Steps: 30, CFG: 1, Sampler: DPM++ 2M, Size: 1024x1360

**Prompt 2 (With cum - combined with YACL):**
```
beautiful woman with short frizzy black hair and green eyes, pouty shy face, leaning forward showing off her big b00bs, she has a minimalistic tattoo that says "I love cum" on her chest, she has lots of cum on her face, cumonface <lora:boobDiffusion:1> <lora:cumonface:1.2>
```
Settings: Steps: 30, CFG: 1, Sampler: DPM++ 2M, Size: 1024x1360

**Prompt 3 (Simple):**
```
beautiful woman with big b00bs, naked, sexy pose <lora:boobDiffusion:1>
```

## Keywords

- `b00bs` - **TRIGGER WORD** (l33t)
- `big breast`
- `slim curvy body`
- `naked`
- `sexy pose`
- `leaning forward`
- `showing off`

## Tested combinations

- YACL Cum on Face (cumonface)
- Flux Dev fp8

## Recommended settings

- **Steps:** 30
- **CFG:** 1
- **Sampler:** DPM++ 2M
- **Size:** 1024x1360

## Notes

- Uses l33t trigger `b00bs` (zeros instead of o's)
- High-quality training data (all above 1MP)
- Works well combined with YACL (cumonface) LoRA
- Simple and effective breast enhancement

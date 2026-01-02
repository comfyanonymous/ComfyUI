# Corset L Lingerie FLUX

[← Back to CLOTHING Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `corsetlling-flux-000012.safetensors` |
| **Civitai** | https://civitai.com/models/1096641/corset-l-lingerie-flux |
| **Trigger word** | None (use clothing descriptions) |
| **Strength** | 1.0-1.2 |
| **Type** | Clothing / Lingerie |

## Description

LoRA for generating corset and lingerie styles. Creates elegant corset outfits with stockings, garter belts, and other lingerie pieces. Works well with various poses and settings.

## Key features

- Corset lingerie generation
- Works with various colors (purple, red, green, etc.)
- Compatible with stockings and high heels
- Full body portrait poses
- Boudoir/intimate settings

## Recommended settings

- **Strength:** 1.0-1.2
- **Distilled CFG Scale:** 3.5
- **Steps:** 20-50
- **Sampler:** Euler
- **Scheduler:** Simple
- **Clip skip:** 1-2
- **Size:** 832x1216

## Sample prompts

**Prompt 1 (French brunette purple corset):**
```
A high-resolution photograph of a 20-year-old French Brunette woman. She has a warm, friendly smile and is looking at viewer with her beautiful Green eyes. She has a light skin tone, Brunette hair, and a slender, ((petite physique with hourglass curves)). This is a ((detailed full body portrait photo)) of a young woman. She is posing in the white room. Her arms above her head grab the wall, her bottom body pushed forward close to the camera, low angle, Detailed face, full body portrait, purple corset, high heel sandals
```
Settings: Steps: 50, CFG scale: 3.5, Size: 832x1216

**Prompt 2 (Simple snow mountain):**
```
Blonde woman wearing red bra and miniskirt in snowy mountain
```
Settings: Steps: 20, CFG scale: 3.5, Size: 832x1216

**Prompt 3 (Green ribbon wrap New Year):**
```
A high-resolution photograph of a 20-year-old French Brunette woman in green ribbon wrap celebrating the New Year 2025. She has a warm, friendly smile and is looking at viewer with her beautiful Green eyes. She has a light skin tone, Brunette hair, and a slender, ((petite physique with hourglass curves)). This is a ((detailed full body portrait photo)) of a young woman. She is posing in the white room. Her arms above her head grab the wall, her bottom body pushed forward close to the camera, low angle, Detailed face, full body portrait, purple corset, high heel sandals
```
Settings: Steps: 50, CFG scale: 3.5, Size: 832x1216

**Prompt 4 (Purple boudoir with hires):**
```
<lora:corsetlling-flux:0.8>, purple corset, stockings, cll, A close-up, high-resolution, professional photograph of a young woman with the appearance of an artist. She has long, wavy purple hair, a heart-shaped face, and a slightly larger chest that accentuates her outfit. She is wearing a tight purple corset and stockings, posing elegantly with one hand on her hip and the other resting lightly on her shoulder. The background is a dimly lit, moody boudoir setting with dark purple and gold accents. The lighting highlights her features, creating an artistic interplay of shadows and soft glow.
```
Settings: Steps: 20, CFG scale: 1, Sampler: Euler, Schedule: Simple, Distilled CFG Scale: 3.5, Hires upscale: 2x with 4x-UltraSharp, Denoising: 0.3

## Keywords

- `corset`
- `purple corset`
- `red corset`
- `lingerie`
- `stockings`
- `high heel sandals`
- `cll` (optional trigger)
- `boudoir`
- `tight corset`
- `hourglass curves`
- `low angle`
- `full body portrait`

## Notes

- Strength 0.5-0.8 when combining with other LoRAs
- Strength 1.0-1.2 for standalone use
- Works great with character LoRAs
- Low angle poses work particularly well
- Purple and red corsets most common
- Hires upscale 2x recommended for detail
- Use `cll` as optional trigger for corset style

## Quality Stats
- **Downloads:** 618
- **Rating:** 70
- **Tips:** $0

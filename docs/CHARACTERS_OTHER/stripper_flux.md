# Stripper Flux

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 566 |
| **👍** | 54 |
| **Tips** | 50 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Stripper_Flux.safetensors` |
| **Original filename** | `Stripper_flux-000010.safetensors` |
| **Civitai** | https://civitai.com/models/864939/stripper-flux |
| **Trigger word** | None |
| **Strength** | 0.7 |
| **Type** | CHARACTER / Occupation Style |

## Description

Generates strippers in FLUX. Creates women in strip club settings with stripper poles, stages, and club atmosphere. Works well with character LoRAs to turn any character into a stripper scene.

**Use cases:**
- Strip club scenes
- Pole dancing poses
- Stage performances
- Crowded club atmosphere

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 25 |
| **CFG** | 3.5 |
| **Sampler** | Undefined |
| **Size** | 832x1216 |
| **Strength** | 0.7 |

## Sample Prompts

**Basic stripper scene:**
```
front view, realistic photo of a woman, very large breasts, standing, looking at viewer, ultra high quality, very detailed skin, sharp focus, highly detailed, small bikini, nip slip, long hair working as a stripper woman, dancing in a strip club on stage, stripper pole, crowded, <lora:Stripper_Flux:0.7>
```
Settings: Steps 25, CFG 3.5, 832x1216

**With character LoRA:**
```
front view, realistic photo of a woman, large breasts, standing, looking at viewer, ultra high quality, very detailed skin, sharp focus, highly detailed, <lora:CHARACTER_LORA:1>, working as a stripper woman, dancing in a strip club on stage, stripper pole, crowded, <lora:Stripper_Flux:0.7>
```
Settings: Steps 25, CFG 3.5, 832x1216

## Keywords

- `stripper woman`
- `dancing in a strip club`
- `on stage`
- `stripper pole`
- `crowded`
- `working as a stripper`
- `nip slip`
- `small bikini`

## Scene Elements

| Element | Prompt Addition |
|---------|-----------------|
| **Pole dancing** | `stripper pole`, `pole dancing` |
| **Stage** | `on stage`, `spotlight` |
| **Crowd** | `crowded`, `audience` |
| **Outfit** | `small bikini`, `lingerie`, `nip slip` |

## Best Checkpoints

- FLUX Dev
- FLUX Checkpoint Dev

## Recommended Combinations

**With character LoRA:**
```
<lora:CHARACTER_NAME:1>
<lora:Stripper_Flux:0.7>
```

**With MILF character:**
```
<lora:EBMilf_V3:1>
<lora:Stripper_Flux:0.7>
```

## Notes

- No trigger word required
- Lower strength (0.7) recommended to blend with other LoRAs
- Works well with various character LoRAs
- Adds strip club environment and atmosphere
- Compatible with outfit/clothing descriptions
- Creates authentic stage/pole dancing scenes
- Good for crowded club atmosphere

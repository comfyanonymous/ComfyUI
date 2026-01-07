# High Detail Eyes

[← Back to STYLE_ENHANCEMENT Index](INDEX.md)

## Info
- **File:** `Flux_High_Detail_Eyes.safetensors`
- **Original filename:** `Flux_high_detail_eyes.safetensors`
- **Civitai:** https://civitai.com/models/1490479/high-detail-eyes
- **Trigger:** None (Flux version)
- **Strength:** 0.6 (recommended)
- **Type:** CONCEPT / Enhancement

## Description
LoRA for generating detailed, beautiful eyes on women. Works best with green, blue, and brown eye colors. The Flux version requires no trigger word - just add to any image containing a face.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Strength | 0.6 (can go lower with other LoRAs) |

## Prompt Tips
Specify eye color in prompt for best results:
- `green eyes`
- `blue eyes`
- `brown eyes`

## Example Usage
Simply add the LoRA to any portrait generation:
```
<lora:Flux_High_Detail_Eyes:0.6>
```

No trigger word needed for Flux version.

## Best Eye Colors
1. Green eyes
2. Blue eyes
3. Brown eyes

## Keywords
- `green eyes`
- `blue eyes`
- `brown eyes`
- `detailed eyes`
- `pretty eyes`

## Best Checkpoints
- FLUX Dev
- Any FLUX-based checkpoint

## Notes
- **No trigger word required** for Flux version (Pony version uses "High detail eyes")
- Recommended weight: 0.6
- Can use lower weights (0.3-0.5) when combining with other LoRAs
- **Warning:** May slightly change face features, especially with character LoRAs (like Lara Croft, Agent Alona)
- For character LoRAs, consider using lower strength (0.3-0.4) to minimize face changes
- Works on any image containing a face

# Flux Realistic Hands

[Back to HANDS Index](INDEX.md)

## Overview

Hand realism LoRA with trigger word for FLUX models. Improves hand realism and works well when combined with other LoRAs.

## Basic Information

- **File:** `Flux_Realistic_Hands.safetensors`
- **Original Filename:** `fitzka.safetensors`
- **Type:** Enhancement / Hand Fix
- **Civitai:** https://civitai.com/models/1232423/flux-realistic-hands
- **Trigger Word:** `fitzka` (REQUIRED)
- **Recommended Strength:** 0.5-0.7

## Compatibility

- FLUX.1-dev
- FLUX.1-schnell

## Description

A FLUX-specific LoRA that enhances hand realism using a trigger word. This LoRA requires the trigger word "fitzka" to activate and works particularly well for realistic hand poses and interactions. It's designed to be compatible with other LoRAs for enhanced results.

## Keywords

- fitzka (REQUIRED trigger)
- hand reach
- punching viewer
- holding

## Settings

**Recommended Strength:** 0.5-0.7

- **0.5:** Subtle realism enhancement
- **0.6:** Balanced improvement
- **0.7:** Strong realistic effect

Higher strengths may cause over-processing. Start low and adjust upward.

## Sample Prompts

```
fitzka, hand reach, extending towards camera
```

```
fitzka, punching viewer, dynamic action pose
```

```
fitzka, holding object, realistic grip
```

```
fitzka, detailed hands, natural skin texture
```

## Tips

- **Always include the trigger word "fitzka"** in your prompt
- Works well at lower strengths (0.5-0.7)
- Can be combined with other FLUX LoRAs for enhanced results
- Use action keywords like "hand reach" or "holding" for specific poses
- "punching viewer" creates dramatic forward-facing hand poses

## Combination Tips

- Pairs well with character LoRAs
- Compatible with style LoRAs
- Can be used alongside other hand enhancement LoRAs at reduced strengths

## Notes

- Trigger word is mandatory for this LoRA to work
- FLUX-specific, not compatible with other model types
- Focus on realistic hand rendering rather than anatomy correction

---

[Back to HANDS Index](INDEX.md)

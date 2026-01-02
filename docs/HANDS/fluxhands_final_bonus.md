# FluxHands Final Bonus

[Back to HANDS Index](INDEX.md)

## Overview

FLUX-specific hand fix LoRA that helps with hand anatomy, finger count, and hand positioning. Designed specifically for FLUX models.

## Basic Information

- **File:** `FluxHands_Final_Bonus.safetensors`
- **Original Filename:** `lora-000016.TA_trained.safetensors`
- **Type:** Enhancement / Hand Fix
- **Civitai:** https://civitai.com/models/805324/fluxhands-final-bonus
- **Trigger Word:** None (use keywords)
- **Recommended Strength:** 0.5-1.0

## Compatibility

- FLUX.1-dev
- FLUX.1-schnell

## Description

A FLUX-optimized LoRA specifically trained to fix common hand generation issues. It improves hand anatomy, ensures correct finger count, and helps with proper hand positioning. Note that it may occasionally confuse left and right hands, so be specific in your prompts if hand orientation matters.

## Keywords

- female hand
- right hand
- left hand
- back side

## Settings

**Recommended Strength:** 0.5-1.0

- **0.5:** Subtle improvements, maintains base model style
- **0.7:** Balanced enhancement
- **1.0:** Maximum hand correction effect

## Sample Prompts

```
female hand, detailed fingers, natural pose
```

```
right hand, five fingers, palm facing viewer
```

```
left hand, back side, elegant gesture
```

```
both hands, realistic anatomy, proper proportions
```

## Tips

- Use specific hand orientation keywords (right hand, left hand) for better control
- May confuse left/right hands occasionally - be explicit in prompts
- Works well for both male and female hands, though "female hand" keyword is available
- Can be combined with other FLUX LoRAs
- Use "back side" keyword to show the back of the hand

## Known Limitations

- May occasionally swap left and right hand orientation
- Best results when hand orientation is specified in the prompt

## Notes

- FLUX-specific, not compatible with other model types
- No trigger word needed, uses keywords for control
- Part of the final bonus release with improved training

---

[Back to HANDS Index](INDEX.md)

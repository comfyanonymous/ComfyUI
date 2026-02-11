# Hitchhike Flux (Option Nude Girl)

[← Back to Index](../INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 553 |
| **👍** | 59 |
| **Tips** | 20 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `HitchhikeFlux.1.0.safetensors` |
| **Civitai** | https://civitai.com/models/1015785/hitchhike-flux-option-nude-girl |
| **Trigger word** | `Hithik` (required), `thuup` (optional, thumb up) |
| **Strength** | 1.0 |
| **Type** | CONCEPT (Hitchhiking Pose / Nudity) |
| **Version** | 1.0 |

## Description

Hitchhiking pose LoRA for FLUX with optional nudity. Creates realistic hitchhiking scenes with girls on roads or streets. Does not need an NSFW checkpoint to create NSFW images - works with standard flux1-dev.

## Keywords

### Required
- `Hithik` - Main trigger word (activates the LoRA)

### Optional Modifiers
- `thuup` - Thumb up gesture
- `hitchhiking` - Hitchhiking action
- `with thumb up` - Alternative thumb up
- `road` - Road in nature setting
- `street` - Street in a city setting
- `slim nude girl` - Full nudity
- `slim topless girl` - Topless only

## Prompt Structure

Build prompts using this template:
```
Hithik, <thuup>, <slim nude|slim topless> girl hitchhiking <with thumb up> at <street|road>
```

Then add details for clothing/accessories, hair, etc.

## Sample Prompts

**Prompt 1 (Nude blonde - road, stockings):**
```
hithik, thuup, front view,
(nude slim 25yo girl) hitchhiking with thumb up at a road in nature. The woman wears black holdup stockings, necklace
and has blonde ponytail
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Distilled CFG Scale: 3.5

**Prompt 2 (Nude brunette - road, stockings):**
```
hithik, thuup, front view,
(nude slim 25yo girl) hitchhiking with thumb up at a road in nature. The woman wears black holdup stockings, necklace
and has brunette ponytail
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Distilled CFG Scale: 3.5

**Prompt 3 (Topless brunette - road, denim shorts):**
```
hithik, thuup, front view,
(topless slim 25yo girl) hitchhiking with thumb up at a road in nature. The woman wears denim shorts, black holdup stockings, necklace
and has brunette ponytail
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Distilled CFG Scale: 3.5

**Prompt 4 (Nude brunette - road, thigh high boots):**
```
hithik, thuup, front view,
(nude slim 25yo girl) hitchhiking with thumb up at a road in nature. The woman wears black thigh high boots, necklace
and has short brunette hair
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Distilled CFG Scale: 3.5

**Prompt 5 (Nude brunette - road, stockings, short ponytail):**
```
hithik, thuup, front view,
(nude slim 25yo girl) hitchhiking with thumb up at a road in nature. The woman wears black holdup stockings, necklace
and has short brunette ponytail
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Distilled CFG Scale: 3.5

**Prompt 6 (Basic nude - road):**
```
hithik, thuup, slim nude girl hitchhiking with thumb up at a road in nature. The girl wears holdup stockings...
```

**Prompt 7 (Topless - city street):**
```
hithik, slim topless girl hitchhiking at a street in a city. The girl wears denim shorts and sneakers...
```

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20 |
| **CFG** | 1 |
| **Distilled CFG Scale** | 3.5 |
| **Sampler** | Euler |
| **Schedule type** | Simple |
| **Size** | 896x1152 |
| **Strength** | 1.0 |
| **Model** | flux1-dev-fp8 (works without NSFW checkpoint) |

## Notes

- Does NOT need an NSFW checkpoint - works with standard flux1-dev
- Better poses for hitchhiking compared to base model
- Works well with accessories like holdup stockings, boots, necklaces
- Supports both nude and topless variants
- `road` gives nature/countryside setting, `street` gives urban/city setting
- Can specify age (e.g., "25yo girl")
- Front view works well for hitchhiking poses

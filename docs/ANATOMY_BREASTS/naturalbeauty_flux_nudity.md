# NaturalBeauty FLUX Nudity

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,257 |
| **👍** | 103 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `NaturalBeautyFLUXNudity.safetensors` |
| **Civitai** | https://civitai.com/models/1031648/naturalbeauty-flux-nudity |
| **Trigger word** | `naked` or `topless` |
| **Strength** | 1.0-1.5 |
| **Type** | CONCEPT (Photorealistic Full-Body Nudity) |
| **Version** | v1.0 |

## Description

Photorealistic LoRA for creating full-body nude and topless images of beautiful women with FLUX.1 dev. Result of a year-long project to hand-tag 12k images from a diverse dataset. No watermarks, no tattoos; just beautiful photographs.

**Training Data:**
- 11,714 hand-tagged images (naked and clothed, no tattoos or watermarks)
- Extensive tagging for ethnicity, breasts, nudity, clothing, leg pose, arm pose, hair, eyes, mouth, jewelry, piercing
- Cosine learning rate of 1.25e-5 over 100 epochs with batch size of 17 (UNet only)
- 4x H100, 70 hours
- LoRA: 128 dimension & 128 alpha, bf16
- kohya-ss scripts

## Tags (Supported Keywords)

### Ethnicity
- `a caucasian woman`
- `a black woman`
- `an asian woman`
- `a latina woman`
- `an indian woman`

### Nudity
- `naked`
- `topless`

### Leg Pose
- `standing, wide stance`
- `standing, feet together`
- `kneeling`
- `squatting`

### Footwear
- `barefoot`
- `high heels`

### Hair Length
- `short`
- `medium length`
- `long`
- `very long`

### Hair Style
- `braided` / `braids`
- `crimped`
- `curly`
- `ponytail`
- `straight`
- `wavy`

### Hair Color
- `black` / `dark brown` / `medium brown` / `light brown`
- `brown with blond highlights` / `chestnut brown`
- `auburn` / `colored red` / `bright red` / `orange red`
- `orange` / `red` / `strawberry blond`
- `light blond` / `medium blond` / `bleached blond` / `platinum blond`

### Expression
- `neutral expression`
- `slight smile`
- `full smile`

### Mouth
- `mouth closed`
- `mouth slightly open`
- `mouth open`

### Eyes
- `eyes closed`
- `eyes open`
- `looking at camera`

### Breasts
- `tiny breasts` / `small breasts` / `medium breasts`
- `large breasts` / `huge breasts` / `massive breasts`
- `fake breasts` / `natural breasts`

### Arm Pose
- `arms by side`
- `hands behind head`
- `hands by shoulders`
- `hands holding breasts`
- `hands on thighs`
- `hands on hips`

## Prompt Structure

Build prompts using this template:
```
a {ethnicity} woman, {leg pose}, {naked|topless}, {footwear}, {hair length} {hair style} {hair color} hair, {expression}, mouth {state}, eyes {state}, looking at camera, {breast size} {breast type} breasts, {arm pose}
```

## Sample Prompts

**Prompt 1 (Basic full body - caucasian):**
```
a caucasian woman, standing, wide stance, naked, long wavy medium blond hair, slight smile, mouth closed, eyes open, looking at camera, small breasts, hands behind head
```
Settings: Steps: 20, CFG: 3.5, Sampler: Euler

**Prompt 2 (Latina - public setting):**
```
a latina woman, long wavy brown hair, brown eyes, smiling, naked, bare feet, busy New York Central Station concourse at night, looking at camera, small breasts, standing, wide stance, public nudity, flashing
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 3 (Asian - beach):**
```
an asian woman, kneeling, naked, barefoot, medium length straight brown hair, smiling, mouth closed, eyes open, looking at camera, medium breasts, hands behind head, beach, sand, direct sunlight, wet skin, ocean, calm water
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 4 (Asian - flash photography):**
```
an asian woman, kneeling, naked, bare feet, lomo instant wide, flash photography, New York condo corridor at night, slight smile, demure, eyes open, looking at camera
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 5 (Caucasian - sunset beach):**
```
a caucasian woman, standing, wide stance, hands behind head, small breasts, medium length straight chestnut brown hair, full smile, mouth slightly open, eyes open, looking at camera, bare feet, naked, outdoors, beach, sand, rocks, calm sea, clear sky, sunset lighting, warm color tones
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 6 (Caucasian - festival topless):**
```
a caucasian woman, standing, feet together, venus de milo pose, huge breasts, arms stretched out to the side, topless, athletic arms, wellington boots, tight-fitting dark blue hipster ripped denim jeans, wet jeans, eyes closed, full smile, long wavy light blond hair, in a muddy field at glastonbury festival, dramatic purple storm clouds, festival goers in the background, shallow depth of field, heavy rain, falling rain, wet hair, vignette, muddy boot
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 7 (Caucasian - pink tights playground):**
```
a caucasian woman, standing, wide stance, wearing bright pink opaque tights, white platform heels, naked, topless, large breasts, long wavy pink hair with bangs, smiling, looking at camera, eyes open, hands on hips, standing in front of a swing in a playground, midday sun
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

**Prompt 8 (Caucasian - night street flash photography):**
```
a caucasian woman, standing, feet together, one foot in front of the other, naked, bare feet, lomo instant wide, flash photography, walking down the middle of the street in downtown New York late at night, smiling, eyes open, looking at camera, string lights, wet streets, bokeh, shallow depth of field
```
Settings: Steps: 20, CFG: 3, Sampler: Euler

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20 |
| **CFG/Guidance** | 3-3.5 |
| **Sampler** | Euler |
| **Size** | 832x1216 (or other 3:2 aspect ratio) |
| **Strength** | 1.0-1.5 |
| **Negative prompt** | Not needed |

## Notes

- Optimized for full-body portrait images at 832x1216 pixels
- Other 3:2 aspect ratio sizes should also work well
- No negative prompt needed - optimized to produce good results without one
- Follow-on to the NaturalBeauty FLUX checkpoint released November 2024
- LoRA format chosen over checkpoint based on community feedback
- Trained with kohya-ss scripts on 4x H100 GPUs

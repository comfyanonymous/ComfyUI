# Women in Pantyhose

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 152 |
| **👍** | 11 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `women_in_pantyhose.safetensors` |
| **Original filename** | `Women_in_pantyhose-000002.safetensors` |
| **Civitai** | https://civitai.com/models/1882483/women-in-pantyhose |
| **Trigger word** | `pantyhose` |
| **Strength** | 0.7-1.0 |
| **Type** | CONCEPT |

## Description

Women in pantyhose LoRA trained on 500+ images of women wearing pantyhose in various poses. First Flux experiment by the creator. Works well for realistic pantyhose renders with various poses and styles.

## Key features

- Realistic pantyhose rendering
- Various poses (sitting, laying, squatting, standing)
- Works with different styles (sheer, seamless, black)
- Compatible with topless/nude compositions
- Feet focus capability
- Full body and specific angle shots

## Recommended settings

- **Steps:** 25-35
- **CFG:** 3.5-10
- **Sampler:** DDIM / Euler
- **Size:** 832x1216

## Sample prompts

**Prompt 1 (Goth style sitting):**
```
woman wearing black seamless pantyhose. She is wearing goth makeup, with pale skin and large breasts. She is topless, sitting in a smoky room. The picture shows her full body, sitting on a chair, with crossed legs and black high heels <lora:women_in_pantyhose:1>
```
Settings: Steps: 25, CFG: 10, Size: 832x1216

**Prompt 2 (Simple laying):**
```
1girl, pantyhose, high heels, laying on bed, legs spread, large breasts, brown hair <lora:women_in_pantyhose:1>
```
Settings: Steps: 30, CFG: 7.5, Sampler: DDIM

**Prompt 3 (Yoga squat from behind):**
```
Petite, young, 18 years old, european, redhead with long flowing hair, wearing just very sheer black pantyhose:1.2, topless, arches her back, she is making deep squats on the floor in yoga studio and looking at viewer, The image captures her from below behind her and she looks at viewer. The yoga studio is modern and clean design Photorealistic, full body shot, sunset lighting <lora:women_in_pantyhose:1>
```
Settings: Steps: 35, CFG: 3.5, Size: 832x1216

**Prompt 4 (Feet focus with spread legs):**
```
1 girl sitting on bed, she is wearing black seamless pantyhose, She has blond flowing hair. She is showing her feet to camera, nylon feet, photorealistic, full body shot, spreaded legs, feet on the floor, not reinforced, feet focus, no underwear, not blurry, sensual vibe, pussy visible <lora:women_in_pantyhose:1>
```
Settings: Steps: 35, CFG: 3.5, Size: 832x1216

## Keywords

- `pantyhose`
- `black seamless pantyhose`
- `sheer black pantyhose`
- `nylon feet`
- `seamless`
- `high heels`

## Tested combinations

**With goth/fashion style LoRAs:**
```
<lora:women_in_pantyhose:1.0>
<lora:FullRebelLady:0.6>
<lora:BigTTgothGF:1.0>
```

**With pantyhose feet LoRA:**
```
<lora:women_in_pantyhose:1.0>
<lora:Pantyhose_Feet:0.7>
```

## Notes

- First Flux experiment by creator
- Trained on 500+ images
- Works with various poses
- Good for feet focus shots
- Compatible with other style/character LoRAs
- Renamed from original filename with version number

---

*Last updated: 2026-01-01*

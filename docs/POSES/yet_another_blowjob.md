# Yet Another Blowjob Lora V1

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `blowtest_v1_000002750.safetensors` |
| **Original filename** | `blowtest v1_000002750.safetensors` (renamed) |
| **Civitai** | https://civitai.com/models/960922/yet-another-blowjob-lora-v1 |
| **Trigger word** | None (use descriptive prompt) |
| **Strength** | 0.45-1.0 |
| **Type** | Pose / CONCEPT |

## Description

Flexible blowjob LoRA trained with JoyCaption. Understands clothing and various settings. Works well with character LoRAs and Acornisspinning checkpoint (use lower weights 0.45-0.6 for that). V1 release - V2 will have standardized prompts.

### Prompt template
```
{angle} photograph depicting a (woman), kneeling in front of a man, performing oral sex on a man, his legs and lower abdomen visible. with the man's large erect penis in her mouth, sucking on the large penis, indoors, {location}, the mans penis centre of frame. she is looking at the viewer with a lustful gaze, long hair
```

**Customizable parts:**
- `{angle}` → high-angle, above, low angle pov
- `{location}` → modern british livingroom, bedroom, office, etc.
- Add clothing, ethnicity, expressions as needed

## Sample prompts

**Prompt 1 (Detailed Pakistani scene):**
```
low angle pov photograph depicting a cute brown-skinned pakistani embroidered hijabi woman, in between the thighs of a (pale-skinned white caucasian albino man:1.1), performing oral sex on a man, his legs and lower abdomen visible. with the man's large erect penis in her mouth, sucking on the thick penis, the mans penis centre of frame. she is looking at the viewer with a lustful gaze. she has a shy embarrassed expression and is blushing deeply, with a naughty seductive mischievous smile. she is wearing an ornately embroidered sequined blue cotton kameez. She has thick lips and almond-shaped eyes. <lora:blowtest_v1_000002750:1>
```

**Prompt 2 (Simple living room):**
```
high-angle photograph depicting a woman, kneeling in front of a man, performing oral sex on a man, his legs and lower abdomen visible. with the man's large erect penis in her mouth, sucking on the large penis, indoors, modern british livingroom, the mans penis centre of frame. she is looking at the viewer with a lustful gaze, long hair <lora:blowtest_v1_000002750:1>
```

## Keywords

- `performing oral sex on a man`
- `kneeling in front of a man`
- `his legs and lower abdomen visible`
- `penis in her mouth`
- `sucking on the large penis`
- `looking at the viewer with a lustful gaze`
- `the mans penis centre of frame`
- `high-angle` / `low angle pov` / `above`

## Tested combinations

- Acornisspinning checkpoint (use 0.45-0.6 strength)
- Character LoRAs
- Realistic FLUX Dick LoRA
- Ethnicity LoRAs

## Strength guide

| Checkpoint | Strength |
|------------|----------|
| Standard FLUX | 0.8-1.0 |
| Acornisspinning | 0.45-0.6 |

## Recommended settings

- **Steps:** 45
- **CFG:** 1-2
- **Sampler:** DPM++ 2M
- **Schedule:** Beta (alpha: 0.6, beta: 0.6)
- **Size:** 768x1024

## Notes

- Understands clothing - can specify outfits
- Flexible with locations and settings
- Works with character LoRAs
- Use lower weights on Acornisspinning checkpoint
- Can add watermark text in prompt (trained on "Badlora.com" watermarks)
- V1 trained with varied JoyCaption prompts

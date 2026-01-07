# Blowjobs and Co

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `blowjobs_co_v025.safetensors` |
| **Original filename** | `blowjobs&co_v025.safetensors` (renamed) |
| **Civitai** | https://civitai.com/models/902999/blowjobsandco |
| **Trigger word** | None (use descriptive prompt) |
| **Strength** | 0.8-1.0 |
| **Type** | Pose / CONCEPT |

## Description

Flexible LoRA for blowjobs, handjobs, deepthroats and cumshots. High fidelity and good flexibility. POV blowjob/sucking tip pictures are easy. POV handjobs work most of the time. Distant blowjobs require precise prompting. Still in beta - can be inconsistent.

### What works well
- POV blowjob
- Sucking the tip of the penis
- Sucking an item
- POV handjobs (mostly)

### What needs precise prompting
- Distant blowjobs: "a character is kneeling and performing a blowjob on a standing man"

## Sample prompts

**Prompt 1 (POV close-up):**
```
The image is a POV close-up photograph of a young Caucasian woman about to suck a penis, she has her mouth open and she is drooling saliva. She appears to be in her early 20s has fair skin long brown hair and expressive blue eyes. She is positioned on her knees looking up at the camera with a focused expression. The man has a fair skin. The background is blurred but it seems to be an indoor setting possibly a bedroom indicated by the presence of a bed with white sheets and a beige carpeted floor. The lighting is bright and natural suggesting the photograph was taken during the day. <lora:blowjobs_co_v025:1>
```

**Prompt 2 (Simple):**
```
POV photograph of a woman sucking the tip of a penis, looking up at camera, bedroom setting <lora:blowjobs_co_v025:1>
```

## Keywords

- `POV close-up photograph`
- `about to suck a penis`
- `mouth open`
- `drooling saliva`
- `on her knees`
- `looking up at the camera`
- `sucking the tip`
- `performing a blowjob`
- `kneeling`

## Recommended settings

- **Steps:** 20-40
- **CFG:** 2.5-5 (or 1 with Distilled CFG 3)
- **Sampler:** Euler/Simple, Beta, DDEIS/DDIM, Flux Realistic
- **Schedule:** Beta (alpha: 0.6, beta: 0.6) or Simple
- **Size:** 896x1152
- **Hires:** 1.5x with 0.5 denoising (optional)

## Known issues (beta)

- Angle biases
- Cum/saliva inconsistency
- Can be unpredictable

## Notes

- Still in beta version 0.25
- POV scenes work best
- Precise prompting needed for distant shots
- High fidelity when it works
- Can be inconsistent

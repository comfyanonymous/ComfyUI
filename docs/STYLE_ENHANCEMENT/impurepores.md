# ImpurePores (Detailed Skin/Eyes | Realism Enhancer)

[← Back to STYLE_ENHANCEMENT Index](INDEX.md)

## Info
- **File:** `ImpurePores.safetensors`
- **Original filename:** `ImpurePores.safetensors`
- **Civitai:** https://civitai.com/models/902169/impurepores-detailed-skineyes-or-realism-enhancer
- **Trigger:** `xlr_skin`
- **Strength:** 0.8 (use sparingly)
- **Type:** CONCEPT / Realism Enhancement

## Description
Adds realistic skin texture imperfections to Flux generations. Can enhance smooth skin realism or add specific imperfections like blemishes, acne, wrinkles based on prompts. Use sparingly as it can also enhance unwanted imperfections.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 15-40 |
| CFG | 1 |
| Sampler | DDIM / Euler |
| Scheduler | Beta (0.6, 0.6) |
| Size | 720x904 / 896x1152 |

## Prompt Format
Base trigger for smooth textures:
```
xlr_skin
```

For specific imperfections:
```
xlr_skin, [blemish description]
xlr_skin, [acne description]
xlr_skin, [wrinkles description]
```

## Example Prompts

### Basic Close-up - Visible Pores
```
Impure pores, xlr_skin, The image shows a close-up of an individual's face with a focus on the skin texture. There appears to be a visible pores
```
Settings: Steps 15, CFG 1, DDIM, 720x904

### Blotchy Imperfect Skin
```
Impure pores, xlr_skin, The image shows a close-up of an individual's face with a focus on the skin texture. There appears to be a visible pores and blotchy imperfect skin
```
Settings: Steps 15, CFG 1, DDIM, 720x904

### Red Freckles with Red Hair
```
Impure pores, xlr_skin, The image shows a close-up of an individual's face with a focus on the skin texture. There appears to be a visible pores and blotchy imperfect skin with red freckles and red hair, low light in the room
```
Settings: Steps 15, CFG 1, DDIM, 720x904

### Blemishes and Acne
```
Impure pores, xlr_skin, In the image, you can see a close-up of a person's face. The skin texture appears to be normal with natural skin imperfections like pores and fine lines visible. There is also some mild blemishing on the forehead area, which could be indicative of minor acne or minor skin issues. The person's eyes are closed, giving a serene and relaxed appearance.
```
Settings: Steps 15, CFG 1, DDIM, 720x904

### Hispanic Woman - Fashion Editorial
```
high resolution, model photograph, closeup, focused on face, Hispanic woman, long black hair, wet straight hair, hair swept back behind her, ears visible, thin hoop earrings, stunning fit build, perfect shape, medium skin tone, thick lips, long eyelashes, dark eyeshadow, natural look, detailed skin, blemishes, freckles, glossy wet lips, sultry look, Kubrick stare, parted lips, wearing beige floral patterned frock, high-neck frock, sfw, arms behind back, pushing bust towards camera, covered bust, standing in front of a brick wall, surrounded by fireflies, glowing particles, dust in the air, golden hour lighting, dramatic shadow, cinematic, highly detailed, fashion editorial photo, high quality, DSLR camera, symmetrical photo, upper body, rule of thirds, woman in perfect center, looking at camera. <lora:ImpurePores:0.8> xlr_skin
```
Settings: Steps 40, CFG 1, Euler, Beta (0.6, 0.6), Distilled CFG 3, 896x1152

## Keywords
- `xlr_skin` (trigger)
- `pores`
- `freckles`
- `moles`
- `natural blemishes`
- `rough skin texture`
- `smooth skin texture`
- `wrinkles`
- `acne`
- `blotchy imperfect skin`
- `visible pores`

## Best Checkpoints
- flux1-dev-fp8
- FLUX-DEV-FP8

## Notes
- **Use sparingly** - strength 0.8 or lower recommended
- Can increase realism of smooth skin
- Also enhances imperfections when prompted
- Works best with close-up face shots
- Combine with skin texture descriptors for targeted effects
- For clean skin: use `xlr_skin` alone
- For imperfections: add `blemishes`, `acne`, `wrinkles` to prompt

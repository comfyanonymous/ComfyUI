# Beauty Enhancer + Realistic eyes

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 2,227 |
| **👍** | 235 |
| **Tips** | 0 |
| **Score** | ⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `beauty_enhancer_realistic_eyes.safetensors` |
| **Original filename** | `Beauty_Enahancer__Realistic_eyes_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/1397935/beauty-enhancer-realistic-eyes |
| **Trigger word** | None |
| **Strength** | 0.025-0.2 (Flux) / 0.5-1.85 (SDXL/Pony) |
| **Type** | CONCEPT |

## Description

Makes people look more beautiful/handsome while creating realistic eyes with enhanced sharpness and details. Multi-model support (Flux, SDXL, Pony). While Flux already creates good eyes, this LoRA adds additional sharpness and detail.

**IMPORTANT for Flux:** Has extremely strong effect on face/body shape - use very low strengths (0.025-0.2). Start from 0.2 and explore smaller values.

## Key features

- Enhanced eye realism and sharpness
- Face beautification
- Works with Flux, SDXL, and Pony
- Positive and negative strength both work
- Trained on amber, blue, brown, green, gray, hazel eye colors

## Recommended settings

### For Flux
- **Steps:** 25-30
- **CFG:** 7 (higher than usual for Flux)
- **Strength:** 0.025-0.2 (very low!)
- **Size:** 832x1216 or 1024x1024

### For SDXL/Pony
- **Steps:** 30
- **CFG:** 5.5-7.5
- **Sampler:** Euler_a (Pony), DPM++ 2M (SDXL)
- **Strength:** 0.5-1.85 (positive) or -1.0 (negative for cartoon effect)

## Sample prompts

**Prompt 1 (Fashion portrait - Flux):**
```
Beautiful 25 yo female model posing for a fashion portrait photoshoot. Headshot, skin details, calming look, eye focus <lora:beauty_enhancer_realistic_eyes:0.15>
```
Settings: Steps: 30, CFG: 7.5, Size: 832x1216

**Prompt 2 (Eye close-up - Flux):**
```
Close-up portrait of a female model's face, focusing on the eyes. Blue eyes. Capture fine skin textures, pores, and subtle details around the eyes, highlighting the iris and pupil with high clarity. The lighting should enhance the natural skin tones and bring out the intricate details in the eyes, creating a hyper-realistic, detailed, and intimate shot <lora:beauty_enhancer_realistic_eyes:0.15>
```
Settings: Steps: 30, CFG: 7.5, Size: 832x1216

**Prompt 3 (Athletic Asian girl - Flux):**
```
A cinematic highly realistic full body photo of a kneeling Asian girl with fit athletic body, muscular abs, huge massive fake tits, and long wavy shiny black hair in a high ponytail with a sidecut and blue rose tattoo on her arm and eyebrow piercings who has blue eyes She wears a white midriff t-shirt with the phrase "Daddy's Favorite Mouth" printed on it across her breasts, An arrow pointing up is also printed on the shirt, above the text. She also wears ripped skintight blue jeans, black choker, long fishnets gloves. She is smiling lewdly. Her very long tongue is hanging out and it is pierced. Facing viewer. She has a detailed young face, white albino skin <lora:beauty_enhancer_realistic_eyes:1>
```
Settings: Steps: 25, CFG: 3.5, Size: 1024x1024

**Prompt 4 (Swimmer - Flux with aidmafluxpro1.1):**
```
competition swimsuits, a high-resolution professional photograph of a young woman standing on a pool starting block in an indoor swimming facility, wearing a form-fitting green and black competition swimsuit with an sleek elegant design, wet hair slicked back, tan skin flushed and damp as if recently in the water, toned athletic physique, confident and happy expression, detailed realistic skin and eyes, shallow reflections of overhead lights on water surface, lane dividers visible in the pool, bleachers in the background, cinematic studio lighting highlighting her form, shallow depth of field, editorial fashion photography style, lifestyle influencer aesthetic, vibrant yet natural colors, extremely detailed, high-definition professional photography, modern athletic magazine style, aidmafluxpro1.1 <lora:beauty_enhancer_realistic_eyes:1>
```
Settings: Steps: 30, CFG: 1, Sampler: euler_simple, Size: 832x1216

**Prompt 5 (Bunny girl - Pony):**
```
real amateur photography, score_9, score_8_up, score_7_up, portrait, best quality, bunny girl, young girl, who appears to be in her early twenties, white bunny suit, shy expression, 50mm lens, cinematic composition, pretty face, Perfect Face, (detailed face), dark studio, low lights, godrays, photorealism, extreme detail, photography, real, amateur photography, hard shadows, ((realistic skin, realistic vision, photorealism, pores, skin imperfections)), very long hair, bangs <lora:beauty_enhancer_realistic_eyes:0.9>
```
Settings: Steps: 30, CFG: 5.5, Sampler: DPM++ 2M Karras, Model: CyberRealistic Pony

## Keywords

- `realistic eyes`
- `detailed eyes`
- `eye focus`
- `skin details`
- `fine skin textures`
- `pores`

## Strength guide

| Model | Positive Range | Negative Range | Notes |
|-------|----------------|----------------|-------|
| Flux | 0.025-0.2 | N/A | Very strong face effect |
| SDXL | 0.5-1.85 | -0.5 to -1.0 | Negative creates cartoon effect |
| Pony | 0.7-1.0 | Similar to SDXL | Use Euler_a sampler |

## Negative strength effects (SDXL/Pony)

- Distances person from camera
- Creates richer background
- Reduces realism
- Around -1.0: Special drawing/cartoon effect (requires "realistic eyes" in prompt)

## Tested combinations

**With fitness/body LoRAs (Flux):**
```
<lora:beauty_enhancer_realistic_eyes:1.0>
<lora:Fitness_Model:1.0>
<lora:Tongue_Long_Tongue_Out:1.0>
<lora:Perfect_Full_Round_Breasts:1.0>
```

**With FLUX Pro style:**
```
<lora:beauty_enhancer_realistic_eyes:0.3>
<lora:aidmafluxpro1.1:1.0>
```

## Notes

- Multi-model: Works with Flux, SDXL, and Pony
- For Flux: Use CFG 7 (higher than usual)
- For Flux: Start at 0.2 strength, go lower if face distorts
- Extremely small values (0.025) can work for Flux
- No trigger word needed
- Trained on 6 eye colors: amber, blue, brown, green, gray, hazel
- Renamed from original filename with typo ("Enahancer")

---

*Last updated: 2026-01-01*

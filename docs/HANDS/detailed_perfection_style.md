# Detailed Perfection Style (Hands + Feet + Face + Body)

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 83,040 |
| **👍** | 5,059 |
| **Tips** | 45,391 |
| **Score** | ⭐⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Detailed_Perfection_Style_F1D.safetensors` |
| **Original filename** | `perfection style v2d.safetensors` |
| **Civitai** | https://civitai.com/models/411088 |
| **Trigger word** | None (use quality tags) |
| **Strength** | 0.9-1.0 |
| **Type** | STYLE / Quality Enhancement / All-in-One |
| **Version** | Perfection F1D v2.5 |

## Description

All-in-one quality enhancement LoRA for hands, feet, face, and body. Trained on SDXL 1.0 + F1D. Can be used as an extra layer without any prompt when combined with other LoRAs.

### Key Features
- Hand anatomy improvement
- Feet detail enhancement
- Face perfection
- Body detail
- Works with any model (Juggernaut, Realistic Stock Photo, etc.)
- Can be used without specific prompts

### Available Versions
- **F1D** (FLUX) - This version
- SDXL
- SD1.5
- Pony
- Illustrious

## Sample Prompts

**Prompt 1 (Barbarian portrait):**
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Detailed_Perfection_Style_F1D:1>
A perfect photo of a close up of a barbarian man with a serious look, solo, looking at viewer, 1boy, brown eyes, closed mouth, male focus, facial hair, scar, thick eyebrows, portrait, beard, close-up, realistic, manly, detailed face, detailed body, detailed hands, detailed eyes, detailed nose, detailed ears, detailed hair, detailed, perfection, detailed teeth, detailed skin texture, wrinkly, detailed fingers, detailed mouth, beauty, realism, real, detailed hair, detailed pores, detailed background, sharp image, detailed lips, perfection style. shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, epic, beautiful lighting, inspiring
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

**Prompt 2 (Handshake - hand focus):**
```
UHD, 4k, ultra detailed, cinematic, a photograph of two people shaking hands, hand focus, epic, beautiful lighting, inspiring <lora:Detailed_Perfection_Style_F1D:1>
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

**Prompt 3 (Feet focus - nail polish):**
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Detailed_Perfection_Style_F1D:1>
A perfect photo of a woman with blue nail polish sitting on a chair, 1girl, solo, barefoot, nail polish, feet, toes, shadow, soles, close-up, blue nails, toenails, realistic, toenail polish, foot focus, aqua nails, detailed face, detailed body, detailed hands, detailed eyes, detailed nose, detailed ears, detailed hair, detailed, perfection, detailed teeth, detailed skin texture, wrinkly, detailed fingers, detailed mouth, beauty, realism, real, detailed hair, detailed pores, detailed background, sharp image, detailed lips, perfection style, epic, beautiful lighting, inspiring
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

**Prompt 4 (Woman driving - cinematic film):**
```
cinematic film still of <lora:Detailed_Perfection_Style_F1D:0.9>
A perfect photo of a woman driving a car in the rain, 1girl, solo, long hair, black hair, jewelry, teeth, blurry, lips, ring, clenched teeth, ground vehicle, realistic, car, driving, steering wheel, detailed face, detailed body, detailed hands, detailed eyes, detailed nose, detailed ears, detailed hair, detailed, perfection, detailed teeth, detailed skin texture, wrinkly, detailed fingers, detailed mouth, beauty, realism, real, detailed hair, detailed pores, detailed background, sharp image, detailed lips, perfection style, shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

**Prompt 5 (Blonde portrait):**
```
cinematic film still of <lora:Detailed_Perfection_Style_F1D:0.9>
A perfect photo of a woman with blonde hair and a necklace, 1girl, solo, long hair, looking at viewer, blonde hair, simple background, white background, closed mouth, lips, portrait, realistic, detailed face, detailed body, detailed hands, detailed eyes, detailed nose, detailed ears, detailed hair, detailed, perfection, detailed teeth, detailed skin texture, wrinkly, detailed fingers, detailed mouth, beauty, realism, real, detailed hair, detailed pores, detailed background, sharp image, detailed lips, perfection style, shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

## Negative Prompt (Recommended)

```
ugly, deformed, noisy, blurry, low contrast, anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, photograph, deformed, glitch, noisy, realistic, stock photo, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured, noise, noisy, ugly breasts, tripod, camera, (censorship, censored, worst quality, low quality, normal quality, lowres, low details, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry), morbid, ugly, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, JPEG artifacts, out of focus, glitch, duplicate, (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3), ugly fingers, bad fingers, (((ugly nipples, bad nipples, deformed nipples))), (((Bad teeth, ugly teeth)))
```

## Quality Tags

### Essential Tags
- `detailed face`
- `detailed body`
- `detailed hands`
- `detailed eyes`
- `detailed nose`
- `detailed ears`
- `detailed hair`
- `detailed teeth`
- `detailed skin texture`
- `detailed fingers`
- `detailed mouth`
- `detailed pores`
- `detailed lips`
- `perfection`
- `perfection style`

### Enhancement Tags
- `wrinkly`
- `beauty`
- `realism`
- `real`
- `sharp image`
- `UHD`
- `4k`
- `ultra detailed`

### Cinematic Tags
- `shallow depth of field`
- `vignette`
- `highly detailed`
- `high budget`
- `bokeh`
- `cinemascope`
- `moody`
- `epic`
- `gorgeous`
- `film grain`
- `grainy`
- `beautiful lighting`
- `inspiring`

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 1024x1024 |
| **Strength** | 0.9-1.0 |

## Related LoRAs by Same Author

- **Detailed Style XL (Hand focus)** - "Detailed Perfection style extension"
- **Skin Tone (Cinematic Photography) Style XL** - Human skin color
- **Realistic Skin Texture style XL** - Detailed skin
- **Facial Expressions (detailed emotions) style XL**
- **Perfect Eyes (Variety of sclera) XL**
- **Hands asset model**
- **Feet asset model**

## Notes

- No trigger word needed - use quality tags for best results
- Works as an extra layer with any other LoRA
- Can be used without specific prompts
- Compatible with Juggernaut, Realistic Stock Photo, and most models
- Use 0.9-1.0 strength for full effect
- Works well with cinematic and photographic styles
- Multiple model versions available (SDXL, SD1.5, Pony, Illustrious)


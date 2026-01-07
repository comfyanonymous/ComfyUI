# Realistic Portraits of Black Women in Europe and America

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | N/A |
| **👍** | N/A |
| **Tips** | N/A |
| **Score** | - |

**Note:** Civitai URL not found during search.

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Realistic_Black_Women_Portraits.safetensors` |
| **Original filename** | `【欧美】写实黑人女性人像_1.0.safetensors` |
| **Civitai** | Not found |
| **Trigger word** | None (descriptive prompts) |
| **Strength** | 0.8-1.0 (recommended: 1.0) |
| **Type** | CHARACTER (Black Women Portraits) |

## RECOMMENDED - Planned for Future Workflows

This LoRA generates easy, consistent faces of young black women with high quality skin detail and studio lighting. Useful for portrait workflows and character generation.

## Description

FLUX LoRA for generating realistic portraits of Black women with European/American styling. Creates high-quality studio portraits with detailed skin, proper lighting, and professional photography aesthetics. Works well for both bust shots and full body images.

**Capabilities:**
- Realistic Black women portraits
- Studio lighting / edge lighting
- High-quality skin with pores and details
- Professional fashion photography
- Various poses (bust, full body)
- Works with film grain aesthetics (Fujifilm XT3)

## Sample Prompts

### Studio Portrait with Dress
```
A photo of a 20-year-old black girl with curly black hair, smiling, wearing a dress, full body, edge lighting, studio lighting, solid color background, ultra-high quality, clear focus, film grain, Fujifilm XT3, 8K UHD, highly delicate skin
```
Settings: Steps 20, CFG 3.5, Euler, 768x1024, Hires: 8x-NMKD-Superscale, Denoise 0.23

### High-End Fashion
```
A photo of a black girl wearing high-end fashion, with small curly black hair, smiling, full body image, edge lighting, studio lighting, looking at the camera, with a dark red blurred background, dslr, Ultra high quality, clear focus, sharp stickiness, degree of freedom, film particles, Fujifilm XT3, crystal clear, 8K UHD, highly delicate luster eyes, highly delicate skin, skin pores
```
Settings: Steps 30, CFG 3.5, Euler, 768x1024, Hires: 8x-NMKD-Superscale, Denoise 0.23

### Bust Portrait
```
A photo of a beautiful black girl with small curly black hair, wearing a dress, a bust, edge lighting, studio lighting, looking at the camera, dslr, Ultra high quality, clear focus, sharp stickiness, degree of freedom, film particles, Fujifilm XT3, crystal clear, 8K UHD, highly delicate luster eyes, highly delicate skin, skin pores
```
Settings: Steps 30, CFG 3.5, Euler, 768x1024

### Venice Evening Glamour
```
A close-up from the front shows the black-haired caucasian model seated on the edge of a marble fountain in a quiet Venetian piazza under a rich golden sunset. Her legs are crossed elegantly, back straight with her shoulders pulled gently back to reveal her neckline and posture. She wears a burgundy red and black satin evening dress with a plunging deepneck and a low back, hugging her hips and revealing generous cleavage. Her heels are gold stilettos, detailed female face, green eyes, red lips
```
Settings: Steps 30, CFG 5.5, 1024x1024

## Keywords

### Subject
- `black girl`
- `black woman`
- `20-year-old`
- `beautiful black girl`

### Hair
- `curly black hair`
- `small curly black hair`
- `long straight black hair`

### Lighting
- `edge lighting`
- `studio lighting`
- `dslr`
- `golden sunset`

### Quality
- `ultra-high quality`
- `clear focus`
- `film grain`
- `Fujifilm XT3`
- `8K UHD`
- `crystal clear`

### Skin
- `highly delicate skin`
- `skin pores`
- `highly delicate luster eyes`

### Composition
- `full body`
- `bust`
- `looking at the camera`
- `solid color background`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-30 |
| **CFG** | 3.5-5.5 |
| **Sampler** | Euler |
| **Size** | 768x1024 / 1024x1024 |
| **Strength** | 0.8-1.0 (recommended: 1.0) |
| **Clip Skip** | 1-2 |

## Hires Fix Settings

| Parameter | Value |
|-----------|-------|
| **Upscaler** | 8x-NMKD-Superscale / ESRGAN 4x+Anime6B |
| **Hires Steps** | 10-20 |
| **Denoise** | 0.20-0.35 |
| **Upscale By** | 1.2x-4.0x |

## Recommended Combinations

### With Skin Detail
```
<lora:Realistic_Black_Women_Portraits:1>
<lora:detail_enhancer_flux_v1:0.7>
```

### With Realism
```
<lora:Realistic_Black_Women_Portraits:1>
<lora:flux_realism_lora:0.6>
```

## Notes

- Use strength 1.0 for closest to original LoRA style
- Best with studio lighting and professional photography prompts
- Works for full body and bust shots
- Good for fashion and portrait photography
- Can also work with Asian women prompts (see examples)
- Generates consistent, easy faces of young black women
- Planned for use in future portrait workflows


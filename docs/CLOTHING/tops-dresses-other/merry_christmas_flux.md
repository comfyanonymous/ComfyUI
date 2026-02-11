# Merry Christmas Flux

[← Back to Index](../INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 309 |
| **👍** | 17 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Merry_Christmas_Flux.safetensors` |
| **Original filename** | `merry_christmas_flux-1.safetensors` |
| **Civitai** | https://civitai.com/models/1068475 |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | CONCEPT (Christmas Clothing) |

## Description

Christmas dress LoRA trained with 500 high quality photos. Specializes in generating:
- Christmas dresses in red and white
- Bodycon dresses with Christmas arts
- Candy cane stripes and snowflake patterns
- Ball gowns with holiday embroidery
- Festive holiday backgrounds

**Recommendation:** Use Adetailer extension for best face results.

## Sample Prompts

### Red Satin Dress with Christmas Tree
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, dslr, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting, 1girl, wearing a stunning red silk ball gown with golden floral embroidery and delicate white lace along the hemline, festive holiday background with twinkling lights and a decorated Christmas tree, soft focus background, studio lighting, looking at viewer, photorealistic, collarbone accentuated, depth of field, elegant holiday dress design, solo
```
Settings: Steps 20, CFG 1, Euler, 640x960, Distilled CFG 3.5

### White Ball Gown with Snowflake Embroidery
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, dslr, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting, 1girl, wearing an elegant white satin ball gown with intricate silver snowflake embroidery and shimmering golden accents, festive holiday background with snowy decor and twinkling fairy lights, soft focus background, studio lighting, looking at viewer, photorealistic, collarbone accentuated, depth of field, graceful holiday dress design, solo
```
Settings: Steps 20, CFG 1, Euler, 640x960, Distilled CFG 3.5

### Candy Cane Stripe Christmas Dress
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, DSLR, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting, sunny, tight dress, Christmas print dress in red and white with candy cane stripes and snowflake patterns, 1 girl, face, woman, medium breasts, blonde hair, slim waist, slim body, neutral holiday-themed background, soft focus background, studio lighting, delicate details, long cascading hair, looking at viewer, photorealistic, accentuated collarbone, depth of field, elegant Christmas dress design, solo, cinematic shadow
```
Settings: Steps 20-35, CFG 1, Euler, 640x960, Distilled CFG 3.5

### Red Satin with White Lace (Red Hair)
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, dslr, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting, 1girl, wearing a elegant red satin dress with white lace accents and a sweetheart neckline, slim body type, long wavy hair, festive holiday background, soft focus, studio lighting, intricate details, looking at viewer, photorealistic, collarbone accentuated, depth of field, unique holiday design, red hair
```
Settings: Steps 35, CFG 1, Euler, 640x960, Distilled CFG 3.5

### White Ball Gown with Red Floral Embroidery
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, dslr, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting, 1girl, wearing a white ball gown with intricate red floral embroidery and a flowing train, slim body type, long wavy hair, festive holiday background, soft focus, studio lighting, intricate details, looking at viewer, photorealistic, collarbone accentuated, depth of field, unique holiday design, solo
```
Settings: Steps 35, CFG 1, Euler, 640x960, Distilled CFG 3.5

## Keywords

### Quality Booster (Recommended Prefix)
```
(masterpiece:1.3, realistic:1.3), best quality, ultra detailed, intricate, professional photography, HDR, High Dynamic Range, (8k UHD), RAW photo, dslr, realistic LUT, cinematic LUT, perfect lighting, professional lighting, cinematic lighting, cinematic shadows, iridescent lighting
```

### Dress Styles
- `red silk ball gown`
- `white satin ball gown`
- `tight dress`, `bodycon dress`
- `Christmas print dress`
- `elegant red satin dress`
- `white ball gown with flowing train`

### Patterns & Details
- `candy cane stripes`
- `snowflake patterns`
- `silver snowflake embroidery`
- `golden floral embroidery`
- `white lace accents`
- `sweetheart neckline`
- `shimmering golden accents`

### Backgrounds
- `festive holiday background`
- `twinkling lights`
- `decorated Christmas tree`
- `snowy decor`
- `twinkling fairy lights`
- `neutral holiday-themed background`

### Photography Style
- `soft focus background`
- `studio lighting`
- `depth of field`
- `collarbone accentuated`
- `cinematic shadow`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-50 |
| **CFG** | 1 (Distilled CFG 3.5) |
| **Sampler** | Euler |
| **Size** | 640x960 (portrait) |
| **Strength** | 1.0 |
| **Clip Skip** | 1-2 |

### Resolution Options
| Input | Output (with Hires.fix) |
|-------|------------------------|
| 512x768 | Standard |
| 768x512 | Landscape |
| 540x960 | Standard |
| 640x960 | Standard |
| 540x540 | 1080x1080 |
| 540x960 | 1080x1920 |

## ADetailer Settings (Recommended)

| Parameter | Value |
|-----------|-------|
| **Model** | face_yolov8n.pt |
| **Mask blur** | 4 |
| **Confidence** | 0.3 |
| **Dilate erode** | 4 |
| **Inpaint padding** | 32 |
| **Denoising strength** | 0.4 |
| **Inpaint only masked** | True |

## Notes

- Trained on 500 high quality photos
- Works best with Flux1-dev
- Adetailer recommended for face quality
- Use quality prefix for best results
- Red/white color schemes work best
- Supports both ball gowns and tight bodycon dresses
- Good with festive holiday backgrounds


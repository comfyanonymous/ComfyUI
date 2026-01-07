# Insta Choc 1.3

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 103 |
| **👍** | 10 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Insta_Choc_1.3.safetensors` |
| **Original filename** | `Insta_choc_1.3.safetensors epoch 5.safetensors` |
| **Civitai** | https://civitai.com/models/1383383 |
| **Trigger word** | None (descriptive prompts) |
| **Strength** | 1.0 |
| **Type** | CHARACTER (Instagram Model Style) |

## RECOMMENDED - Planned for Future Workflows

Generates sexy faces of young attractive Black and Latina women with Instagram-model aesthetic. Good for diverse social media style portraits.

## Description

Character LoRA for generating realistic Instagram-model-style visuals of African American and Latina women. Not based on any specific person - trained to reflect common traits found in real-world social media content.

**Capabilities:**
- Fuller body types with thick thighs
- Slim waists with hourglass figures
- Distinctive facial features
- Diverse hairstyles (curly, natural waves, styled edges)
- Various skin tones (medium-dark, caramel)
- Instagram influencer aesthetic
- Culturally nuanced outputs

**Note:** Early version - may have some "Flux shine". Future versions will improve skin/hair texture variety.

## Sample Prompts

### Casual Plaid Outfit (Sexy Face)
**RECOMMENDED - Very attractive young Black woman face**
```
<lora:Insta_Choc_1.3:1>this photograph features a young woman with a medium-dark skin tone and curly, voluminous hair styled in loose, natural waves. She has a contemplative, slightly serious expression on her face, with her gaze directed off-camera. She is dressed in a casual, eclectic outfit consisting of a loose, unbuttoned plaid shirt in shades of orange, brown, and green. Underneath the shirt, she wears a deep orange top with a V-neck that reveals a hint of a dark bralette beneath. The top has a slightly distressed, textured appearance.
```
Settings: Steps 150, CFG 1, DPM++ 2M, 512x512, Distilled CFG 3, Hires: ESRGAN_4x 2x

### Pool Party Bougie Style
```
<lora:Insta_Choc_1.3:1>19-year-old bougie girly athletic Black woman with an hourglass figure, featuring muscular thighs, wide round hips, a flat, toned stomach, and light caramel skin. She has reddish hair styled with curly edges and glossy lips. She is wearing a see-through crochet bathing suit in rainbow colors. The outfit accentuates her small waist, wide hips, and athletic build. She also has tattoos, few skin tags, and diamond Bulgari jewelry. The scene is set at a luxurious mansion pool party with people in the background partying in swimwear, holding red cups, and some smoking weed. The environment should have a strong depth of field to highlight her presence while keeping the background detailed but slightly out of focus. The woman is wearing Cartier sunglasses and a diamond hip-hop grill. The overall look is bougie, Instagram-model-like, with a confident smile and dimples showing.
```
Settings: Steps 66, CFG 1, DPM++ 2M, 1024x1024, Distilled CFG 3

## Keywords

### Body Type
- `hourglass figure`
- `thick thighs`
- `wide round hips`
- `slim waist`
- `small waist`
- `flat, toned stomach`
- `muscular thighs`
- `athletic build`
- `fuller body`

### Skin Tone
- `medium-dark skin tone`
- `light caramel skin`
- `African American`
- `Black woman`
- `Latina`

### Hair
- `curly, voluminous hair`
- `natural waves`
- `reddish hair`
- `curly edges`
- `styled edges`

### Style
- `Instagram-model-like`
- `bougie`
- `confident smile`
- `dimples`
- `glossy lips`
- `tattoos`
- `diamond jewelry`
- `Bulgari jewelry`
- `Cartier sunglasses`
- `hip-hop grill`

### Settings
- `pool party`
- `luxurious mansion`
- `depth of field`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 66-150 |
| **CFG** | 1 |
| **Distilled CFG** | 3 |
| **Sampler** | DPM++ 2M |
| **Size** | 512x512 / 1024x1024 |
| **Strength** | 1.0 |

## Hires Fix Settings

| Parameter | Value |
|-----------|-------|
| **Upscaler** | ESRGAN_4x |
| **Hires Upscale** | 2x |
| **Hires Steps** | 150 |
| **Denoise** | 1.0 |

## Recommended Combinations

### With Realism
```
<lora:Insta_Choc_1.3:1>
<lora:flux_realism_lora:0.6>
```

### With NSFW
```
<lora:Insta_Choc_1.3:1>
<lora:MysticXXX-v6:0.5>
```

### With Detail Enhancement
```
<lora:Insta_Choc_1.3:1>
<lora:detail_enhancer_flux_v1:0.7>
```

## Notes

- No trigger word needed - use descriptive prompts
- Early version (1.3) - may have typical Flux shine
- Not based on any real person/celebrity
- Designed for diversity often missing in default models
- Good for Instagram influencer / social media style
- Works for both African American and Latina women
- First prompt produces particularly attractive faces
- Supports various body types and skin tones


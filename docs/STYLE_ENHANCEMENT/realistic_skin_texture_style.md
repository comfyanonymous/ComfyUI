# Realistic Skin Texture Style (Detailed Skin)

[← Back to STYLE_ENHANCEMENT Index](INDEX.md)

## Info
- **File:** `Realistic_Skin_Texture_Style.safetensors`
- **Original filename:** `skin texture style v5.safetensors`
- **Civitai:** https://civitai.com/models/580857/realistic-skin-texture-style-detailed-skin-xl-sd15-f1d-pony-illu-zit
- **Trigger:** `skin texture style`
- **Strength:** 0.9-1.0
- **Type:** STYLE / Enhancement

## Description
Multi-model skin texture enhancement LoRA. Adds highly detailed, realistic skin pores, texture, and natural imperfections. Works with FLUX, SDXL, SD1.5, Pony, Illustrious, and ZIT models. Can be combined with Skin Tone Style XL for human skin color.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 20-33 |
| CFG | 1-7 |
| Sampler | Euler |
| Distilled CFG | 3.5 |
| Size | 1024x1024 / 832x1216 |

## Prompt Format
Include trigger phrase and skin descriptors:
```
<lora:Realistic_Skin_Texture_Style:1> skin texture style, detailed skin pore, perfect skin
```

Or use style prefixes:
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Realistic_Skin_Texture_Style:0.9>
```

## Example Prompts

### Portrait - Freckled Woman
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Realistic_Skin_Texture_Style:1>
a closeup of a white woman with freckles on her face and a messy hair and deep pink rose lips, perfect skin, detailed skin pore, realism style, perfect image, perfect anatomy, sharp image, detailed image, high quality photography, skin texture style, solo, looking at viewer, short hair, blue eyes, brown hair, green eyes, parted lips, teeth, lips, portrait, close-up, freckles, photorealistic, staring at camera
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Hyperrealistic - Hand with Pistol
```
Hyperrealistic art of <lora:Realistic_Skin_Texture_Style:1>
a closeup of a man's hand holding a colt pistol in his hand, perfect skin, detailed skin pore, realism style, perfect image, perfect anatomy, sharp image, detailed image, high quality photography, skin texture style, solo, long sleeves, holding, weapon, holding weapon, gun, handgun, m1911, photorealistic, hand focus, Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Close-up - Mouth and Tongue
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Realistic_Skin_Texture_Style:0.9>
detailed photorealism style, hyperrealism art style, realistic textures, photorealistic style, A cinematic skin texture style still image of a close up of a person's mouth with a teeth, detailed skin pore, film still, still photography style, sharp style, detailed style, Kodak film skin tone style, fujifilm skin tone style, professional photography style, skin textured, skin texture style, 1girl, solo, open mouth, teeth, tongue, tongue out, lips, eyelashes, close-up, realistic
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Hyperrealistic - Old Man Smoking
```
Hyperrealistic art of <lora:Realistic_Skin_Texture_Style:1>
a closeup of an old man with a mustache and a hat smoking a cigarette while looking at camera, perfect skin, detailed skin pore, realism style, perfect image, perfect anatomy, sharp image, detailed image, high quality photography, skin texture style, solo, hat, facial hair, black background, portrait, beard, smoke, cigarette, smoking, Extremely high-resolution details, photographic, realism pushed to extreme, fine texture, incredibly lifelike
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Feet Close-up
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Realistic_Skin_Texture_Style:0.9>
detailed photorealism style, hyperrealism art style, realistic textures, A cinematic skin texture style still image of a woman's bare feet sitting on a concrete ledge, detailed skin pore, film still, professional photography style, skin textured, skin texture style, 1girl, solo, outdoors, barefoot, nail polish, feet, toes, shadow, soles, red nails, close-up, toenails, realistic, foot focus, dirty feet, blurry background
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Portrait - Braided Hair Woman
```
UHD, 4k, ultra detailed, cinematic, a photograph of <lora:Realistic_Skin_Texture_Style:1>
detailed photorealism style, hyperrealism art style, realistic textures, A cinematic skin texture style still image of a woman with a braid in her hair and many freckles, detailed skin pore, film still, still photography style, professional photography style, skin textured, skin texture style, 1girl, solo, long hair, looking at viewer, smile, brown hair, brown eyes, braid, teeth, grin, single braid, portrait, realistic, sweater, freckles, photorealistic
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1024x1024

### Asian Girl in Car - NSFW
```
realism, realistic girl, asian girl, sitting in a car, inside a rolls royce, sexy pose, naked breasts, black dress, white mink coat, glasses, pantyhose, photorealism, real girl, photo, dynamic pose, dynamic angle, hand near face, sexy look, 8k, hdr, realistic face, realistic skin, pantyhose, high heels
```
Settings: Steps 20, CFG 7, Euler

### Athletic Girl Pool - NSFW
```
R3alisticF, skin texture style, detailed, photograph, hauntingly beautiful intricate details, 1girl, 20 years old, long hair, green eyes, blonde hair, soft defined muscles, thin, slight abs, fit, athletic hips, athletic ass, strong athletic muscular legs, wet skin, wet hair, photo, detailed, sharp focus, playful innocent, flirtatious, seductive, subtle smile, wearing a white erotic bathing suit, sun hat, glasses, wet clothing, body silhouette, skin indention, fit girl with perfect body, cute, pleasant, seductive, raw, realistic, sunset, god rays, outdoors steeping out of a crystal clear pool of water an exotic private garden at a palace, focus on feet, water dripping of her skin, good lighting, Perfect hand, High Detail, Perfect Composition, silhouetted
```
Settings: Steps 33, CFG 3.5, 832x1216

## Keywords
- `skin texture style` (trigger)
- `detailed skin pore`
- `perfect skin`
- `realistic skin`
- `hyperrealism`
- `photorealistic`
- `fine texture`
- `Kodak film skin tone style`
- `fujifilm skin tone style`

## Recommended LoRA Combinations
- **Skin Tone Style XL** - human skin color enhancement
- **Hands XL F1D** - better hand details
- **Real Nipples and Areola Textures-GMR** - NSFW anatomy
- **Perfect Full Round Breasts & Slim Waist** - body shape
- **KFT's HLA Image Enhancer** (0.35) - image quality

## Best Checkpoints
- FLUX Dev
- flux_dev_fp32

## Notes
- Works across multiple model architectures (FLUX, SDXL, SD1.5, Pony, Illustrious, ZIT)
- Excellent for close-up portraits and body detail shots
- Use strength 0.9-1.0 for optimal skin texture
- Combine with hyperrealism style prompts for best results
- Include "detailed skin pore" in prompts for enhanced effect
- Works for hands, feet, face, and full body shots

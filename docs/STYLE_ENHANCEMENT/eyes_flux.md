# Eyes (FLUX)

[← Back to STYLE_ENHANCEMENT Index](INDEX.md)

## Info
- **File:** `Eyes_FLUX.safetensors`
- **Original filename:** `Eyes-000001.safetensors`
- **Civitai:** https://civitai.com/models/850406/eyes
- **Trigger:** None
- **Strength:** 0.35-1.0
- **Type:** CONCEPT / Enhancement

## Description
Multi-model eye enhancement LoRA. Makes better, more detailed and realistic eyes. Simple to use - just add to any generation for improved eye quality.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 25-50 |
| CFG | 1-6 |
| Sampler | Undefined / Euler / Heun |
| Strength | 0.35-1.0 |

## Example Prompts

### Eye Close-up - Blue Eyes
```
eye focus, close-up, solo, eyelashes, realistic, looking at viewer, reflection, blue eyes
This is a high-resolution photograph of a human eye, focusing on the eye's details. The image captures the eye in a close-up, macro view, revealing intricate details. The eye is a bright blue, with a gradient effect from the darker blue of the iris to the lighter blue of the sclera. The pupil is black and round, taking up a significant portion of the iris.
```
Settings: Steps 28, CFG 3.5, 1024x1024

### Eye Close-up with Depth of Field
```
eye focus, solo, close-up, blurry, 1girl, eyelashes, depth of field
This is a high-resolution photograph focusing on a close-up view of a single eye, specifically the right eye, of a person. The image is taken from a side angle, capturing the eye and part of the forehead. The eye is a striking blue-gray color with a clear, shiny appearance, indicating it is well-moisturized and healthy.
```
Settings: Steps 28, CFG 3.5, 1024x1024

### Fashion Model Portrait
```
fashion model, elegant and charming, blushing, realistic details and sharp facial features and expressions, (perfect natural breasts), portrait of A 1 Woman:1.4 20yo, (intricately detailed, clear and sharp perfect round realistic brown_eyes:1.35), (droopy eyes), symmetrical lips, light glossy red_lipstick, long eyelashes, (vivid and colorful), (perfect composition)
```
Settings: Steps 40, CFG 6, Euler

### E-Girl Portrait
```
A stunningly beautiful slim 20 year old with brown hair blonde highlights and unique make is wearing goth clothes and smiling at the viewer in her bedroom, blue eyes
```
Settings: Steps 30, CFG 3.5, 832x1216, Strength 1.0

### Dark-skinned Woman Portrait
```
portrait of a dark-skinned woman, glowing skin, flowing silk and delicate lace swirling around her upper body and arms as if alive, cow-boy shot showing shoulders and hands, elegant fingers manipulating floating fabric, soft warm studio light highlighting texture and curves, subtle rim light enhancing lace transparency, shallow depth of field blurring background, cinematic ultra-realistic render, (fine skin texture:1.2), emotion of graceful allure and control
```
Settings: Steps 40, CFG 1, Heun

### Nude Portrait - Redhead
```
AP_Nude, Solo, 1Woman, Stark Naked, Fully Nude, Raw Ultra Hires Photo of a nude young woman standing in a well-lit, modern indoor setting. The subject is a fair-skinned, slender woman with long, straight, reddish-blonde hair that falls to her shoulders. She has a small to medium-sized bust, with small, perky breasts and a flat stomach. ((freckles, hourglass body shape. Real Nipples and Areola Textures)), RNAT, APW_Flux, Pleasant:Redhead, Freckles
```
Settings: Steps 50, CFG 5, 832x1216, Strength 0.35

### Merida Cosplay
```
Photorealistic, glamour photo, highly detailed, 1woman, (Merida from Brave:0.9), solo:1.15, sultry:1.15, cute round face, sexy:1.2, petite, freckles:0.7, tiny perky breasts:1.5, dynamic posing:1.35, swept bangs, medium hair, wavy hair:0.8, orange hair, ginger hair, blue eyes, full heart shaped lips:1.3, lips parted:1.5, (sheer blue camisole:0.9), see through:1.2, no bra, looking at viewer, seductive eyes:1.25, mischievous expression:1.15, seductive smile, blush, in an elegant bar, leaning forward on the counter
```
Settings: Steps 25, CFG 3.5, 832x1216, Strength 1.0

## Keywords
- `eye focus`
- `close-up`
- `eyelashes`
- `realistic`
- `looking at viewer`
- `reflection`
- `blue eyes` / `brown eyes` / `green eyes`
- `depth of field`

## Recommended LoRA Combinations
- **FLUX Image Upgrader / Detail Maximizer** (FLUX v0.3) - detail enhancement
- **Flux Detailer** (V3) - additional details
- **Epic gorgeous Details** - balance
- **NSFW FLUX1 D UNLOCKED** - NSFW unlock
- **Hourglass Body Shape** (0.5-1.0) - body shape
- **Real Nipples and Areola Textures-GMR** (0.6) - anatomy
- **Freckles FLUX** (1.0) - freckles
- **InSaNe DETAIL SLIDER** (V1) - detail slider

## Best Checkpoints
- FLUX Dev
- flux1-dev

## Notes
- No trigger word required
- Supports multiple models
- Lower strength (0.35) for subtle enhancement
- Higher strength (1.0) for prominent eye detail
- Works well with eye color specifications in prompt
- Good for both close-up eye shots and full portraits
- Combine with detail enhancers for best results

# Tongue/Long Tongue Out (FLUX + SDXL)

[← Back to CONCEPTS Index](INDEX.md)

## Info
- **File:** `Long_Tongue_Out_FLUX.safetensors`
- **Original filename:** `concept_long_tongue_out_flux_1_standard-000017.safetensors`
- **Civitai:** https://civitai.com/models/293833/tonguelong-tongue-out-flux-sdxl
- **Trigger:** `sticking tongue out`
- **Strength:** 0.8-1.2 (around 1.0 recommended)
- **Type:** CONCEPT

## Description
LoRA trained on 80 images for "tongue out" and "long tongue out" concepts. Captioned with GPT-4 Vision. Tested on FLUX 1.D fp8, NF4 and FLUX 1.D full. Can be combined with other LoRAs.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 20-25 |
| CFG | 1-3.5 |
| Sampler | Euler / IPNDM |
| Strength | 0.8-1.2 |

## Prompt Format
Main structure:
```
[View], [mouth openness], {style and scene description}, {character description} with {her/his/their} [tongue length] sticking out
```

## Keywords

### View
- `Frontview`
- `Sideview`

### Tongue Length
- `long tongue out`
- `tongue out`
- `very long tongue`

### Mouth Openness
- `mouth wide open, uvula`
- `mouth open`

## Example Prompts

### Cute Girl with Cum Effect
```
18 year old girl, freckles, cute round face, choker, pink wavy side swept bobcut with blue and purple highlights, smile, tongue out, butterfly hair pin, outdoors, a thick white murky frothy liquid that resembles semen is dripping all over her forehead cheeks and chin, pink eyeshadow, lipstick, her skin is extremely pale, detailed body, detailed face, detailed eyes, clear image, detailed, vivid image, hd, high quality, 4k
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 1416x792, Strength 0.9

### With Braces
```
18 year old girl, freckles, cute round face, choker, pink wavy side swept bobcut with blue and purple highlights, smile, teeth, braces, she is sticking her tongue out, butterfly hair pin, outdoors, wearing pink tinted heart shaped glasses, peace sign, pink eyeshadow, lipstick, detailed body, detailed face, detailed eyes, clear image, detailed, vivid image, hd, high quality, 4k
```
Settings: Steps 25, CFG 1, Euler, Distilled CFG 3.5, 1416x792, Strength 0.9

### British Woman - Bedroom
```
This image depicts a young British woman with bright eyes, a curvy figure and natural brown hair styled in a bun, kneeling on the floor of a bedroom. The woman's bathrobe is open, and is holding her own breasts, covering her nipples with her hands, squeezing her breasts together, handbra. Her bright eyes are fixed on the viewer, with a subtle blush on her cheeks. Her mouth is open with her tongue out. The image is rendered in stunning 8K resolution, with hyper-realistic detail that makes the scene feel almost lifelike. The viewer's perspective is from above eye level, looking down on the woman.
```
Settings: Steps 25, CFG 3.5, 1024x1024, Strength 1.0

### Portrait - Studio
```
SFW, Professional DSLR photo, young Caucasian woman, portrait, 18 years old, ((pale skin)), high ponytail hair, in a studio setting, grinning, global illumination, (((facing the viewer))), looking at the viewer, silver hoop earrings, glamour makeup, large firm breasts, dressed in a tight black halter top, full depth of field, frontview, [focus on head and upper torso] [sticking tongue out, mouth open wide]
```
Settings: Steps 25, CFG 1, IPNDM, Beta (0.6, 0.6), Distilled CFG 3.5, 1024x1024, Strength 0.4

### Asian Girl - Tongue Piercing
```
A cinematic highly realistic full body photo of a kneeling Asian girl with fit athletic body, muscular abs, huge massive fake tits, and long wavy shiny black hair in a high ponytail with a sidecut and blue rose tattoo on her arm and eyebrow piercings who has blue eyes She wears a white t-shirt with the phrase "Daddy's Favorite Mouth" printed on it and a skintight blue jeans, black choker, long fishnets gloves, detailed young face, white albino skin. She is smiling lewdly. Her very long tongue is hanging out and it is pierced.
```
Settings: Steps 25, CFG 3.5, 1024x1024, Strength 1.0

### Black Widow Cosplay
```
photo of a Black Widow wearing a T-shirt and tight pants, kneeling on the floor in the living room, leaning forward with her hands on the floor between her knees, sticking tongue out, text on a shirt says "feed me", seductive look, big breasts, dynamic pose, raven hair
```
Settings: Steps 20, CFG 1, Flux Realistic, Distilled CFG 3.5, 896x1152, Strength 0.5

## Recommended LoRA Combinations
- **Cum On Face FLUX** (1.0) - cum facial effects
- **Braces FLUX** (1.0) - teeth braces
- **HandBra** (1.0) - hand covering breasts
- **Saggy Flux Tits** (0.6) - breast shape
- **Beauty Enhancer + Realistic eyes** (1.0) - face enhancement
- **Perfect Full Round Breasts & Slim Waist** (1.0) - body shape

## Best Checkpoints
- flux_dev
- flux1-dev
- FLUX Dev

## Notes
- Also available for SDXL
- Works well with character LoRAs (use lower strength 0.4-0.5)
- For subtle effect use 0.4-0.5 strength
- For prominent tongue use 0.9-1.2 strength
- Combine with facial expression or ahegao concepts
- Add "tongue piercing" for pierced tongue

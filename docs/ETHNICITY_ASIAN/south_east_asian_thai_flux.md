# South East Asian & Thai Women - Flux

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 335 |
| **👍** | 29 |
| **Tips** | 50 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `South_East_Asian_Thai_Flux.safetensors` |
| **Original filename** | `seasianmilf.safetensors` |
| **Civitai** | https://civitai.com/models/1278761 |
| **Trigger word** | None (descriptive prompts) |
| **Strength** | 1.0 (strongest) or lower for mix |
| **Type** | CHARACTER / Ethnicity |

## Description

LoRA for South East Asian Thai/Filipina type women. Created because FLUX tends to default to tall Korean model types. Trained on 48 images over 10 epochs (2400 steps).

### Key Features
- Short, petite body proportions (fights FLUX's "tall Korean" default)
- Age range: 20-50 (younger and mature)
- Dark complexion options
- Thai/Filipina facial features
- Can also aid in generating Latina women
- ~30% full body shots in training data

### Tips
- Use `level camera angle` to avoid amateurish top-down angles in full body shots
- Full strength (1.0) for strongest results
- Lower values for mixing with other LoRAs
- Limited NSFW capability (not enough training data to bypass censorship)

## Sample Prompts

**Prompt 1 (Full body - Thai street):**
```
cinematic photo A photo realistic <full body portrait from head to toe:1> of a short petite young aged 25 year old Thai woman with dark complexion and natural long straight hair, she has a natural short petite body, she has a delicate feminine Asian face with a small weak jaw, high cheek bones and narrow upper face, she has beautiful wide smile, with big teeth, she has big pouty lips, she she wears a pretty summer top and mini skirt, she wears high heel strappy sandals, she is standing in busy street in Thailand and her full body is in the frame, <lora:South_East_Asian_Thai_Flux:1> <lora:maripol-flux1-dev-v1:0.7> . 35mm photograph, film, bokeh, professional, 4k, highly detailed
```
Settings: Steps 28, CFG 1, Euler, 768x1024, Distilled CFG 3.5
Negative: drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly

**Prompt 2 (Portrait - serious expression):**
```
cinematic photo A photo realistic full body portrait of a petite young aged 25 year old Thai woman with dark complexion and natural long straight hair, she has a delicate feminine Asian face with a small weak jaw, high cheek bones and narrow upper face, she hold a serious expression with her mouth closed, she has big pouty lips, she she wears a pretty summer top <lora:South_East_Asian_Thai_Flux:1> <lora:maripol-flux1-dev-v1:0.7> . 35mm photograph, film, bokeh, professional, 4k, highly detailed
```
Settings: Steps 24, CFG 1, Euler, 768x1024, Distilled CFG 3.5

**Prompt 3 (Comic book style - Ning character):**
```
[Primary Outfit Colour: Black]
[Hair Colour: Black]
[Panthose Colour: Sheer Black]
[Accessory Colour: Black]
[Lip & Nail Colour: Deep Red]
comic book style illustration
Full-body front view of Ning rendered in high-detail ink and colour, with bold linework and stylized shading.
She has strong black eyeliner, and (T-Shirt Colour) eyeshadow.
Her straight, silky smooth waist-length (Hair Colour) hair is parted in the centre.
On her feet: Black Dr Marten Boots.
The backdrop reveals a cosy living room.

Stylized texture detailing evokes painterly depth within a clean inked aesthetic. TPI, Seasianmilf, aidmafluxpro1.1
```
Settings: Steps 30, CFG 1, Size 832x1216, Clip skip 2

**Prompt 4 (Goth Asian - dark seduction):**
```
The image is a portrait of a young goth asian woman with dark hair styled as a ponytail, showing airbj gesture: Hand next to mouth in a fist like form, cheek bulge on the opposite side of her fist, mouth slightly opened. She is standing in front of a stone wall and appears to be in a dimly lit room. The woman is wearing a black top with lace sleeves and a purple corset underneath. She has a black choker necklace around her neck and is looking directly at the camera with a seductive expression. Dark seduction. Her makeup is dramatic, with dark lipstick and dark eyeliner. The overall mood of the image is dark and edgy. Dim warm lights illuminate the scene, creating a play of shadows across the picture.
```
Settings: Steps 32, CFG 2.1, Size 832x1216, Clip skip 2

## Keywords

- `Thai woman`
- `Filipina`
- `South East Asian`
- `dark complexion`
- `short petite`
- `petite body`
- `delicate feminine Asian face`
- `small weak jaw`
- `high cheek bones`
- `narrow upper face`
- `big pouty lips`
- `long straight hair`
- `level camera angle` (for full body)

## Facial Features

Trained to produce:
- Small, weak jaw
- High cheekbones
- Narrow upper face
- Big pouty lips
- Delicate feminine features
- Natural dark complexion options

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 24-32 |
| **CFG** | 1-2.1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 768x1024 / 832x1216 |
| **Strength** | 1.0 (full) or lower for mix |
| **Clip skip** | 2 |

## Recommended Combinations

**With Maripol (80s Polaroid style):**
```
<lora:South_East_Asian_Thai_Flux:1>
<lora:maripol-flux1-dev-v1:0.7>
```

**With FLUX Pro Detailer:**
```
<lora:South_East_Asian_Thai_Flux:1>
<lora:aidmafluxpro1.1:1>
```

**With character LoRAs:**
```
<lora:South_East_Asian_Thai_Flux:1>
<lora:[Character_LoRA]:1>
```

## Notes

- Created to counter FLUX's tendency to generate "tall Korean model" types
- Works for 20-50 age range (young and mature)
- ~30% full body training - use `level camera angle` to avoid top-down angles
- Can help generate Latina women as well
- Limited NSFW capability (minimal explicit training data)
- Use at 1.0 for strongest ethnicity effect, lower for mixing


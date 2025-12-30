# 12. Characters - Other

[← Back to Index](INDEX.md)

## Overview

Other character and style LoRAs for consistent character generation. This section contains versatile character generation tools that work across multiple styles and ethnicities, providing powerful options for creating diverse and realistic characters.

---

## Table of Contents

- [Mystic XXX](#mystic-xxx)
- [Amateur Flux](#amateur-flux)
- [Caira - Sci-Fi Character](#caira---sci-fi-character)

---

## Mystic XXX

| Parameter | Value |
|-----------|-------|
| **Files** | `MysticXXX-v6.safetensors`, `MysticXXX-v4.safetensors` |
| **Civitai** | https://civitai.com/models/1295758/nsfw-fluxorwan-22orqwen-mystic-xxx |
| **Trigger word** | None |
| **Strength** | 0.4-1.0 |
| **Type** | NSFW / Style / Character Generation |
| **Compatibility** | FLUX, WAN 2.2, Qwen |

### Available versions
- **v6** (`MysticXXX-v6.safetensors`) - latest version, recommended
- **v4** (`MysticXXX-v4.safetensors`) - older version, useful for workflow character generation

### Description
Powerful LoRA for enhancing NSFW image generation. Works with various characters and ethnicities. v4 is particularly useful when generating consistent characters in workflows.

### Keywords
- `mysticxxx`
- `nsfw`
- `photorealistic`
- `detailed skin`

---

## Amateur Flux

| Parameter | Value |
|-----------|-------|
| **File** | `amateur.safetensors` |
| **Civitai** | https://civitai.com/models/683226/amateur-flux?modelVersionId=764714 |
| **Trigger word** | `amateurlora` |
| **Strength** | 0.2-1.0 |
| **Type** | Style / Photography |

### Description
**IMPORTANT LoRA** - particularly useful when generating characters.

LoRA for generating amateur-style photos - candid, home-made, Reddit r/gonewild aesthetic. Adds authentic, non-professional look: compression artifacts, natural lighting, spontaneous poses. Works great with other LoRAs (desiespresso, MysticXXX, nipples).

### Sample prompts

**Prompt 1 (Reddit gonewild bathroom):**
```
nude photo of girl, uploaded to reddit r/gonewild, she is a slut, Full body shot photo of a young bottomless Caucasian woman standing in a bright bathroom, atmospheric vibe; the lighting is high, sterile environment. The woman has fair skin with natural textures such as subtle pores, and her braided wavy blonde hair and a cute face, adding a soft glow where the light touches. She wears a red off-shoulder t-shirt that contrasts with the dark surroundings. she is bottomless, which gives her image a sexy, hot style. She is only wearing a short t-shirt (revealing her perfect shaved pussy, revealing her bald innie pussy:1.5). She is holding her iphone taking a photo in the mirror. <lora:amateur:1>
```

### Keywords
- `amateurlora`
- `amateur`, `candid`
- `uploaded to reddit`, `r/gonewild`
- `snapchat`, `flickr`, `instagram`
- `compression artefacts`

---

## Caira - Sci-Fi Character

| Parameter | Value |
|-----------|-------|
| **File** | `Caira_Flux_13_09_25.safetensors` |
| **Civitai** | https://civitai.com/models/1953612/cairaflux130925 |
| **Trigger word** | `Caira` (optional, character activates at strength) |
| **Strength** | 0.4-1.2 |
| **Type** | Character / Sci-Fi |

### Description
Virtual character "Caira" - a fictional woman created for the science fiction novel "In the Shadow of the Sun." She is NOT a real person.

**Character traits:**
- Young woman, ~18-20 years old
- Blonde/white hair (long, thick, slightly curly)
- Blue/green eyes with long thick eyelashes
- Slim athletic ballerina figure
- Hourglass figure with long slender legs
- Medium to large natural breasts

### Sample prompts

**Prompt 1 (Prison cell cuffed):**
```
cinematic film still <lora:Caira_Flux_13_09_25:1.2> Caira, 20 years old young woman, blonde, blue eyes, kneeling on the floor in a prison cell, wrists tied to the floor, looking up, making eye contact, one tear, pleading, open mouth, close-up, top view, breasts, perfect body, beautiful woman, supermodel, sexy, ultra high quality, 8k <lora:Cuffed:1> cuffs, handcuffs, chain, slave. shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy
```

**Prompt 2 (Prison lingerie with stockings):**
```
cinematic film still <lora:Caira_Flux_13_09_25:1.2> A photorealistic profile angle photograph of a woman in lingerie standing in a doorway, looking over her shoulder with a confident expression. The woman is wearing a black bra and black thigh-high stockings with suspenders. She appears to be in her early twenties, with long blonde hair tied in a ponytail. She is slim and is wearing black lingerie, including a bra and panties. The background is a prison cell, the wall is a dirty, concrete floor. <lora:ffstockings8_DEV:0.8> ffstockings, fully fashioned stockings. shallow depth of field, vignette, highly detailed, bokeh, cinemascope, moody, film grain
```

**Prompt 3 (Beach sunset topless):**
```
cinematic photo 1 woman, solo, Young 20-year-old blonde, very young woman from the northern province, with completely white hair, perfect blue eyes, long, thick eyelashes, thick and slightly curly hair, thick round lips, ballerina, dancer, great ballerina figure, hourglass figure, medium- to large-sized, slightly sagging natural breasts, slim body, long, slender legs. Very natural skin texture: pores, veins, and hair with some natural imperfections. A photo-realistic shoot from the front about a young woman standing on a beach during sunset, wearing only denim shorts. she is standing with her arms at her sides, looking directly at the viewer with a smile on her face. 35mm photograph, film, bokeh, professional, 4k, highly detailed
```

**Prompt 4 (Urban lace underwear):**
```
cinematic photo 1 woman, solo, Young 20-year-old blonde, very young woman, with completely white hair, perfect blue eyes, long, thick eyelashes, medium- to large-sized natural breasts, slim body, hourglass figure, long, slender legs. Very natural skin texture: pores, veins with some natural imperfections. A photorealistic close-up of a young woman posing elegantly in an urban setting. She is wearing a sheer, beige lace bra and matching panties. The background is a busy city street with people walking and buildings in the distance. The lighting is soft and natural. 35mm photograph, film, bokeh, professional, 4k
```

**Prompt 5 (Bar scene dark jacket):**
```
cinematic photo 1 woman, solo, <lora:Caira_Flux_13_09_25:1>, Young 20-year-old blonde, very young woman, with completely white hair, perfect blue eyes, long, thick eyelashes, medium to large breasts, slim body, hourglass figure, long, slender legs. Very natural skin texture. A photo-realistic portrait shoot about a woman with long, wavy blonde hair wearing a brown lace bra and a dark jacket, standing in a dimly lit bar with shelves of bottles in the background. she is wearing a dark gray jacket over a sheer lace bra that reveals her cleavage, and a choker necklace. The lighting is soft and warm. 35mm photograph, film, bokeh, professional, 4k, highly detailed
```

**Prompt 6 (Ballet dancer Hegre style):**
```
cinematic photo 1 girl, solo, full-length photo, <lora:Caira_Flux_13_09_25:0.4>, girl-woman, 18 years old, very young woman, from the northern province, (completely white hair:1.6), thick and curly hair, (perfect blue eyes:1.2), thick long eyelashes, medium-small slightly sagging natural breasts, pronounced nipples, slim athletic figure, ballerina figure, slim long legs, (ballerina, ballet, slim). Photographed by Peter Hegre, creating a sensual and intimate atmosphere, emphasizing refined elegance. Inspired by Alphonse Maria Mucha, Hegre-Art inspired fashion conceptual photography. Highly detailed cinematic portrait of a young ballet dancer, dynamic pose, showcasing ballet footwork, detailed pointe shoes, dramatic lighting, volumetric lighting, 8k, hd, photorealistic
```

### Keywords
- `Caira` - character name
- `blonde` / `white hair` / `blonde-white hair`
- `blue eyes` / `green eyes`
- `ballerina` / `dancer` / `ballerina figure`
- `hourglass figure`
- `slim athletic figure`
- `long slender legs`
- `medium to large breasts`
- `young 20-year-old`

### Character description template
```
Young 20-year-old blonde, very young woman from the northern province, with completely white hair, perfect blue eyes, long, thick eyelashes, thick and slightly curly hair, thick round lips, ballerina, dancer, great ballerina figure, hourglass figure, medium- to large-sized, slightly sagging natural breasts, slim body, hourglass figure, and long, slender legs. Very natural skin texture: pores, veins, and hair with some natural imperfections.
```

### Tested combinations

**Combination 1 (Handcuffs):**
```
<lora:Caira_Flux_13_09_25:1.2>
<lora:Cuffed:1>
```

**Combination 2 (Fully fashioned stockings):**
```
<lora:Caira_Flux_13_09_25:1.2>
<lora:ffstockings8_DEV:0.8>
```

**Combination 3 (Doll version blend):**
```
Checkpoint: FasciumSRPO v2.0
<lora:Caira_Flux_13_09_25:0.5>
<lora:Caira-Doll:0.5>
```

**Combination 4 (Ballet with second version):**
```
<lora:Caira_Flux_13_09_25:0.4>
<lora:Caira_Flux_31_08_25:0.4>
```

**Combination 5 (RealSkin enhancement):**
```
Checkpoint: CyberRealistic Flux v1.5
<lora:Caira_Flux_13_09_25:1>
<lora:Cairo_SD15:0.8>
<lora:RealSkin_xxXL:2.5>
```

### Recommended checkpoints
- FLUX Dev
- FasciumSRPO v2.0
- Flux1-DedistilledMixTuned v3.0
- CenKreChro v1.0 FP8
- CyberRealistic Flux v1.5

### Notes
- Fictional character from sci-fi novel "In the Shadow of the Sun"
- Lower strength (0.4-0.5) when combining with other character LoRAs
- Higher strength (1.0-1.2) for strong character likeness
- Works well with cinematic film still style
- Compatible with stockings, handcuffs, and other clothing LoRAs
- Ballerina/dancer prompts work particularly well

---

[← Back to Index](INDEX.md)

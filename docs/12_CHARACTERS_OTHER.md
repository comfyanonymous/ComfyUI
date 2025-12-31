# 12. Characters - Other

[← Back to Index](INDEX.md)

## Overview

Other character and style LoRAs for consistent character generation. This section contains versatile character generation tools that work across multiple styles and ethnicities, providing powerful options for creating diverse and realistic characters.

---

## Table of Contents

- [Mystic XXX](#mystic-xxx)
- [Amateur Flux](#amateur-flux)
- [Caira - Sci-Fi Character](#caira---sci-fi-character)
- [Normal European Woman (EBL)](#normal-european-woman-ebl)
- [Isabella Flux CFH](#isabella-flux-cfh)
- [Brunette](#brunette)
- [Brunette Bombshell](#brunette-bombshell)
- [Ona Instagram Model](#ona-instagram-model)

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

## Normal European Woman (EBL)

| Parameter | Value |
|-----------|-------|
| **File** | `EBL.safetensors` |
| **Civitai** | https://civitai.com/models/2203426/normal-european-woman |
| **Trigger word** | `EBL` |
| **Strength** | 1.0 |
| **Type** | Character |
| **Version** | v1.1 |

### Description
Character LoRA for generating a consistent "normal yet sensual" European woman. Trained with various face generated images until obtaining face consistency. Useful for modeling, influencer-style images, and general character work.

### Character traits
- Normal European appearance
- Sensual/attractive features
- Consistent face across generations
- Versatile for various scenarios

### Sample prompts

**Prompt 1 (Simple sexy posing):**
```
<lora:EBL:1> EBL, sexy posing, black underwear
```

**Prompt 2 (Cowgirl scene - NSFW):**
```
<lora:EBL:1> EBL. A photograph of a nude woman with long hair, with a black thong, engaged in cowgirl anal sex with a muscular man. She is looking back at the camera, with her hands on her buttocks, and her small breasts visible, squat on top of the man. The man's penis is inside her anus. The scene is set in a messy garage
```

### Keywords
- `EBL` - **REQUIRED** trigger word
- `sexy posing`
- `modeling`
- `influencer`
- `European woman`

### Tested combinations

**Combination 1 (SRPO + Cowgirl):**
```
<lora:EBL:1>
<lora:srpo_256_base_oficial_model_fp16_lora:1>
<lora:flux_anal_cowgirl:1>
```

**Combination 2 (Bare skin):**
```
<lora:EBL:1>
<lora:FLUX_secrets_bare_skin_sevenof9:1>
```

### Use cases
- Modeling/fashion photography
- Influencer-style content
- Sensual/NSFW scenes
- Consistent character across multiple images

### Notes
- Always use trigger word `EBL` in prompt
- Works with various NSFW LoRAs
- Good face consistency across generations
- Versatile for many scenarios and styles

---

## Isabella Flux CFH

| Parameter | Value |
|-----------|-------|
| **File** | `Isabella_Flux_CFH-000011.safetensors` |
| **Civitai** | https://civitai.com/models/937908/isabella-flux-cfh |
| **Trigger word** | None (character activates at strength) |
| **Strength** | 0.8-1.0 |
| **Type** | Character |
| **Version** | V1 |

### Description
A beautiful young woman from the Flux CFH (Character Family Hub) series. Born from many sources from the internet as well as AI-generated faces. Part of a consistent character family that works well with other CFH LoRAs.

**Disclaimer:** Entirely fictional AI-generated character. Any resemblance to real persons is coincidental.

### Character traits
- Beautiful young woman
- Striking blue eyes
- Variable hair color (works with blonde, black, brunette)
- Slender, toned figure
- Versatile for fashion, lingerie, and sensual scenes

### Sample prompts

**Prompt 1 (Bohemian style):**
```
young woman, full-body woman with a rude appearance, midriff, long mesy blond hair, in bohemian style clothing, eyes blue.
```

**Prompt 2 (Sheer green nightgown with Sexy Nighty):**
```
A sheer, light green S3xYN1gHtY flows delicately over the young woman's slender frame, the translucent fabric catching the light with an ethereal glow. The soft pastel hue enhances the depth of her striking blue eyes, complementing the gentle curves of her silhouette. Her long, sleek black hair cascades in loose waves over her shoulders, a few wisps framing her face as she tilts her head slightly. One hand grazes the thin straps of S3xYN1gHtY, adjusting them absentmindedly, while the other rests lightly against her thigh.
```

**Prompt 3 (Leather harness with Sheer Clothes):**
```
A photograph of a slender woman with long blonde hair, wearing a black leather harness with straps and metal rings that exposes her breasts, standing confidently with an inviting emotion. She has fair skin and is wearing black leather wrist cuffs. The scenery is a minimalist room with a modern design. The lighting is soft and warm, highlighting her figure.
```

**Prompt 4 (Candlelit bedroom):**
```
A sheer, light green S3xYN1gHtY flows delicately over the young woman's slender frame, the translucent fabric catching the flickering glow of candlelight with an ethereal shimmer. The soft pastel hue enhances the depth of her striking blue eyes, reflecting the warm golden tones dancing around the dimly lit room. Her long, sleek black hair cascades in loose waves over her shoulders, a few wisps framing her face as she tilts her head slightly. Dozens of candles illuminate the bedroom, their soft glow casting a warm, intimate aura.
```

**Prompt 5 (Purple nightgown):**
```
S3xYN1gHtY, A stunning young woman in her early 20s stands gracefully, her pose effortlessly elegant, accentuated by S3xYN1gHtY. The rich purple S3xYN1gHtY contrasts beautifully against her smooth skin, its delicate fabric flowing over her toned frame with an air of refined sensuality. Her long, sleek black hair cascades down her back, framing her striking blue eyes.
```

**Prompt 6 (Black nightgown on satin bed):**
```
A high-resolution photograph of a breathtaking blonde woman with long, wavy golden hair, sitting on the edge of a plush, satin-covered bed in a well-lit bedroom. She wears a classic black S3xYN1gHtY that clings to her curves, the sheer fabric revealing just enough to captivate. Her legs are crossed, her posture relaxed yet undeniably seductive. She leans slightly forward, resting her arms on her thighs, drawing attention to the way the fabric contours her form. Her intense gaze meets the viewer's, exuding confidence and irresistible allure.
```

### Keywords
- `young woman`
- `striking blue eyes`
- `slender frame` / `toned figure`
- `long hair` (blonde, black, brunette)
- `elegant` / `graceful`
- `sensual` / `seductive`

### Tested combinations

**Combination 1 (Sexy Nighty):**
```
<lora:Isabella_Flux_CFH:0.8>
<lora:Sexy_Nighty_CFH:1.2>
```

**Combination 2 (Sheer Clothes + Small Breasts):**
```
<lora:Isabella_Flux_CFH:0.8>
<lora:FLUX_sexy_clothes_v3_Sevenof9:1>
<lora:Breast_size_slider_Small_breasts:-1.45>
```

**Combination 3 (Long Sleeve Tied Up):**
```
<lora:Isabella_Flux_CFH:1>
<lora:Long_Sleeve_Tied_Up_CFH:0.8>
```

### CFH Family compatibility
Part of the CFH (Character Family Hub) series, works well with:
- Sexy Nighty CFH
- Long Sleeve Tied Up CFH
- Other CFH clothing LoRAs

### Notes
- No specific trigger word needed - character activates at strength
- Use strength 0.8 for subtle, 1.0 for strong character likeness
- Works great with various hair colors
- Excellent for lingerie and fashion photography
- Part of CFH ecosystem for consistent character+clothing combinations
- Fictional AI-generated character

---

## Brunette

| Parameter | Value |
|-----------|-------|
| **File** | `Brunette.safetensors` |
| **Original filename** | `Brunette-000003.safetensors` |
| **Civitai** | https://civitai.com/models/994929/brunette?modelVersionId=1114805 |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | Character |
| **Version** | V1 |

### Description
Young woman, model girl character LoRA. Created from 50 photos of many unknown women. Not a real person. Generates consistent brunette character type.

### Sample prompts

**Prompt 1 (Selfie by window):**
```
amateur smartphone selfie, overexposure, Low-resolution photo, selfie shot on a mobile phone. young woman, 18 yo (european:1.6) girl, brunette, brown eyes, small breasts, skinny, thin, petite brunette, brown eyes, long hair, thin legs, skinny ass, dark eyeliner, natural lip color, wearing black tight mini dress, High heels. One hand shirtpull, revealing. Girl smiles standing on by window posing seductive, looking at viewer, night. skin texture style, realism, detailed <lora:Brunette:1>
```

**Prompt 2 (Lace outfit bedroom):**
```
revealing her small to medium-sized breasts. The lace fabric adds a touch of elegance and sophistication to the outfit., both of which accentuate her slender figure and small breasts., and is smiling warmly at the camera. She is wearing a tight, holding a smartphone with a colorful case, with a few loose strands falling over her face. She is wearing a black, minimalist room with a white wall and a large abstract painting featuring earthy tones and abstract patterns in the background. The painting is framed in a black frame and hangs on the wall above the sofa., minimalist bedroom. <lora:Brunette:1>
```

### Keywords
- `brunette`
- `young woman`
- `european girl`
- `petite brunette`
- `brown eyes`
- `model girl`

### Tested combinations

**Combination 1 (Petite selfie style):**
```
<lora:Brunette:1>
<lora:Boreal-FD:0.7>
<lora:thin_skinny_legs_ass_flux-gmr:1>
<lora:Realistic_Photos_Detailed_Skin:0.7>
```

### Notes
- Works well for generating consistent brunette characters
- Fictional character - not a real person
- Combine with Boreal-FD for realistic photography style
- Works great with Thin Legs Skinny Ass for petite body type
- Add Realistic Photos LoRA for enhanced skin textures

---

## Brunette Bombshell

| Parameter | Value |
|-----------|-------|
| **File** | `Brunette_Bombshell_by_Sarcastic_TOFU.safetensors` |
| **Civitai** | https://civitai.com/models/1082146/brunettebombshellbysarcastictofu?modelVersionId=1215066 |
| **Trigger word** | `Brunette_Bombshell` |
| **Strength** | 0.5-0.6 |
| **Type** | Character / Style |
| **Compatibility** | FLUX |

### Description
Quality LoRA for Flux trained on numerous quality SFW & mostly NSFW images of beautiful brunette women. Can create very high quality AI image outputs for brunette characters. Works for both safe and NSFW content.

### Sample prompts

**Prompt 1 (Bedroom nude):**
```
A Raw Hires photograph of a fair-skinned stark naked white 20 year old French Brunette woman with shoulder-length, wavy brown hair, She has a warm, friendly smile and is looking at viewer with her beautiful Green eyes. She is posing with her right hand raised and placing behind her head. She is lying on a white bed. She has small to medium breasts, a slim physique with breathtaking hourglass curves, and is posing with her legs spread apart, revealing her (shaved pussy) and vulva. The background shows a white, quilted bedspread and a minimalist bedroom setting. Brunette_Bombshell <lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.5>
```

**Prompt 2 (Jacuzzi scene):**
```
Photograph of a nude young woman with light skin and long, wavy brown hair tied in a ponytail, lounging in a modern white Jacuzzi tub. She has small breasts, a slim physique, and is positioned with one leg raised, partially submerged in water. The background features a large window showing a blurred outdoor scene with leafless trees. The tub has multiple jets visible, and the setting is bright and clean, ((Dripping Water Droplets can be seen on her skin and wet hair)). emphasizing the woman's relaxed and confident expression. Brunette_Bombshell, Fine_Wet_Woman, BubbleBath_Nudes <lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.5>
```

**Prompt 3 (Beach wet):**
```
A high-resolution photograph of a 20-year-old French Brunette woman. She has a warm, friendly smile and is looking at viewer with her beautiful Green eyes. She has a light skin tone, Brunette hair, and a slender, ((petite physique with hourglass curves)). She is standing on a beach and tying up her hair. Her skin and hair are dripping wet. Brunette_Bombshell <lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.6>
```

### Keywords
- `Brunette_Bombshell` - **trigger word**
- `brunette`
- `French Brunette woman`
- `wavy brown hair`
- `slim physique`
- `hourglass curves`

### Tested combinations

**Combination 1 (Bedroom with anatomy):**
```
<lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.5>
<lora:NippleDiffusion:0.5>
<lora:PussyDiffusion:0.5>
```

**Combination 2 (Jacuzzi/wet scene):**
```
<lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.5>
<lora:BubbleBath_Nudes:0.5>
<lora:Fine_Wet_Woman:1>
```

**Combination 3 (Beach/nudist):**
```
<lora:Brunette_Bombshell_by_Sarcastic_TOFU:0.6>
<lora:Nudist_Beach_Flux:1>
<lora:Boreal-FD:0.5>
<lora:Boudoir_Nudity_Style:0.3>
```

### Notes
- Use trigger word `Brunette_Bombshell` for best results
- Works at lower strength (0.5-0.6) for subtle effect
- Author: Sarcastic_TOFU
- Part of Sarcastic_TOFU's character series

---

## Ona Instagram Model

| Parameter | Value |
|-----------|-------|
| **File** | `Ona_model_Lora_Flux1-dev.safetensors` |
| **Civitai** | https://civitai.com/models/1873363/ona-instagram-model-virtual-character-lora-ig?modelVersionId=2120399 |
| **Trigger word** | `Ona_model` |
| **Strength** | 0.55 |
| **Type** | Character / Instagram Model |
| **Compatibility** | FLUX Dev |

### Description
Stunning and versatile virtual Instagram model character. Trained on a wide variety of photos to bring a realistic and expressive model with a distinct look. Ona can adapt to many scenes and styles - from casual outdoor adventures to high-fashion studio shots, lifestyle photos, fashion editorials, and travel scenes.

### Character traits
- 25 year old appearance
- Tanned/sun-kissed skin
- Black thick hair, long straight
- Hourglass body, athletic
- Big lips, big eyes, black eyes
- Supermodel body type
- Natural breasts

### Key features
- Versatile poses & expressions (thoughtful gaze to joyful smile)
- Adaptable style (swimwear, casual, street clothes, adventure gear)
- High-quality results with realistic skin texture and hair
- Works well for Instagram/Snapchat selfie style

### Sample prompts

**Prompt 1 (Bathroom mirror selfie NSFW):**
```
Perfect Eyes, perspective, score_9, score_8_up, score_7_up, masterpiece, high quality, realistic, detailed, (photorealistic:1.4), navel focus, Natural skin), 1girl, 25year old, slut, gorgeous girl, (cute girl), tanned, makeup, long haircut straight hair, ((black thick hair:1.7)), suntan skin, (hourglass body), natural breast, athletic girl, big lips, soft lips, big mouth, nipples, supermodel body, slutty face, big eyes, black eyes, gorgeous perfect face, big natural breasts, (((((naked:1.2, breasts revealed, nipples revealed)))) ((((A Snapchat selfie with pink iPhone in the mirror of a beautiful woman with tanned skin taking a selfie while masturbating in the bathroom, fingering vagina, pussy revealed)))). The overall mood is playful and theatrical, with a focus on performance art. The lighting emphasizes the performer, creating a dynamic and engaging composition. masterwork, masterpiece, best quality, detailed, depth of field, high detail, best quality, very aesthetic, 8k, dynamic pose, depth of field, dynamic angle, instagirl, Ona_model, RNAT <lora:Ona_model_Lora_Flux1-dev:0.55>
```

**Prompt 2 (Beach topless):**
```
Photograph of Ona_model standing on a sunlit beach. Ona_model wears a black thong. Ona_model is topless. She is turned slightly to the side, showcasing her toned buttocks. Her arms are covering her breasts. The background is a bright, sunlit sky with a blurred ocean. Ona_model's skin is sun-kissed, and she has a bracelet on her right wrist. The sunlight creates a halo effect around her head. <lora:Ona_model_Lora_Flux1-dev:0.55>
```

**Prompt 3 (Mirror selfie with cellphone):**
```
Perfect Eyes, perspective, score_9, score_8_up, score_7_up, masterpiece, high quality, realistic, detailed, (photorealistic:1.4), navel focus, Natural skin), 1girl, 25year old, slut, gorgeous girl, (cute girl), tanned, makeup, long haircut straight hair, ((black thick hair:1.7)), suntan skin, (hourglass body), natural breast, athletic girl, big lips, soft lips, big mouth, nipples, supermodel body, slutty face, big eyes, black eyes, gorgeous perfect face, big natural breasts, (((((naked:1.2, breasts revealed, nipples revealed)))) ((((A Snapchat selfie with pink iPhone in the mirror of a beautiful woman with tanned skin taking a selfie while masturbating in the bathroom, fingering vagina, pussy revealed)))). The overall mood is playful and theatrical, with a focus on performance art. masterwork, masterpiece, best quality, detailed, depth of field, high detail, best quality, very aesthetic, 8k, dynamic pose, depth of field, dynamic angle, instagirl, Ona_model, RNAT, Mirror selfie, holding cellphone <lora:Ona_model_Lora_Flux1-dev:0.55>
```

### Keywords
- `Ona_model` - **TRIGGER WORD**
- `instagirl`
- `tanned`, `suntan skin`
- `hourglass body`
- `athletic girl`
- `supermodel body`
- `black thick hair`

### Tested combinations

**Combination 1 (Full NSFW Instagram setup):**
```
Checkpoint: Jib Mix Flux
<lora:Ona_model_Lora_Flux1-dev:0.55>
<lora:FLUX_Gone_Wild:1>
<lora:Instagram_Girl:1>
<lora:Real_Nipples_Areola_Textures:1>
<lora:Real_Mirror_Selfie_FLUX:1>
```

**Combination 2 (Simpler NSFW):**
```
<lora:Ona_model_Lora_Flux1-dev:0.55>
<lora:FLUX_Gone_Wild:1>
<lora:Instagram_Girl:1>
```

### Compatible checkpoints
- FLUX Dev
- Jib Mix Flux

### Notes
- Use trigger word `Ona_model` in prompt
- Recommended strength 0.55
- Works great with Instagram/Snapchat selfie style LoRAs
- Combine with FLUX Gone Wild for NSFW content
- Combine with RNAT for realistic nipples
- Virtual character - not a real person

---

[← Back to Index](INDEX.md)

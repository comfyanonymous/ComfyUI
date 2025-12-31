# 13. Body Types & Shape

[← Back to Index](INDEX.md)

This section covers LoRAs specifically designed to control and enhance body types and shapes. These LoRAs allow you to generate different body proportions, from hourglass figures to specific anatomical features, providing diverse and realistic body representation in your generated images.

## Table of Contents

- [Hourglass V2](#hourglass-v2)
- [Flux Bodies Female](#flux-bodies-female)
- [Thin Legs Skinny Ass GMR](#thin-legs-skinny-ass-gmr)
- [Perfect Big Round Ass](#perfect-big-round-ass)
- [FLUX Female Anatomy](#flux-female-anatomy)
- [Breast Size Slider / Small Breasts](#breast-size-slider--small-breasts)

---

## Hourglass V2

| Parameter | Value |
|-----------|-------|
| **File** | `hourglassv2_flux.safetensors` |
| **Civitai** | https://civitai.com/models/129130?modelVersionId=932199 |
| **Trigger word** | None |
| **Strength** | 0.7-1.0 |
| **Type** | Body Shape |

### Description
LoRA for generating hourglass body shape - wide hips, narrow waist, feminine curves.

### Keywords
- `hourglass body shape`
- `wide hips`
- `small waist`
- `curvy`
- `perfect body`

---

## Flux Bodies Female

| Parameter | Value |
|-----------|-------|
| **File** | `collbdy.safetensors` |
| **Civitai** | https://civitai.com/models/776651/flux-bodies-female |
| **Trigger word** | None (use `<lora:collbdy:1>`) |
| **Strength** | 1.0 |
| **Type** | Body Shape / Diversity |
| **Compatibility** | FLUX |

### Description
Versatile LoRA that celebrates the diversity of the female body, capturing various shapes and sizes with stunning accuracy. Crafted to represent a wide range of body types, allowing creators to explore and depict realistic, diverse forms. Ideal for inclusivity in designs or enhancing the natural beauty of different body types.

### Key features
- Wide range of body types
- Realistic diverse forms
- Flexibility for various body shapes
- Natural beauty enhancement

### Sample prompts

**Prompt 1 (Bear skin rug):**
```
image of a woman with long wavy hair, background blue and red smoke, topless, she is standing on a bear skin rug, she is wearing a fluffy white skirt, light from window is shining down on her, eye level shot, smiles, looking at the viewer, <lora:collbdy:1>
```

### Notes
- Works well for depicting body diversity
- Combine with other anatomy LoRAs for enhanced results

---

## Thin Legs Skinny Ass GMR

| Parameter | Value |
|-----------|-------|
| **File** | `thin_skinny_legs_ass_flux-gmr.safetensors` |
| **Original filename** | `thin skinny legs ass flux-gmr.safetensors` |
| **Civitai** | https://civitai.com/models/730817/thin-legs-skinny-ass-qwen-krea-flux-d-xl-gmr |
| **Trigger word** | None |
| **Strength** | 0.8-1.0 |
| **Type** | Body Shape |
| **Compatibility** | QWEN Image, Flux D, XL |

### Description
LoRA for generating thin legs and skinny butt. Works with QWEN Image, Flux D and XL. Creates petite, slim body proportions.

### Sample prompts

**Prompt 1 (Street winter):**
```
amateur selfie, overexposure, Low-resolution photo, shot on a mobile phone. young woman, 18 yo (european:1.6) girl, blonde, grey eyes, small breasts, skinny, thin, petite blonde, grey eyes, long hair, ponytail, dark eyeliner, natural lip color, perky nipples, wearing white see-through white shirt, grey tight miniskirt, nylon pantyhose, High heels, jacket. Girl standing seductive, smiles. The background shows crowded street, winter, daylight. <lora:thin_skinny_legs_ass_flux-gmr:0.9>
```

**Prompt 2 (Window night):**
```
amateur smartphone selfie, overexposure, Low-resolution photo, selfie shot on a mobile phone. young woman, 18 yo (european:1.6) girl, brunette, brown eyes, small breasts, skinny, thin, petite brunette, brown eyes, long hair, thin legs, skinny ass, dark eyeliner, natural lip color, wearing black tight mini dress, High heels. One hand shirtpull, revealing. Girl smiles standing on by window posing seductive, looking at viewer, night. skin texture style, realism, detailed <lora:thin_skinny_legs_ass_flux-gmr:1>
```

**Prompt 3 (Mall selfie):**
```
amateur selfie, overexposure, Low-resolution photo, selfie shot on a mobile phone. young woman, 18 yo (european:1.6) girl, brunette, Brown eyes, small breasts, skinny, thin, petite brunette, brown eyes, long dark hair, ponytail, dark eyeliner, natural lip color, perky nipples, wearing darkred leather chocker, oversized sweater, mini skirt, brown tights, sneakers, puffer jacket. Girl smiles standing seductive, looking at viewer. The background shows crowded mall <lora:thin_skinny_legs_ass_flux-gmr:0.85>
```

**Prompt 4 (Bedroom morning):**
```
amateur smartphone selfie, overexposure, Low-resolution photo, selfie shot on a mobile phone. young woman, 18 yo (european:1.6) girl, blonde, grey eyes, small breasts, skinny, thin, petite blonde, grey eyes, long hair, glasses, dark eyeliner, natural lip color, wearing black leather collar, sheer top, pokies, string panties. Girl pouts smiles sitting seductive on bed, looking at viewer. Bedroom, sunny morning <lora:thin_skinny_legs_ass_flux-gmr:0.85>
```

### Keywords
- `skinny`
- `thin`
- `thin legs`
- `skinny ass`
- `petite`
- `slim`

### Tested combinations
- Character LoRAs (Anitka, Brunette)
- See-through/sheer clothing LoRA
- Boreal-FD (Boring Reality) LoRA
- Selfie POV LoRA
- Nipple Pokies LoRA
- FLUX Pout LoRA

### Notes
- Works well with amateur/smartphone selfie style
- Combine with character LoRAs at 0.8-0.9 strength
- Good for petite European girl look

---

## Perfect Big Round Ass

| Parameter | Value |
|-----------|-------|
| **File** | `Perfect_Big_Round_Ass.safetensors` |
| **Civitai** | https://civitai.com/models/958789/perfect-big-round-ass |
| **Trigger word** | None (use keywords) |
| **Strength** | 0.7-1.0 (NOT above 1!) |
| **Type** | Anatomy / Body Shape |

### Description
LoRA significantly improving generated buttocks quality - round, firm, defined. **IMPORTANT:** Don't use strength above 1.0!

### Keywords
- `round buttocks`
- `firm buttocks`
- `perfect ass`

---

## FLUX Female Anatomy

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_Female_Anatomy.safetensors` |
| **Original filename** | `FLUX Female Anatomy.safetensors` |
| **Civitai** | https://civitai.com/models/678412/flux-female-anatomy |
| **Trigger word** | None |
| **Strength** | 0.6-1.0 |
| **Type** | Anatomy / Body Shape / NSFW |

### Description
LoRA for enhancing and improving female anatomy generation in FLUX models. Provides better anatomical accuracy and detail for realistic body representation.

### Sample prompts

**Prompt 1 (Yoga studio):**
```
A beautiful woman practicing yoga in a bright studio, nude, full body visible, natural lighting, realistic skin texture, detailed anatomy, professional photography
```

**Prompt 2 (Bedroom morning):**
```
Young woman stretching in bed, morning light through window, nude, relaxed pose, soft bedding, realistic body proportions, intimate atmosphere
```

**Prompt 3 (Beach sunset):**
```
Woman walking on beach at sunset, nude, back view, natural body, golden hour lighting, waves in background, candid moment
```

**Prompt 4 (Art studio model):**
```
Figure model posing in art studio, nude, classical pose, natural daylight, artistic setting, easels in background, realistic anatomy
```

**Prompt 5 (Bathroom mirror):**
```
Woman looking at herself in bathroom mirror, nude, soft lighting, steam from shower, realistic reflections, natural body
```

**Prompt 6 (Pool scene):**
```
Woman emerging from swimming pool, nude, water droplets on skin, outdoor setting, summer day, natural curves, realistic wet skin
```

**Prompt 7 (Bedroom window):**
```
Woman standing by bedroom window, nude, morning light, thoughtful expression, natural body proportions, soft shadows
```

**Prompt 8 (Outdoor shower):**
```
Woman in outdoor shower, tropical setting, nude, water streaming down body, natural surroundings, realistic wet skin texture
```

**Prompt 9 (Dressing room):**
```
Woman in dressing room, nude, trying on jewelry, full length mirror, warm lighting, realistic body, candid moment
```

**Prompt 10 (Garden):**
```
Woman in private garden, nude, surrounded by flowers, natural daylight, peaceful expression, realistic anatomy, artistic composition
```

**Prompt 11 (Spa setting):**
```
Woman relaxing in spa, nude, lying on massage table, soft towels, warm ambient lighting, peaceful atmosphere, realistic body proportions
```

### Keywords
- `nude`
- `natural body`
- `realistic anatomy`
- `full body`
- `natural curves`
- `realistic skin texture`
- `body proportions`

### Recommended combinations
- `<lora:flux_realism_lora:0.7>` - Enhanced realism
- `<lora:detail_enhancer_flux_v1:0.6>` - Additional detail
- `<lora:Flux_Skin_Detailer:0.7>` - Skin texture improvement

### Notes
- Works well with various poses and settings
- Combine with realism LoRAs for best results
- Suitable for artistic nude photography style

---

## Breast Size Slider / Small Breasts

| Parameter | Value |
|-----------|-------|
| **File** | `Small_breasts_v2.safetensors` |
| **Original filename** | `Small breasts_v2.safetensors` |
| **Civitai** | https://civitai.com/models/663135/breast-size-slidersmall-breasts-flux |
| **Trigger word** | None |
| **Strength** | -0.4 to 1.4 (slider) |
| **Type** | Body Slider / Breast Size |
| **Version** | v2.0 |

### Description
Breast size slider LoRA for FLUX. Created for generating small breasts but works as a slider for different sizes. Positive strength = smaller breasts, negative strength = bigger breasts.

### How to use (v2.0)
- **Positive strength (0.2 to 1.4)** = Smaller breasts
- **Negative strength (-0.4 to 0)** = Bigger breasts
- Range: -0.4 to 1.4

**Note:** v3.0+ has reversed logic (positive = bigger)

### Known issues
- Changes lightness/darkness of image
- Sometimes generates blurry images

### Sample prompts

**Prompt 1 (Ballet dancer):**
```
Realistic, deep focus style, urb, realistic breast, small breasts, ballet, Professional even lighting, sharp lens, sharp focus, A fit 42 year old European woman, full body shot, Woman with a side shave and a colorful undercut, dressed in a crop top, fishnet sleeves, and cargo pants, leaning against a classic muscle car in a parking lot, chatting and laughing with a group of friends as they plan their next adventure. realism, chiaroscuro, cinematic quality, rays of light, play of shadow and light <lora:Small_breasts_v2:1>
```

**Prompt 2 (Toilet selfie):**
```
score_9, score_8_up, score_7_up, score_6_up, masterpiece, best quality, realistic, realism, amateur, analogue photo, cellphone selfie of 1girl, youthful 18 yo (european:1.6) girl, small breasts, perky breast, skinny, pale skin, Tiny, petite body, brunette, ponytail, grey eyes, perfect face, beautyfull face, no makeup, wearing sheer strap top, stringi thongs, glasses, kneels on wet floor tiles, public toilet, looking up at viewer, seductive pose, (open mouth, tongue out), skin texture style, sharp detailed edges, shot from above <lora:Small_breasts_v2:0.8>
```

**Prompt 3 (Hong Kong rooftop):**
```
A beautiful 25-year-old Hong Kong student with black hair, long straight hair, striking brown eyes, soft freckles, bold eyebrows, and glasses for a nerdy appearance. She is in a modern, stylish and somewhat slutty outfit, and is captured in an upper body camera shot, striking a flirty pose as she looks towards the viewer. She lifts her crop top with her right hand, flashing one of her breasts because she's horny and in love. She has very small breasts with hard nipples. A high-rise rooftop overlooking a city skyline during early evening, after sunset, a slight breeze moving the hair, and the distant glow of Hong Kong city lights starting to come alive. shirtlift, Real Nipples and Areola Textures, RNAT <lora:Small_breasts_v2:0.3>
```

**Prompt 4 (Japanese nudist beach):**
```
RAW photo, Young and pretty 20y Japanese woman, (looking viewer), This photograph captures a woman sunbathing on a sandy beach, with the ocean waves gently crashing in the background. She is seated on a colorful beach towel with a blue and yellow pattern. Her legs are spread apart, and her pubic area is exposed, indicating she is completely nude. A Japanese woman is sitting on a beach towel on the sand, relaxing, and is startled to see a middle-aged Japanese man walking by, so she covers her chest with one hand. (The woman's embarrassed expression:1.5), She is Asian, has very small breasts, and is skinny and thin <lora:Small_breasts_v2:0.8> (flat_chest:1.5)
```

**Prompt 5 (Leather mini-skirt):**
```
18yo woman, long hair, skinny, naked topless, droplet-shaped_breasts, long legs, low cut leather mini-skirt, black pantyhose, high heels, sexy pose, full body portrait, Photorealistic, film noir, analog style, soft lighting, subsurface scattering, heavy shadow, stone wall background, open leather jacket <lora:Small_breasts_v2:0.2>
```

### Keywords
- `small breasts`
- `perky breast`
- `flat chest`
- `tiny breasts`
- `petite body`
- `skinny`

### Tested combinations
- Petite body type for FLUX LoRA
- Phlux (Photorealism) LoRA
- Deep Focus style LoRA
- NSFW FLUX LoRA
- BreastShaper LoRA
- Nudist Beach Flux LoRA
- Real Nipples and Areola Textures (RNAT)

### Strength examples
| Strength | Effect |
|----------|--------|
| 1.0 | Very small breasts |
| 0.6-0.8 | Small breasts |
| 0.2-0.3 | Slightly smaller |
| -0.2 | Slightly bigger |
| -0.4 | Bigger breasts |

### Notes
- Works as a slider - adjust strength for size
- v2.0: Positive = smaller, Negative = bigger
- Combine with petite body LoRAs for consistent results
- May affect image lighting - adjust as needed

---

[← Back to Index](INDEX.md)

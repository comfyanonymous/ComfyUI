# 13. Body Types & Shape

[← Back to Index](INDEX.md)

This section covers LoRAs specifically designed to control and enhance body types and shapes. These LoRAs allow you to generate different body proportions, from hourglass figures to specific anatomical features, providing diverse and realistic body representation in your generated images.

## Table of Contents

- [Hourglass V2](#hourglass-v2)
- [Flux Bodies Female](#flux-bodies-female)
- [Thin Legs Skinny Ass GMR](#thin-legs-skinny-ass-gmr)
- [Perfect Big Round Ass](#perfect-big-round-ass)
- [FLUX Female Anatomy](#flux-female-anatomy)

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
| **File** | `Skinny_Legs_Ass_GMR.safetensors` |
| **Civitai** | https://civitai.com/models/730817/thin-legs-skinny-ass-gmr |
| **Trigger word** | `rsla` |
| **Strength** | 0.7-1.0 |
| **Type** | Body Shape |

### Description
LoRA for generating thin legs and skinny butt. Works with QWEN Image, Flux D and XL.

### Keywords
- `rsla` - trigger word
- `skinny legs`
- `thin legs`
- `skinny ass`

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

[← Back to Index](INDEX.md)

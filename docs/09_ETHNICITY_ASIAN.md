# 9. Ethnicity - Asian

[← Back to Index](INDEX.md)

---

## Description

LoRAs for generating Asian characters - East Asian (Chinese, Japanese, Korean) and South Asian (Indian, Pakistani).

---

## Table of Contents

1. [Desi Espresso v2](#desi-espresso-v2)
2. [NSFW Indian Women](#nsfw-indian-women)
3. [Flux Asian Beauty](#flux-asian-beauty)
4. [Korean Gone Flux](#korean-gone-flux)

---

## Desi Espresso v2

| Parameter | Value |
|-----------|-------|
| **File** | `desiespresso_flux_v2.safetensors` |
| **Civitai** | https://civitai.com/models/990802/desi-espresso-lora-for-indian-south-asian-faces-flux-1d |
| **Trigger word** | `desiespresso` (v2), `d351 d4rk` (v3) |
| **Strength** | 0.6-1.0 |
| **Type** | Character / Ethnicity |

### Description
LoRA for generating Indian/South Asian faces with various skin tones - from fair to very dark. Ideal for realistic portraits and photos of models with Asian heritage.

### Recommended settings
- **Sampler:** Euler
- **Scheduler:** Beta
- **Steps:** 12-32
- **CFG:** 1
- **LoRA strength:** 0.6-1.0

### Sample prompts

**Prompt 1 (Dark skin beauty):**
```
desiespresso, a dark-skinned Indian woman stands under soft, natural light, wearing a yellow chiffon saree draped gracefully over one shoulder, her hair in loose waves, large gold hoop earrings catching the light, in a traditional haveli courtyard with intricate marble carvings and arched doorways, warm sunset light casting golden reflections on her glowing skin
```

**Prompt 2 (Traditional saree):**
```
desiespresso, extremely high resolution photo, a stunning 25-year-old woman wearing red slik saree, natural makeup, soft lighting, wearing a simple blouse underneath, dark skin, professional photoshoot, studio lighting
```

**Prompt 3 (Bridal look):**
```
desiespresso, a gorgeous indian bride wearing pink lehnga choli at her wedding, gold jewellery, dark skin, intricate henna on hands, professional wedding photography, bokeh background
```

**Prompt 4 (Casual modern):**
```
desiespresso, closeup portrait, indian woman with fair skin, wearing modern western clothing, jeans and crop top, outdoor cafe setting, natural daylight
```

**Prompt 5 (Diverse skin tones):**
```
desiespresso, portrait of indian woman with dusky complexion, traditional nose ring, bindhi on forehead, minimal makeup, soft studio lighting
```

**Prompt 6 (NSFW - nude portrait):**
```
desiespresso, artistic nude portrait of indian woman, dark skin, soft natural lighting, tasteful pose, studio setting, professional photography
```

### Keywords
- `desiespresso` (trigger word)
- `dark-skinned Indian woman`
- `fair skin` / `dusky complexion` / `dark skin`
- `saree`, `lehnga`, `salwar kameez`
- `gold jewellery`, `henna`
- `haveli`, `traditional setting`

### Combinations with other LoRAs
```
<lora:desiespresso_flux_v2:0.8>
<lora:detail_enhancer_flux_v1:0.7>
<lora:flux_realism_lora:0.6>
```

---

## NSFW Indian Women

| Parameter | Value |
|-----------|-------|
| **File** | `NSFW_Indian_Women_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/1376958/nsfw-indian-women-flux-lora |
| **Trigger word** | `woman` |
| **Strength** | 0.6 |
| **Type** | Character / Ethnicity / NSFW |

### Description
LoRA for generating Indian women. Trained with many nude images - model is adapted for NSFW.

### Sample prompts

**Prompt 1 (Kneeling nude):**
```
1 stunning girl kneeling naked with her knees apart exposing her vagina, hair dark wine color. perfect face framed by soft strands. Ear piercings and delicate lace choker adds to her irresistible charm., no blush, <lora:NSFW_Indian_Women_Flux:0.6>
```

**Prompt 2 (Topless jeans):**
```
Woman, Full-body shot of a model wearing no top and blue jeans, standing in a bedroom with her hands on her hips, striking a confident pose with one leg slightly in front of the other, highlighting her entire body.
```

### Keywords
- `woman` - trigger word
- `Indian`
- `naked`
- `exposing`
- `stunning girl`

---

## Flux Asian Beauty

| Parameter | Value |
|-----------|-------|
| **File** | `Flux_Asian_Beauty.safetensors` |
| **Original filename** | `lora-asian-beauty-flux.safetensors` |
| **Civitai** | https://civitai.com/models/799377/flux-asian-beauty |
| **Trigger word** | `asian beauty` |
| **Strength** | 1.0 |
| **Type** | Character / Ethnicity |

### Description
LoRA for generating beautiful East Asian women with captivating features. Works well for portrait and fashion shots.

### Sample prompts

**Prompt 1 (Blue dress):**
```
asian beauty, a woman with brown hair wearing a blue dress and gold earrings, standing in front of a wall. She is wearing a pair of blue eyeshadow, giving her a beautiful and captivating look. <lora:Flux_Asian_Beauty:1>
```

**Prompt 2 (Luxury bedroom NSFW):**
```
Asian beauty 18yo wearing only a black thong, topless, exposed breasts, standing in front of a luxury bedroom, gold and red room, while her porcelain skin glows softly beneath the warm, golden lighting. Her gentle smile hints at secrets shared among the crimson-hued bed sheets and billowy drapes that shroud the room in mystery. small nipples, aidmafluxpro1.1,
```

**Prompt 3 (Victorian gown):**
```
A sumptuous tableau unfolds: Asian beauty resplendent in a ravishing black Victorian gown, reclines upon plush velvet cushions. Raven-black tresses cascade down her back like a waterfall of night, while her porcelain skin glows softly beneath the warm, golden lighting. Her gentle smile hints at secrets shared among the crimson-hued bed sheets and billowy drapes that shroud the room in mystery.
```

**Prompt 4 (Yacht jumpsuit):**
```
homewear simple jumpsuit, facing viewer, looking at viewer, This is a photo of an Asian woman wearing a simple jumpsuit, standing on a yacht, <lora:Flux simple jumpsuit-10:1>, <lora:Flux_Asian_Beauty:1>,
```

**Prompt 5 (Seductive big breasts):**
```
1 girl, score_9, score_8_up, score_7_up, source_Genuine, masterpiece, Anatomically Correct, beautiful face, realistic, Big Breasts, shiny skin, perfect face, very detailed and beautiful face and eyes, attractive face, detailed face, heart brave, Kind Face, detailed skin, a beautiful woman, seductive pose, sexy, sweat, seductive smile, purple eyes, long hair, low light, bed room, clothes, sweat, black shirt, long sleeve, full body, sitting, Breast Squeeze, breasts squeezed together, arms down, elbow squeeze, from below angle, close up in the chest, chest visible, yellow hair, blonde, shot from neck to waist, very detailed, perfect figure, torso is bare, armpit, erect nipple, loose top, no bra, bending over, loose bra, lace, hanging breast, hidden nipples, sweaty skin, side view, side angle, breast sucking, faceless male, stare each other
```

**Prompt 6 (Countryside topless):**
```
score_9, score_8_up, score_7_up, (masterpiece, best quality, ultra-detailed), very aesthetic, soft lighting, bright tone, realism, masterpiece, best quality, amazing quality, (messy hairs:0.8), parted lips, pink, beautiful lips, beautiful eyes, (cameltoe:0.8), long black hairs, sit with m shape leg, seperated leg, one hand over the head, sit on concrete material, (candid photography), (((not look at viewer))), blurred background, country side background, river and banana field, hard nipples, horny, topless, <lora:Flux_Asian_Beauty:1>
```

**Negative prompt for Prompt 6:**
```
score_6, score_5, score_4, (worst quality:1.2), (low quality:1.2), (normal quality:1.2), lowres, bad anatomy, bad hands, signature, watermarks, ugly, imperfect eyes, skewed eyes, unnatural face, unnatural body, error, extra limb, missing limbs, teeth, shoe, gloves, animal, male, man, penis
```

**Prompt 7 (City backdrop):**
```
asian beauty, a woman in a white shirt posing for a picture in front of a backdrop of buildings, trees, and lights. The sky is visible at the top of the image. <lora:Flux_Asian_Beauty:1>
```

### Recommended combinations
- `<lora:Realism_Lora_By_Stable_yogi_SDXL8.1:1>` - Realism
- `<lora:Sinfully_Stylish_.02_for_FLUX:1.4>` - Style
- `<lora:zy_AmateurStyle_v2:1>` - Amateur style
- `<lora:aidmafluxpro1.1:1>` - Pro enhancement

### Keywords
- `asian beauty` - trigger word
- `captivating`
- `beautiful`
- `porcelain skin`
- `golden lighting`

---

## Korean Gone Flux

| Parameter | Value |
|-----------|-------|
| **File** | `korean_gone_flux.safetensors` |
| **Civitai** | https://civitai.com/models/677337/korean-gone-flux?modelVersionId=758214 |
| **Trigger word** | None |
| **Strength** | 0.5-0.55 |
| **Type** | Character / Ethnicity / Photorealistic |

### Description
LoRA for generating photorealistic Korean women. Works well for both SFW and NSFW content with cinematic, film-like quality.

### Sample prompts

**Prompt 1 (Professional nude photography):**
```
Professional Nude Photography, Korean model, Nudes, smiles, blushing, dimples, most beautiful woman on earth, beautiful detailed nipples, waist up, soft tender belly, toned belly, navel, wide hips, gorgeous face, br34sts, gorgeous young cute lovely adorable woman, medium beautiful perky breasts and nipples. <lora:flux_korean:0.55> <lora:Small_Nipples_-_FLUX:0.50> <lora:reclining-nude/v03:0.6>
```

**Prompt 2 (Post-coital bedroom):**
```
Realistic nudity, tired girlfriend, after sex, confused smile, look of wonderment, on bed, nsfw, full body, long toned legs, Korean woman, Korean girl, slightly bedraggled, natural skin tone, sweaty, sleepy, messy sheets, post-coital glow, glistening skin, 30 year old nude Korean, seductive, breasts visible, lover, realistic art style, looking at the viewer, pretty breasts, photorealistic, very aesthetic, movie still, polaroid photo quality, Fuji film photography, a stunning photo of a weary female, she looks like a real-life, RAW candid cinema, 16mm, color graded portra 400 film, remarkable color, ultra realistic, textured skin, remarkably detailed pupils, dark circles under her eyes, realistic skin noise, visible skin detail, skin fuzz, dry skin, shot with a cinematic camera, detailed skin texture, (blush:0.2), (goosebumps:0.6), subsurface scattering, beautiful photograph with style, asian woman, big brown eyes, plump lips, gorgeous, 8k HD, detailed skin texture, ultra realistic, textured skin, analog raw photo, cinematic grain, whimsical, nsfw, photography, capturing moments, creative composition, Cinematic portrait photography, capture subject in a way that resembles a still frame from a movie, cinematic lighting, narrative quality, drawing viewers into the scene and evoking a sense of cinematic immersion, capturing emotion, professional, engaging, compelling, slender toned build, jet black hair, light warm skin tone, minimal makeup
```

**Prompt 3 (Rooftop fashion):**
```
Flash portrait shot portrait of a young woman of Korean descent standing in Rooftop Terraces (with permission from property owners), with delicate fair skin and visible skin texture, wearing a Lime Leggings, and her Asymmetrical Bob hair, which makes her look vibrant and avant-garde., The woman's gaze is firm and direct., The overall composition combines fashion and natural elements to create a striking and powerful visual effect.,<lora:filmfotos:0.2>,<lora:Korean-Fashion:0.5>,<lora:flux_korean:0.5>,
```

**Prompt 4 (Seoul back alley):**
```
Beautiful Korean woman with radiant skin and wavy, long black hair. She is wearing low rise black leather pants, white Converse all star sneakers, a white crop top revealing a toned midriff. She has a fiery expression and her hands on her hips. She is standing in a Seoul back alley. Night time. Raining, puddles reflecting the lights of neon signs.
```

### Keywords
- `Korean woman` / `Korean girl` / `Korean model`
- `photorealistic`
- `realistic`
- `cinematic`
- `film photography`
- `Fuji film` / `portra 400`
- `textured skin`
- `detailed skin texture`

### Style characteristics
- Film-like, cinematic quality
- Detailed skin textures (pores, fuzz, goosebumps)
- Natural skin tones
- Works well with film photography LoRAs
- Good for both fashion and intimate scenes

### Recommended combinations
- `<lora:filmfotos:0.2>` - Film photography style
- `<lora:Korean-Fashion:0.5>` - Korean fashion style
- `<lora:Small_Nipples_-_FLUX:0.50>` - Anatomy detail

---

[← Back to Index](INDEX.md)

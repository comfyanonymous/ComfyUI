# 14. Style & Enhancement

[← Back to Index](INDEX.md)

## Overview

This section covers LoRAs specifically designed for improving image quality, enhancing details, and providing overall visual enhancement to your generations. These LoRAs focus on boosting textures, lighting, facial details, anatomy fixes (particularly hands), and unlocking NSFW capabilities. They are essential tools for achieving professional-quality outputs and can be combined with other LoRAs for optimal results.

## Table of Contents

- [Detail Enhancer FLUX](#detail-enhancer-flux) - Critical detail and texture booster
- [Detailed Perfection Style](#detailed-perfection-style) - Full body detail enhancement
- [Hands XL (Hand v2)](#hands-xl-hand-v2) - Hand anatomy fix and improvement
- [FluxHands Final Bonus](#fluxhands-final-bonus) - Hand fix for FLUX
- [Flux Realistic Hands](#flux-realistic-hands) - Hand fix with trigger word
- [Hand Detail FLUX & XL](#hand-detail-flux--xl) - Hand detail enhancement
- [MoreFace LoRA](#moreface-lora) - Enhanced facial detail for FLUX
- [FLUXTASTIC V3](#fluxtastic-v3) - NSFW unlock and enhancement
- [NSFW Master FLUX](#nsfw-master-flux) - NSFW unlock (recommended 0.8)
- [Professional Nude Photography V3](#professional-nude-photography-v3) - High quality nude photography
- [SECRET SAUCE HERO V2.1](#secret-sauce-hero-v21) - High-quality style enhancement (5600 images)
- [SuperDemidov Style](#superdemidov-style) - Russian photographer Egor Demidov style
- [Flux Improved Female Nudity V2](#flux-improved-female-nudity-v2) - NSFW unlock (4100 images, 53h training)
- [FluxUnchained LoRA](#fluxunchained-lora) - NSFW unlock extracted from FluxUnchained model
- [FLUX Cum on Face](#flux-cum-on-face) - Facial/cum effect without face alteration
- [Game of Cum V2](#game-of-cum-v2) - Best cum effect LoRA (face, breasts, body)
- [Realistic People Photograph FLUX](#realistic-people-photograph-flux) - Hyper-realistic professional photo style

---

## Detail Enhancer FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `detail_enhancer_flux_v1.safetensors` |
| **Civitai** | https://civitai.com/models/651351/detail-enhancer-flux |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 |
| **Importance** | ⭐⭐⭐ CRITICAL - use in workflows |

### Description
LoRA for enhancing details, textures and lighting. **No trigger word required.**

### Usage
Add to any prompt as a detail booster:
```
<lora:detail_enhancer_flux_v1:0.7>
```

---

## Detailed Perfection Style

| Parameter | Value |
|-----------|-------|
| **File** | `perfection style v1.safetensors` |
| **Civitai** | https://civitai.com/models/411088/detailed-perfection-style |
| **Trigger word** | `perfection style` |
| **Strength** | 0.6-1.0 |
| **Type** | Enhancement |

### Description
LoRA for improving full body details - hands, feet, face, body. Works with XL, F1D, SD1.5, Pony, Illu.

### Keywords
- `perfection style`
- `RNAT` (Real Nipples and Areola Textures)

---

## Hands XL (Hand v2)

| Parameter | Value |
|-----------|-------|
| **File** | `Hand v2.safetensors` |
| **Civitai** | https://civitai.com/models/hands-xl |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 |
| **Type** | Enhancement / Anatomy Fix |
| **Compatibility** | XL, SD1.5, FLUX.1-dev, Pony, Illustrious, Zit |

### Description
LoRA for fixing and improving hand generation. Trained on multiple base models for broad compatibility. Significantly improves hand anatomy and reduces common deformities like extra fingers.

### Sample prompts

**Prompt 1 (Military cosplay):**
```
realism, realistic girl, selfie, standing, in room, in military costume cosplay, cat ears, microphone, unloading, body armor
```

**Prompt 2 (Bathroom selfie):**
```
Clear sharp realitic photo of a cute woman in her early twenties with big eyes with long long pink voluminous wavy hair combed to one side, one side of the hair is thicker than the other side and wearing very skin-tight floral button short shirt with no bra that reveals her belly button and a pink skirt, taking a selfie in a bathroom mirror. She is standing in front of a glass door, holding a black smartphone in her right hand and her left hand is by her hips, smiling at the camera. The woman is slim and has large breasts with visible cleavage. She has a confident expression. <lora:Hand v2:0.5>
```

**Prompt 3 (Luxury portrait):**
```
A luxurious, high-resolution digital portrait featuring a woman with striking, symmetrical facial features and porcelain-like skin. Her almond-shaped eyes are highlighted with subtle, earth-tone makeup, drawing attention to her radiant gaze. Her long, flowing auburn hair cascades in soft waves, framing her face elegantly. She poses gracefully, her fingers lightly touching her chin, exuding confidence and poise. <lora:Hand v2:0.5>
```

### Keywords
- `hands`
- `fingers`
- `detailed hands`
- `perfect hands`

### Notes
- Also check the related Feet model from the same author
- Gun/weapon focus version also available

---

## FluxHands Final Bonus

| Parameter | Value |
|-----------|-------|
| **File** | `FluxHands_Final_Bonus.safetensors` |
| **Original filename** | `lora-000016.TA_trained.safetensors` |
| **Civitai** | https://civitai.com/models/805324/fluxhands-final-bonus |
| **Trigger word** | None (use keywords) |
| **Strength** | 0.5-1.0 |
| **Type** | Enhancement / Hand Fix |
| **Compatibility** | FLUX |

### Description
LoRA for improving hand generation in FLUX models. Helps with hand anatomy, finger count, and hand positioning. Note: The LoRA may occasionally confuse left/right hands when specified.

### Known limitations
- Sometimes confuses 'right hand' with 'left hand' and vice versa
- May require multiple generations for perfect results

### Sample prompts

**Prompt 1 (Nurse):**
```
A sexy female nurse standing next to bed in a hospital room. IV drip. Her is holding her nurse scrubs outfit, shirt open revealing her breasts. Her expression is calm and professional. The room is clean and sterile, with medical equipment surrounding the bed and soft, fluorescent lighting overhead. 21yo sexy babe. exposed breasts, Real Nipples and Areola Textures, RNAT, topless, exposed breasts, hourglass body shape, female hand, back side, right hand, left hand, pants pull down,
```

**Prompt 2 (Simple hand focus):**
```
solo, female hand, right hand, back side
```

**Negative prompt (recommended):**
```
cartoon, anime, deformed iris, deformed pupils, cgi, 3d, render, sketch, cartoon, drawing, anime, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad hands
```

**Prompt 3 (Irish redhead):**
```
hyper realistic photo lifelike Irish redhead woman with large breasts, incredibly detailed, bokeh, ultra realistic, 8k, incredibly cute Irish girl, lace bra, covering her breast, shy, hands over breasts outdoor, highlands
```

**Prompt 4 (Game show):**
```
a stunning European woman with shoulder-length black hair, wearing an elegant black evening dress, smiling directly at the camera, standing behind a modern chrome side table, on the table is a classic red TV game show red game show button TV quiz show button glossy red dome push button large red emergency button with a glossy dome and chrome base, she is holding a red arrow-shaped sign that says "BUZZ ME", the arrow is pointing toward the buzzer, studio lighting, shallow depth of field, photorealistic, fashion photography style, high-resolution, Canon EOS R5 quality
```

### Keywords
- `female hand`
- `right hand` / `left hand`
- `back side`
- `hands over breasts`
- `holding`

### Notes
- Use negative prompt for better anatomy
- Combine with other anatomy LoRAs for best results
- Good for scenes requiring hand interaction with objects

---

## Flux Realistic Hands

| Parameter | Value |
|-----------|-------|
| **File** | `Flux_Realistic_Hands.safetensors` |
| **Original filename** | `fitzka.safetensors` |
| **Civitai** | https://civitai.com/models/1232423/flux-realistic-hands |
| **Trigger word** | `fitzka` |
| **Strength** | 0.5-0.7 |
| **Type** | Enhancement / Hand Fix |
| **Compatibility** | FLUX |

### Description
LoRA for improving hand realism in FLUX models. Uses trigger word `fitzka` in prompts. Works well combined with other LoRAs for complex scenes.

### Sample prompts

**Prompt 1 (Flying broomstick):**
```
fitzka, oversize bra nipple, flying broomstick, a asian woman, messy hair flying left, wear low cut skimpy tulle dress, blue bow headband, smirk, leaning away standing on a long broomstick in middle of the air, aerial view of tokyo bay area view, blue sky (view from above:0.2), <lora:Flux_Skinny_Thinspo_Petite:0.3> <lora:MysticXXX-V6:0.65> <lora:Flux_Realistic_Hands:0.6>
```

**Prompt 2 (Judo gym):**
```
YTKstyle, rotated view camera, fitzka, pov front upperbody photo of one skinny pale asian woman, brown messy side updo tail hair with bang, angry face, blush, wear long sleeve judo suit with long white pants, brown belt, puffy areolas, long nipple, on blue and orange judo mat of bright big empty judo gym, bend over, fighting pose, standing one bare foot with leg open, show nipple, hand reach under camera, punching viewer, face down, low angle, chest focus, close up, <lora:Flux_Realistic_Hands:0.7>
```

**Prompt 3 (Victorian dominatrix):**
```
A detailed high resolution cinematic photograph. kpsX, fitzka, aidmafluxpro1.1, aidmarealisticskin, hairy armpits, asymmetrical face, natural small breasts, analog photography, bent over desk, skin pores visible, dominant facial expression, muscular physique, soft natural lighting, red stockings, silk high heels, choker jewelry, mature adult woman, aging skin with wrinkles, 8k resolution, professional photography, slutty makeup style, Victorian era silk dress, full-body centered composition, solo subject, tall and skinny stature, beautiful eyes, ankle boots, small erect nipples, dominatrix, ((from behind and from side)), angry, gorgeous forty years old finnish woman, in subway, <lora:Flux_Realistic_Hands:0.5>
```

### Keywords
- `fitzka` - **REQUIRED** trigger word
- `hand reach`
- `punching viewer`
- `holding`
- `realistic hands`

### Recommended combinations
- `<lora:MysticXXX-V6:0.65>` - NSFW enhancement
- `<lora:aidmaRealisticSkin:0.05>` - Skin detail
- `<lora:aidmaFLUXPro1.1:0.05>` - Pro enhancement

### Notes
- Always include `fitzka` trigger word in prompt
- Lower strength (0.5-0.7) works well
- Combines effectively with other LoRAs

---

## Hand Detail FLUX & XL

| Parameter | Value |
|-----------|-------|
| **File** | `Hand_Detail_FLUX_XL.safetensors` |
| **Original filename** | `Detailed_Hands-000001.safetensors` |
| **Civitai** | https://civitai.com/models/260852/hand-detail-flux-and-xl?modelVersionId=1003317 |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 |
| **Type** | Enhancement / Hand Detail |
| **Compatibility** | FLUX, SDXL |

### Description
LoRA trained to add more details to hands. Improves hand anatomy and texture. V2 is a prior epoch in case V1 is overtrained for some base models.

### Sample prompts

**Prompt 1 (Luxury portrait):**
```
A highly detailed ultrarealistic ((full body)) wide-angle photo of voluptuous platinum blonde haired naked sexy 18 years old young woman. pale natural skin. Her long hair, flowing waves cascade down her back, framing her slender hourglass figure. She wears a stunning, see through satin white robe, featuring a deep leg slit and open front, ((topless)) exposing her large ample naked chest and curves, perfect huge droplet-shaped breasts, exposing her small pink nipples. Topless, bottomless, shaved pussy, pink pussy, parted lips, perfect_pussy, She exudes sensuality. Draped over her shoulder, a luxurious white feather boa, revealing her shoulder. Her glossy light pink lips shimmer in the sunlight. her piercing blue eyes, highlighted by sharp eyeliner, captivate anyone who looks her way. Her (perfect hands) and nails are painted light pink french nails. She is adjusting her hair with the right hand. Her forearm is adorned with a jeweled bracelet chain, adding an exotic, luxurious detail. The background shows a highly detailed living room of a beautiful modern house <lora:Hand_Detail_FLUX_XL:1.0>
```

**Prompt 2 (Simple):**
```
1girl
```

**Prompt 3 (Gym fitness):**
```
perfect lighting, ultra realistic, 8k resolution, ultra-detailed, thin Asian fitness model, professional photography, one knee on a bench doing single arm rows with a dumbbell in a crowded gym on Venice Beach, people in the background, smiling, small bright pink sports bra, very tight blue leggings high on her waist creating a slight camel toe, thigh gap, beautiful city vista, fit, medium breasts, Asian, freckles on face, looking at the viewer, sunny, brunette, skinny, pale skin, playful, realistic face, Sporty ponytail hair style with highlights
```

### Keywords
- `perfect hands`
- `detailed hands`
- `hand detail`
- `french nails`
- `adjusting hair with hand`

### Recommended combinations
```
<lora:BreastShaper_splendid_droplets_Flux:0.8>
<lora:aidmaFLUXPro1.1:0.4>
<lora:SameFace_Fix:-0.7>
<lora:SkinDetails_flux_lora:0.4>
<lora:Hand v2:1.0>
<lora:Hand_Detail_FLUX_XL:1.0>
```

### Notes
- Can be combined with Hand v2 LoRA for enhanced results
- Works well with skin detail and realism LoRAs
- Use with SameFace_Fix for face variety

---

## MoreFace LoRA

| Parameter | Value |
|-----------|-------|
| **File** | `morefaceV2-lora.safetensors` |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | Enhancement / Face Detail |
| **Compatibility** | FLUX.1-dev |
| **Civitai** | https://civitai.com/models/866492/moreface-lora |

### Description
This LoRA adds more face detail to FLUX.1 Dev generations. Works best with the creator's Hyper 8 steps Flux.1 Dev checkpoint.

### Recommended settings
- **Strength:** 1.0
- Works well combined with Hyper 8 steps Flux.1 Dev checkpoint

### Sample prompts

**Prompt 1 (Blonde in cafeteria):**
```
a gorgeous woman with long light-blonde hair wearing a low cut tanktop, Short hair, split-color hair, dressed in a printed maxi skirt with a crop top and stacked heels, looking away, cafeteria, detailed masterpiece most beautiful artwork in the world Ultrarealistic, Kodak UltraMax 400, dramatic lighting, <lora:morefaceV2-lora:1>
```

**Prompt 2 (Woman on stairs):**
```
a woman descending stairs wearing a very sexy dress with thigh highs cleavage long legs slim waist looking at viewer standing up, Very long wave hair, Dark gray hair, dressed in a feathered top with leather pants and a wide-brimmed hat, Landscape, amidst the shifting sands of eternity, (highly detailed masterpiece Realistic extremely hyper aesthetic trending on artstation), Kodak Ektachrome E100, split lighting, <lora:morefaceV2-lora:1>
```

**Prompt 3 (Fitness girl in Iceland):**
```
a fitness girl, Short curly pixie cut, Brunette, dressed in a high-neck mesh top with a leather mini skirt and knee-high boots, modelshoot style shot, dramatic landscapes of Iceland are like something out of a fairy tale, with waterfalls, glaciers, and volcanic terrain, (masterpiece best quality ultra-detailed best shadow amazing realistic picture), Fujichrome Provia 100F, Starlight, <lora:morefaceV2-lora:1>
```

**Prompt 4 (African woman):**
```
a african woman with sexy pose, Cloudy hair, Light brown hair, dressed in a vintage-inspired high-waisted bikini with a halter top and cat-eye sunglasses, two shot angle, agarta subterranean world, (high quality awardwinning masterwork 4k highly detailed), Sony A9 II, Light and shadow plays, <lora:morefaceV2-lora:1>
```

**Prompt 5 (School setting):**
```
waning light, (highly detailed masterpiece Realistic extremely hyper aesthetic trending on artstation), School, dressed in a off-the-shoulder sweatshirt with leggings and sneakers, Lightest blond hair, Quiff haircut, a Gorgeous woman, <lora:morefaceV2-lora:1>
```

### Notes
- Best results with Hyper 8 steps Flux.1 Dev checkpoint
- Combines well with style and anatomy LoRAs
- Use diverse prompts with detailed clothing and lighting descriptions

---

## FLUXTASTIC V3

| Parameter | Value |
|-----------|-------|
| **File** | `FLUXTASTIC_V3.safetensors` |
| **Civitai** | https://civitai.com/models/646167/fluxtastic-v3-boobs-and-more |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | NSFW Unlock / Enhancement |
| **Compatibility** | FLUX |

### Description
NSFW unlock LoRA trained on 400+ hand-reworked real images. Removes FLUX censorship and enables high-quality NSFW generation. V3 uses new training parameters with only real (non-AI) training images.

### Recommended settings
- **Strength:** 1.0
- **Guidance:** 4.0
- Use natural language prompting with detailed descriptions

### Sample prompts

**Prompt 1 (Bathroom selfie):**
```
a selfie of a young woman taking a selfie in a bathroom mirror. She is holding her phone up to take the picture with her right hand and her left hand is resting on her chest. The woman is shirtless and her body is slightly turned to the side. She has shoulder-length dark hair and is looking directly at the camera with a slight smile on her lips. The background is a white tiled wall.
```

**Prompt 2 (Pool scene):**
```
A topless 18-year-old girl with a messy yet charming updo, her fit physique on full display as she stands topless, breasts exposed, at the edge of a refreshing swimming pool. The simple background of the poolside setting allows her to be the main focus, depth of field blurs the surrounding lounge chairs and palm trees. With a playful laugh, she glances back at the camera. Full body candid photo, skimpy bikini bottom.
```

**Prompt 3 (Bathtub):**
```
a young woman sitting in a bathtub. She is completely naked, with her body facing the camera and her arms resting on the edge of the tub. She has dark hair that is pulled back in a ponytail and is looking directly at the camera with a slight smile on her face.
```

**Prompt 4 (Outdoor nature):**
```
A full body shot of a tall beautiful fit young woman with long dark brown hair. She is standing in front of a large oak tree and is looking directly at the camera with a slight smile on her face. The woman is shirtless, pulling her white yoga pants down. Her small perky breasts are prominently displayed. The image is a low angle shot taken from a side angle.
```

**Prompt 5 (Indoor bedroom):**
```
nfsw, naked, nude, 18-year-old girl, amateur selfie, wears pink hotpants, lifting her tshirt above her small perky tits, pink detailed nipples, she is holding the phone with which she is taking a selfie in her left hand, she is kneeling on the parquet floor of her room, brunette, straight bangs, short hair, socks
```

### Notes
- V3 trained exclusively on real images (no AI-generated)
- Works well with detailed natural language descriptions
- Can be combined with other anatomy LoRAs
- Use descriptive prompts for best results

---

## NSFW Master FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `NSFW_master.safetensors` |
| **Civitai** | https://civitai.com/models/667086/nsfw-master-flux |
| **Trigger word** | None |
| **Strength** | 0.7-0.8 (recommended 0.8) |
| **Type** | NSFW Unlock / Enhancement |
| **Compatibility** | FLUX Dev |

### Description
All-in-one NSFW LoRA for FLUX. Enables NSFW generation with good quality results. Some minor issues may exist that will be fixed in future updates.

### Sample prompts

**Prompt 1 (Pool villa):**
```
A cinematic panavision style full body photo of a young beautiful supermodel woman (26 years old), standing in a luxurius pool with a italien villa in the background, Skinny slim skinny body, skinny toned body, (narrow skinny waist), lon skinny legs, ((she is wearing a gold Shiny Minimalistic Slingshot bikini)), (Highly revealing gold slingshot bikini), (Ripped abs:1.9), ((Toned abs:1.9)), dark tanned skin, tight gap, ((camel toe)), Long black shiny hair in a high ponytail, (gold choker).
```

**Prompt 2 (Goth bedroom):**
```
A stunningly beautiful slim 20 year old with brown hair blonde highlights and unique make is wearing goth clothes and smiling at the viewer in her bedroom, full body
```

**Prompt 3 (Bedroom nude):**
```
The image is a photograph featuring a stunningly beautiful woman with a slender, athletic build, positioned in a minimalist, softly lit room. She has a fair complexion, ginger hair, bluish green eyes and wearing a light red headband. She is lying on a bed with a light gray blanket and a few scattered pillows in muted colors, including pastel shades of pink and blue. Her breasts are small to medium-sized, and she has a natural pubic area visible.
```

### Recommended combinations

**Combination 1 (Reclining nude):**
```
<lora:NSFW_master:0.7>
<lora:Reclining_Nude:0.4>
<lora:Oiled_Skin:0.4>
```

**Combination 2 (Skinny body):**
```
<lora:NSFW_master:0.7>
<lora:Whisper_FLUX:1>
<lora:Breast_size_slider:1>
<lora:aidmaImageUpgrader:0.3>
<lora:Flux_Skinny_Thinspo_Petite:1>
```

### Keywords
- `stunningly beautiful`
- `slender` / `slim` / `skinny`
- `athletic build`
- `full body`
- `naked` / `nude`
- `natural pubic area`

### Notes
- Recommended weight: 0.8
- Works well with body type LoRAs (skinny, petite)
- Combine with Oiled Skin for glossy effect
- Good base for various NSFW scenes

---

## Professional Nude Photography V3

| Parameter | Value |
|-----------|-------|
| **File** | `Professional_Nude_Photography_V3.safetensors` |
| **Original filename** | `FLUX_FD-Nude-Feamle-V3-R16.safetensors` |
| **Civitai** | https://civitai.com/models/652149/professional-nude-photography-v30-flux |
| **Trigger word** | None |
| **Strength** | 0.7-0.8 (reduce for 3D images) |
| **Type** | Style / NSFW / Photography |
| **Compatibility** | FLUX Dev, GGUF Q-4_0 |

### Description
High-quality professional nude photography LoRA. Trained on female anatomy. Produces excellent quality results. Works well at native resolutions 1280x1536 or 1536x1920.

### Recommended settings
- **Strength:** 0.7-0.8 (flexible)
- **Resolution:** 1280x1536 (higher quality) or 1536x1920
- For 3D/other styles: reduce LoRA value

### Sample prompts

**Prompt 1 (Garden portrait):**
```
Sexy Selfie portrait, full body, sitting on a wooden bench, Photography, beautiful 18 years old, wearing glasses, reading an old book, sleepily, very sensual, fake shy, dramatic eye makeup, incredibly pretty, fit, love, highly detailed features, highest quality, highly detailed skin, (freckles on her nose, long hair, natural red hair, expressive blue eyes, soft pale skin.), perfect eyes, detailed eyes, detailed pupils, shy smile, (((gather her hair behind her ear))). Background is a gorgeous english garden in spring, with flowering wisteria, roses, and lavender, and tall weeping willows. Faith Seed, topless, exposed breasts absolute body aesthetics with enhanced proportions, teardrop breasts, natural breasts, Real Nipples and Areola Textures, RNAT, areola, breasts, nipples, realistic,
```

**Prompt 2 (Library):**
```
A chinese female with large breasts in a library naked, ((standing under a sign that says "NUDE 4 FLUX"))
```

**Prompt 3 (Bedroom petite):**
```
1girl, 22 years old, pretty girl, blushing, shy, (petite), solo, blonde hair, beautiful hair, ponytail, dark blue eyes, cute, beautiful smile, (perfect realistic breasts), erect nipple, topless, puffy areola, (pink high waisted shorts that are pulled down revealing pussy), pussy juice, cute girly bedroom, posters, messy, after a long day of school, score_9, score_8_up, score_7_up, cute petite blonde, cute pajamas pulled down, skimpy, pants_down, pants_around_legs, low lights, godrays, hair clip, hair_clip, pajamas down, pussy visible, shaved pussy
```

**Prompt 4 (Cheerleader locker room):**
```
photorealistic, gorgeous cheerleader looking at viewer, mouth open, surrounded by topless cheerleaders, locker room, no shoes, cinematic lighting, topless, exposed breasts, perfect breasts, athletic body, thong on hips, all different race girls teardrop breasts, natural breasts, Real Nipples and Areola Textures, RNAT, areola, breasts, nipples, realistic
```

**Prompt 5 (Stage spotlight):**
```
Hack Forums scrapped posted to WhatsApp r/me_irl Shot on iPhone A full body photo of a slim athletic latina nude woman with large breasts standing on a stage underneath a spotlight, audience visible, pubic hair, hips, hairy pussy, Real Nipples and Areola Textures, RNAT. <lora:BreastShaper_splendid_droplets_Flux_v3.0:.7> <lora:Professional_Nude_Photography_V3:.7> <lora:amateurphoto-version4-final-9555:.35>
```

**Prompt 6 (Sydney penthouse):**
```
In a luxurious penthouse suite overlooking the stunning Sydney Harbour, a beautiful (topless:2.0), (naked:2.0), 20-year-old fitness model stands confidently in the doorway of her elegantly decorated bedroom. Her long black curly hair accentuates her sharp jawline and athletic neck. Large perfect breasts. She gazes directly at the viewer with piercing hazel eyes, full of vitality and determination, enhanced by subtle yet skillfully applied medium makeup that highlights her features. Clad in exquisite red and black panties adorned with intricate lace details, she radiates confidence and allure. The alluring garter belt complements her figure gracefully, while thigh-high stockings hug her toned legs.
```

**Prompt 7 (Sexy secretary):**
```
realistic art of Formidable female as a sexy secretary with glasses, high detailed, A Colombian woman, svelte goddess with majestic elegance, human realistic, intricately detailed, graceful divine with long, wavy hair, styled into a giant braid with bright highlights, naked breasts, topless, adorned with a choker, low-rise panties, net band around her leg, enchanting appearance. human realistic details, dynamic view, wide-angle from slightly below, absolute body aesthetics with enhanced proportions, deep_based_heavy_breasts, naturally Teardrop-shaped_breasts, dynamic, detailed nipples
```

### Recommended combinations

**Combination 1 (Boring Reality):**
```
<lora:Boreal-FD:0.35>
<lora:Professional_Nude_Photography_V3:1>
<lora:Amateur_Photography:0.1>
```

**Combination 2 (Realism + Breast detail):**
```
<lora:XLabs_Flux_Realism:0.8>
<lora:Pyros_PMI:0.4>
<lora:Real_Nipples_Areola_GMR:0.75>
<lora:BreastShaper_splendid_droplets:0.9>
<lora:Professional_Nude_Photography_V3:0.6>
```

**Combination 3 (Full stack):**
```
<lora:Pyros_PMI:1>
<lora:Real_Nipples_Areola_GMR:1>
<lora:BreastShaper_splendid_droplets:1>
<lora:Professional_Nude_Photography_V3:1>
```

**Combination 4 (Busty):**
```
<lora:Pyros_PMI:1>
<lora:Real_Nipples_Areola_GMR:1>
<lora:BreastShaper_splendid_droplets:1>
<lora:Professional_Nude_Photography_V3:1>
<lora:Flux-busty-LoRA:1>
```

**Combination 5 (Real-lora stack):**
```
<lora:real-lora:0.85>
<lora:Real_Nipples_Areola_GMR:0.9>
<lora:BreastShaper_splendid_droplets:0.9>
<lora:Professional_Nude_Photography_V3:0.8>
```

### Keywords
- `topless` / `exposed breasts`
- `teardrop breasts` / `natural breasts`
- `Real Nipples and Areola Textures` / `RNAT`
- `areola` / `nipples`
- `absolute body aesthetics with enhanced proportions`
- `photorealistic`
- `athletic body`

### Notes
- Very high quality output!
- Works great with breast shaper and nipple LoRAs
- Female anatomy only (no male genitalia training)
- Higher resolutions work well (1280x1536, 1536x1920)
- Multiple tested combinations provided above

---

## SECRET SAUCE HERO V2.1

| Parameter | Value |
|-----------|-------|
| **File** | `SECRET_SAUCE_HERO_V2.1.safetensors` |
| **Original filename** | `SECRET SAUCE HERO V2.1.safetensors` |
| **Civitai** | https://civitai.com/models/889205/secret-sauce?modelVersionId=1028490 |
| **Trigger word** | None (optional: `BrooklynMixer`) |
| **Strength** | 0.35-1.0 |
| **Type** | Style / Enhancement |
| **Training** | ~5600 images |

### Description
High-quality style enhancement LoRA trained on approximately 5600 images. Produces ultra-realistic, high-definition outputs with excellent detail and professional photography aesthetics.

**IMPORTANT:** This model's dimension is large, so combining it with many other LoRA models may not be beneficial. Use sparingly with other LoRAs.

### Sample prompts

**Prompt 1 (Full body nude portrait):**
```
BrooklynMixer, ultra high definition and ultra realistic photo of a gorgeous nude young woman looking at the viewer with a seductive smile facing the viewer. Incredibly detailed zoomed out full body professional photo.
```

**Prompt 2 (Sports bra portrait):**
```
a high-resolution photograph featuring a young woman with fair skin and a light tan complexion, she has shoulder-length, curly brown hair that is slightly tousled, and her facial features are delicate with high cheekbones, full lips, and expressive brown eyes, she is wearing a form-fitting, black sports bra that accentuates her ample breasts, with a plunging neckline that reveals a hint of cleavage, the bra is made of a soft, mesh-like material that clings to her body, emphasizing her curves, the background is a plain, light blue wall, which contrasts with her dark skin tone, making her the focal point of the image, the lighting is soft and diffused, creating a warm and inviting atmosphere, the overall composition is intimate and serene, emphasizing the subject's natural beauty and confidence
```

**Prompt 3 (Edgy harness outfit):**
```
A bold woman with jet black hair in a sleek ponytail with shaved sides, standing in a pronounced S-curve pose. She wears a lime-green harness-inspired cropped top with thick black leather thigh straps and silver metal details. The outfit is striking and edgy, no transparency on private areas. Neon-lit studio with a concrete background. Confident, street-flavored energy <lora:aidmaFLuxPro1.1_v0.3:0.8> aidmafluxpro1.1, <lora:Flux_Improved_Female_Nudity_v2:0.4>, <lora:female_anatomy:0.35>, <lora:perfection style v2d:0.6> perfection style, <lora:SECRET_SAUCE_HERO_V2.1:0.35>
```

### Keywords
- `BrooklynMixer` - optional trigger
- `ultra high definition`
- `ultra realistic photo`
- `professional photo`
- `high-resolution photograph`
- `detailed`

### Notes
- Large model dimension - avoid stacking with many other LoRAs
- Works well standalone for high-quality outputs
- Use lower strength (0.35) when combining with other LoRAs
- Best for professional photography style images

---

## SuperDemidov Style

| Parameter | Value |
|-----------|-------|
| **File** | `superdemidov.safetensors` |
| **Civitai** | https://civitai.com/models/976594/superdemidov-style-for-flux?modelVersionId=1093702 |
| **Trigger word** | None |
| **Strength** | 2.5-3.0 (high strength required!) |
| **Type** | Photographer Style |
| **Training** | 20 photos by Egor Demidov |

### Description
LoRA trained on 20 photos of Russian photographer Egor Demidov. Captures his distinctive style: artistic photo processing, curvy girls, exotic and sexy poses, shiny oiled skin.

### Recommended settings
- **Sampler:** Euler
- **Scheduler:** Normal or Simple
- **Strength:** 2.5-3.0 (unusual but required for this LoRA)

### Sample prompts

**Prompt 1 (East Asian lingerie):**
```
This is a high-resolution photograph featuring a woman posed provocatively on a minimalist white chair in a dimly lit room. The woman is of East Asian descent with long, straight, dark brown hair tied in a high ponytail. She has a fit, toned physique with a curvaceous figure, including a prominent, rounded buttocks and a slim waist. She is wearing a black lace bra and matching thong, which accentuates her hourglass shape. The bra has delicate floral lace detailing and is semi-transparent, revealing her medium-sized breasts. She also wears black lace thigh-high stockings that add to the sultry, seductive vibe of the image.
The background is a plain, concrete wall with a smooth texture, illuminated by natural light streaming in from an unseen source, casting soft shadows and creating a warm, intimate atmosphere. The chair she sits on has a minimalist, modern design with thin, white metal legs. A white pillow is placed on the chair, adding a touch of comfort to the otherwise stark setting. The overall aesthetic is one of modern elegance combined with sensuality, with a focus on the subject's body and the interplay of light and shadow.<lora:superdemidov:2.7>
```

**Prompt 2 (Back pose lingerie):**
```
This is a high-resolution photograph of a young woman posing provocatively on a minimalist, modern white metal frame chair with a white cushion. She is positioned with her back to the camera, looking over her shoulder with a sultry expression. Her long, straight dark brown hair cascades down her back. She is wearing a matching set of black lace lingerie, consisting of a bra that accentuates her medium-sized breasts and a thong that highlights her round, firm buttocks. Her skin is a warm, light tan, and she has a toned, athletic physique. She is also wearing sheer black thigh-high stockings and black high-heeled shoes. The lighting in the room is soft, casting natural shadows that enhance the curves of her body and the texture of the lace. The background is a plain, gray concrete wall that contrasts sharply with the white chair and her skin tone. The overall aesthetic is sleek and modern, with a focus on the woman's figure and the interplay of light and shadow. The image exudes a sense of sensuality and confidence. <lora:superdemidov:2.8>
```

**Prompt 3 (Wet look bikini):**
```
This is a high-resolution photograph of a young woman posing provocatively on a concrete floor against a plain white wall. She has a light brown complexion and long, dark brown hair that is wet and slicked back, giving her a sleek, polished appearance. Her facial features are sharp and defined, with full lips painted a deep red, and she has a strong jawline and high cheekbones. She is wearing a revealing black bikini that consists of a halter top with thin straps that crisscross around her neck and a matching thong bottom, emphasizing her toned physique and small, firm breasts. Her skin is glistening, suggesting she might have applied oil or lotion to accentuate her body contours. The lighting is soft yet direct, casting subtle shadows that highlight her curves and the texture of her skin. The concrete floor is grey and slightly textured, adding a raw, industrial element to the setting. The overall mood of the image is sensual and intimate, with a focus on the subject's physical attributes and the play of light and shadow on her body. The background is minimalistic, ensuring that the viewer's attention remains solely on the woman. <lora:superdemidov:2.8>
```

**Prompt 4 (Blue fishnet lingerie):**
```
The image is a high-resolution photograph capturing a woman in a provocative pose. She is lying on her back on a black leather chair with her head slightly tilted to the left, her long, dark brown hair cascading over her shoulders. The woman has a light olive skin tone and a fit, toned physique. She is wearing a matching blue fishnet lingerie set that accentuates her curves; the bra has a plunging neckline, revealing her nipples through the mesh, and the panties are high-cut, also made of fishnet material, highlighting her flat stomach and hips. Her hands are pulling the sides of her panties, emphasizing her slender waist and toned abs. Her nails are painted a bright red, adding a pop of color to the otherwise monochromatic scene. The background is a minimalist setting with a smooth, light grey concrete floor and a black metal chair, which contrasts with the woman's skin and the blue of her lingerie. The lighting is soft but focused, highlighting the sheen of her skin and the texture of the fishnet fabric. The overall mood is sensual and intimate, with a focus on the woman's body and the intricate details of her attire. <lora:superdemidov:2.8>
```

**Prompt 5 (Silk drape boudoir):**
```
Beautifully lit, elegant artistic photograph of a sexy woman radiating elegance and sensuality. (((has straight black long wet hair parted in the middle))). She stands in a softly lit, minimalistic room with warm shadows, her sexy tattooed body partially hidden by a delicate, translucent silk drape that wraps around her curves, accentuating the graceful lines of her silhouette. Her pose is both balanced and relaxed, evoking a sense of mystery and seduction, with subtle lighting highlighting the softness of her skin. Her gaze is sexy and alluring, framed by her hair parted in the right side. The overall composition is reminiscent of classic boudoir art photography, focusing on form, shadow, and the balance between light and dark to evoke a sense of timeless seduction and artistic sensuality. ArtfulNSFW <lora:superdemidov:2.8>
```

### Keywords
- `high-resolution photograph`
- `provocative pose`
- `toned physique`
- `curvaceous figure`
- `shiny skin` / `glistening`
- `minimalist setting`
- `concrete wall/floor`
- `light and shadow`
- `sensual and intimate`

### Style characteristics
- Industrial/minimalist backgrounds (concrete, white walls)
- Oiled/shiny skin effect
- Strong interplay of light and shadow
- Curvy, toned physiques
- Exotic and sexy poses
- High-resolution, detailed photography

---

## Flux Improved Female Nudity V2

| Parameter | Value |
|-----------|-------|
| **File** | `Flux_Improved_Female_Nudity_v2.safetensors` |
| **Civitai** | https://civitai.com/models/643366/flux-improved-female-nudity |
| **Trigger word** | None |
| **Strength (Dev)** | 1.0-2.0 |
| **Strength (Schnell)** | 1.4 (with 5-6 steps) |
| **CFG** | 1 (if using CFG sampler) |
| **Type** | NSFW Unlock / Enhancement |
| **Training** | 53 hours, 4100 images (realistic + cartoons + illustrations) |

### Description
High-quality NSFW unlock LoRA trained for 53 hours on a diverse dataset of 4100 images including realistic photos, cartoons, and illustrations. V2 offers improved consistency over V1 and was trained with multiple resolutions.

### Recommended settings
- **Dev model:** Strength 1.0-2.0 (recommended)
- **Schnell model:** Strength 1.4 with 5-6 steps (more volatile)
- **CFG:** Set to 1 if using CFG sampler
- ComfyUI workflows included in example images on Civitai

### Sample prompts

**Prompt 1 (Nightclub scene):**
```
Realistic dynamic photo of Blondi, 20 years old girl, naked, heels, kneels on the table in crowded nightclub, surronded by multiple guys touching her, hands cover her breast, men hands on girl butts
```

**Prompt 2 (Living room spread):**
```
A beautiful girl sitting on a chair, completely nude and naked, spreading her legs, showing her pussy, small round breasts, in a living room with an old fashioned wall painting
```

**Prompt 3 (Sauna relaxation):**
```
realistic photo of a young woman sitting naked on a white towel in a sauna. sweat runs down her body, wet skin, sweaty, small breasts, warm light, relaxing, smiling,
```

**Prompt 4 (Shower scene):**
```
full body shot, ultra high quality, realistic photo of a young naked woman standing under a shower, wet blond hair, blue eyes, wet body and skin, head slightly tilted back, enjoying the water, small breasts, warm light, relaxing, smiling,
```

**Prompt 5 (Classroom):**
```
A female Student is sitting on a chair in a classroom. She is nude. Her legs are spread and her pussy is visible.
```

**Prompt 6 (Japanese nurse):**
```
Full-body photo of a stunning Japanese nurse backlit by soft light. She stands in a modern bedroom making a heart with her hands on her back, exuding warmth and kindness. She is topless wearing silky panties with a heart detail. Her beautiful face, with delicate features , pale complexion, and fair skin, beams at the viewer. Her black hair falls down her back like silk, framing her slim hourglass figure. Small breasts are perfectly proportioned to her slender physique. Her gaze meets ours, inviting us into her world of care and compassion.
```

**Prompt 7 (Locker room):**
```
Ultra detailed realistic skin texture), 1girl, cute girl, 20 years old, wet and sweaty, freckles, brunette, green eyes, detailed face, cute face, adorable bright eyes, detailed and realistic eyes, petite body, (large breasts;1.3), small waist, HD32K, incredibly detailed, posing in lockerroom, arched back, (teasing, naughty expression), small elegant necklace, god rays, undressing, parted lips look at viewer,spread legs,perfect shaved pussy,(moaning,eyes slutty look),((very horny))
```

**Prompt 8 (Korean doctor):**
```
Full-body photo of a stunning Korean female doctor backlit by soft light. She stands in a modern bedroom holding a labcoat, exuding warmth and kindness. She is topless wearing silky champagne panties with a heart, and a stethoscope. Her beautiful face, with delicate features , pale complexion, and fair skin, beams at the viewer. Her black hair falls down her back like silk, framing her slim hourglass figure. Small breasts are perfectly proportioned to her slender physique. Her gaze meets ours, inviting us into her world of care and compassion.
```

**Prompt 9 (Greek beach):**
```
NSFW Full body nude photo. 1Girl, beautiful tanned naked Greek woman. sitting nude at the beach, A serene detailed beach scene, crystal-clear waters and white sands in sharp detail, untouched by humanity, pristine shore with blue surf, big nude tits, beautiful nude ass, Professional nude photo. Perfect dark brown eyes, highly detailed beautiful expressive, dark brown eyes, detailed dark brown eyes. Detailed Greek face. Ponytail purple hair tie. big dark erect nipples, late afternoon soft fill light. conjuring with blue and purple flames, Sharp 4" x 5" photograph, f22 sharp Depth of field, film, professional, highly detailed dynamic lighting, photorealistic, 8k, raw, rich, intricate details, realistic, sharp backgroundrealism. nicebeach, <lora:nicebeach2:1.0>
```

### Keywords
- `naked` / `nude`
- `topless`
- `spread legs`
- `showing pussy`
- `small breasts` / `large breasts`
- `wet skin` / `sweaty`
- `realistic photo`
- `full body shot`

### Notes
- Works with realistic, cartoon, and illustration styles
- Dev model produces better results than Schnell
- Can be combined with other LoRAs (used at 0.4 in SECRET SAUCE example)
- ComfyUI workflows available in Civitai example images

---

## FluxUnchained LoRA

| Parameter | Value |
|-----------|-------|
| **File** | `fluxunchained-lora-r128-v1.safetensors` |
| **Civitai** | https://civitai.com/models/686766/fluxunchained-lora |
| **Trigger word** | None |
| **Strength** | 0.7-1.0 (R128), 1.5 (R16) |
| **Type** | NSFW Unlock |
| **Rank** | 128 (detailed) / 16 (smaller) |

### Description
LoRA extracted from the FluxUnchained model by socalguitarist. Provides NSFW unlock capabilities with controllable strength. Rank128 version provides more genital detail (recommended for NSFW), Rank16 saves space but requires higher strength.

### Why use LoRA instead of full model
- Works on CivitAI generation services (base models not supported)
- Controllable strength
- Can combine/merge with other LoRAs and base models

### Sample prompts

**Prompt 1 (Bedroom scene):**
```
Pink girly bedroom with teddy bears, sunshine. 18 year old sexy nude girl spread her legs to show her vagina. She is a supermodel and smiles.
```

**Prompt 2 (Night scene):**
```
Beautiful 18 year girl fully naked, very cute, super cute girl, cute face, shy smile, realistic tanned oiled shiny skin, platinum blonde short ponytail hair, blue eyes, flat chest, shaved naked pussy, glowing neon bracelets, glowing neon choker, petite, skinny, standing, worn down unpaved parking lot at night, dark nightime, littered floor
```

**Prompt 3 (Hallway scene):**
```
fully naked swedish woman, young and small slim body with tender thin figure, small naked nude body, white bleached blonde wavy hair, cute and soft face, deeply suntanned seemless naked skin, large round breasts, naked shaved pussy, hands behind back, standing in narrow hallway, worn down wet concrete walls, only light bulb at nightime, concrete walls, concrete floor
```

**Prompt 4 (Spring break - with RealToons):**
```
A high-resolution, hyper-realistic photograph of a gorgeous, carefree college girl basking in the golden glow of a perfect spring break afternoon. She exudes an effortless, sun-kissed beauty. Her long, beach-waved hair, kissed by the sun with natural highlights, cascades down her shoulders. She wears a tiny, stylish bikini, the fabric hugging her curves in all the right places. She stands confidently on the golden sands of a tropical beach, the turquoise ocean stretching endlessly behind her. <lora:Real_Toons_Flux:0.7> RealToons <lora:fluxunchained-lora-r128-v1:0.7>
```

**Prompt 5 (Corporate setting - with RealToons):**
```
A high-resolution, hyper-realistic photograph of a stunningly fashionable businesswoman, standing with smoldering confidence in a sleek corporate conference room. A tailored black blazer, fitted to perfection, clings to her curves. Her skintight pencil skirt hugs her hips and thighs. She stands confidently in sleek, towering stilettos. The city skyline dominates the background through massive floor-to-ceiling windows. <lora:Real_Toons_Flux:1> RealToons <lora:fluxunchained-lora-r128-v1:0.7>
```

**Prompt 6 (Basement scene):**
```
Beautiful 18 year swedish girl, ((fully naked)), very cute, super cute, cute face, shy smile, realistic tanned oiled shiny skin, platinum blonde short ponytail hair, blue eyes, small breasts, small chest, shaved naked pussy, petite, skinny sporty body, worn down mattress, on all sides worn down dirt stained curtains hanging down worn down concrete basement walls, single light bulb in basement, wet stained dirty interior
```

### Keywords
- `fully naked` / `nude`
- `naked shaved pussy`
- `tanned oiled shiny skin`
- `petite` / `skinny`
- `small breasts` / `large breasts`
- `realistic`

### Recommended combinations
- `<lora:Real_Toons_Flux:0.7-1.0>` - RealToons style
- `<lora:fluxunchained-lora-r128-v1:0.7>` - Standard NSFW strength

### Notes
- R128 (Rank 128) recommended for NSFW - more genital detail
- R16 (Rank 16) smaller file, use strength ~1.5 to compensate
- Works well with RealToons style LoRA
- Good for explicit nude scenes with detailed anatomy

---

## FLUX Cum on Face

| Parameter | Value |
|-----------|-------|
| **File** | `Flux_Cum_On_Face.safetensors` |
| **Original filename** | `9cd3fddff95e898ea5534c4eb6a6509b.safetensors` |
| **Civitai** | https://civitai.com/models/924374/flux-cum-on-face |
| **Trigger word** | `Cum on face`, `facial` |
| **Strength** | 1.0-1.2 |
| **Type** | Effect / NSFW |

### Description
LoRA for generating facial/cum on face effects. Works well with character LoRAs without altering the face. Provides realistic cum textures and placement on face, breasts, and body.

### Sample prompts

**Prompt 1 (Gym Asian model):**
```
perfect lighting, ultra realistic, 8k resolution, ultra-detailed, thin Asian instagram model, professional photography, kneeling in a well lit gym with large floor to ceiling windows giving a blow job, smiling, white sports bra, hands on a very large penis that has just cum all over her breasts she is wearing tight light green leggings high on her waist creating a slight camel toe, thigh gap, beautiful city vista, fit, medium breasts, Asian, freckles on face, looking at the viewer, sunny, brunette, skinny, playful, realistic face, Sporty ponytail hair style
```

**Prompt 2 (Goth park scene):**
```
masterpiece, amateurish photo, best quality, high quality, highres, beauty shot of a beautiful young woman, ((Goth)), ((petite)), ((skinny: 1.3)), cute face, beautiful,18 years old, (black bold eyeliner, makeup:1.3), ((blue eyes: 1.3)), ((dark blue eyes)), large eyes, (pale skin: 1.2), caucasian, ear, black hair, (single braid: 1.3), messy hair, hair behind ear, freckles, petite body, (wet hair: 1.2), wet skin, outdoors, in a park, on a grass field, people in background, (naked: 1.2), perky breast, bare breast, bare pussy, facing the viewer, full body image, (focus on breast: 1.2), ((large tattoo on chest with the text "Cum tribute me"))), ((beautiful face)), looking at viewer, large eyes and soft pouty lips. she is embarrassed and smiling shyly, parted lips, seductive smile, cum, facial, cum all over face, ((cum on breast)), cum on chest, kneeling on a picnic blanket, on a sunny day in a crowded public park, (looking at viewer:1.3), (eyes open, open eyes:1.3), mouth open, medium breasts, nipples, Cum on face, ((excessive cum: 1.4)), navel, focus on pelvic, pussy, legs spread, arms behind back, leaning back, best quality, masterpiece, sharp focus, photo realistic, detailed, shallow depth of field, soft lighting, detailed face, detailed eyes, bright lighting, UHD, detailed skin,
```

**Prompt 3 (Pakistani hijabi):**
```
a selfie photo of a pakistani woman with Cum on face, blobs of murky mixture of clear and white (wet gloopy semen:1.2) cum on her face, eye lids and lips, <lora:Flux_Cum_On_Face:1.2>. the semen is dripping down her face with stringy droplets dangling from her eyelashes and eye lids, making it difficult for her see, she has thick lips and a cute, round face. she has a mocking wry grin and is blushing. she is sticking out her tongue and scrunching her face and wrinkling her nose in disgust. she is wearing a black hijab and has blobs of semen in the fabric and on her clothes and cleavage. the photo is taken from a high angle, showing her kneeling on the floor in a living room. she is dark skinned with a yellow tint. she is holding up 2 fingers in a victory sign and tilting her head to the side. she is wearing an yellow kameez with red embroidery with the top buttons undone and giving a view to her cleavage and bra. a triumphant grinning muscular broad-chested black african man is leaning back, relaxed in an easy chair with his thighs apart and wearing a bathtowel around his waist and his head is out of the frame. the livingroom as a mixture of south asian, pakistani and english decor. her lipstick is smeared and her eyes are red and tired. she has strands of messy hair loose from her hijab. a pair of mens shoes and socks and clothes are discarded in a messy pile in the corner of the image. her face is sweaty and makeup and eyeliner smeared. family photos on the wall, homely room. snapchat, instagram, flickr, social media, amateur, candid, iphone, she has an ace of spades tattoo on her breast with the letters "QOS" benieth it
```

**Prompt 4 (Indian BJ):**
```
bj rel, deepbj, 1girl, long hair, breasts, looking at viewer, long hair, 1boy, jewelry, hetero, nude, penis, indoors, lips, pubic hair, pov, erection, oral, fellatio, kayal, brown lipstick, steep arched eyebrows, freckles, realistic, black thong volumetric lighting, very detailed, cinematic film still, highly detailed, high budget, epic, OverallDetail, RAW photo, photorealistic, color graded cinematic, atmospheric lighting, sharp focus, shallow dof. small penis of fat white man, chest hair
```

### Keywords
- `Cum on face`
- `facial`
- `cum all over face`
- `cum on breast` / `cum on chest`
- `excessive cum`
- `wet gloopy semen`
- `semen dripping`

### User feedback
- Works well with character LoRAs without changing the face
- Good for realistic cum texture and placement

### Notes
- Does not alter character face - safe to combine with character LoRAs
- Works with various ethnicities and styles
- Use higher strength (1.2) for more pronounced effect
- Combine with other NSFW LoRAs for complete scenes

---

## Game of Cum V2

| Parameter | Value |
|-----------|-------|
| **File** | `Game_of_cum_v2.safetensors` |
| **Civitai** | https://civitai.com/models/998432/game-of-cum-facialcum-on-face-flux |
| **Trigger word** | `Facial`, `Cum on face` |
| **Strength** | 1.0 |
| **Type** | Effect / NSFW |
| **Version** | V2 |

### Description
Highly rated cum effect LoRA for FLUX. Creates realistic cum on face (facial), cum on breasts, and body covered in cum. Also produces decent looking nipples. Cannot create vagina - combine with anatomy LoRAs for that.

### Capabilities
- Cum on face (facial)
- Cum on breasts
- Body covered in cum
- Decent nipple generation
- Thick white cum texture

### Limitations
- Cannot create vagina (use anatomy LoRAs)

### Sample prompts

**Prompt 1 (Bed scene):**
```
Woman, naked, lying on bed, spread legs, white cum flowing from pussy, thick white cum all over face and body
```

**Prompt 2 (Luxury restaurant):**
```
cute brunette woman smiling sitting in a luxury restaurant table with a glass of transparent cum in her hand, cum on mouth, cum on face, cum on dress, cum all over body, thick white cum, cum splashes on ace and body, cum on table, smiling, Facial <lora:Game_of_Cum:1>
```

**Prompt 3 (Tongue out):**
```
cute black bobtail hair woman smiling with her tongue sticking out, cum on face, cum on dress, cum all over body, thick white cum, cum splashes on ace and body, smiling, Facial <lora:Game_of_Cum:1>
```

**Prompt 4 (Vintage magazine cover):**
```
Canon EOS D5, (cover of vintage 60s style "cumslut" magazine). The cover 1960s model is a Irish ginger woman, 22yo, long flowing thick ginger hair, highly educated, natural beauty, she is stunning, round face, (prominent nose), (cleavage) heavy dark porn makeup, black mascra long thick eyelashes, freckles, clear bright green eyes, Irish ginger woman named Nessa, unassuming, skinny slim figure, moaning from pleasure, slim, fit hourglass figure close up of a mature housewife's face. She is nude, with a leather slave collar. She is wet and sweaty, and looks surprised. There is abundant fresh, stringy, wet, runny, white transparent cum splattered across her face. There is fresh cum all over her face, cum in here half open mouth, Cum on face, ((other text on the magazine cover, head "CUM SLUT", CUM taste", "NESSA LOVES CUM",
```

**Prompt 5 (Japanese lingerie boutique):**
```
Cute Japanese girl with a huge amount of white cum on her face and chest, dribbling down her chin and coating her face in a thick layer. A Japanese celebrity is sitting on a luxurious couch with dark ornate metalwork placed in the center of a spacious lingerie boutique. The interior of the shop features deep, rich decor, with lingerie elegantly displayed on sleek shelves and mannequins, creating a sense of depth and luxury. Her face is softly illuminated by the boutique's warm lighting, highlighting her confident expression and sharp gaze. She is wearing an intricate blue lingerie set made of satin straps and sheer mesh panels. The bra features sheer triangular cups supported by bold blue satin bands, with a row of metallic gold studs adorning the top. Around her neck, she wears a thick blue choker with a gold square charm at the center. The underbust straps are decorated with gold geometric hardware and dangling gold chains. Her hair is elegantly styled, enhancing her dignified presence.
```

### Keywords
- `Facial`
- `Cum on face`
- `cum on mouth`
- `cum on breasts` / `cum on chest`
- `cum all over body`
- `thick white cum`
- `cum splashes`
- `stringy, wet, runny cum`

### User reviews (Civitai)
- "Probably the best in the cum market !!!"
- "So far, the best of its kind."
- "Love the Lora."

### Notes
- Considered the best cum effect LoRA available
- V2 improved over V1
- For vagina generation, combine with anatomy LoRAs
- Works with various settings and ethnicities
- Good for creative/artistic cum scenes

---

## Realistic People Photograph FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `aidmaRealisticPeoplePhotograph-FLUX-V0.1.safetensors` |
| **Civitai** | https://civitai.com/models/726732/realistic-people-photograph-flux |
| **Trigger word** | `aidmaRealisticPeoplePhotograph` |
| **Strength** | 0.5-1.0 |
| **Type** | Style / Realism / Photography |
| **Version** | V0.1 |

### Description
First LoRA in a series aimed at making images hyper-realistic with true photography style. Focused on people photography with professional quality output.

### Features
- Professional photo style
- Way more realistic details
- More sharpness
- Better and more intense colors
- Wider variety of faces
- Cooler image composing

### Sample prompts

**Prompt 1 (Model with sign):**
```
photograph, full body photo, a beautiful blonde super model holding up a sing with text:"realistic people", Canon eos 5d mark 4, Depth of field 100mm, lovely, symmetry, photography, aidmaRealisticPeoplePhotograph
```

**Prompt 2 (Indian goth):**
```
modeling photograph, rakshasa, indian beautiful, dark, mysterious, goth makeup, detailed flawless face, dramatic darkroom lighting, high exposure, (bubbly background:0.7), head and shoulders, 80mm camera
```

**Prompt 3 (OnlyFans style):**
```
beautiful woman, brunette, smiling, 21 years old, brown eyes, messy bun, light makeup, hyperdetailed photography, sexy onlyfans post, in hotel room, upper body, red lingerie
```

**Prompt 4 (Ecuadorean model - detailed):**
```
A 19year old Adult Ecuadorean Woman, light Skin, olive green Eyes, ginger Hairs, french Crop Haircut, wearing a Red and cherry ombre Pink satin bodysuit, mesh bustier layer, pearl spaghetti straps, platform boots with buckles, made of Seersucker, Outdoor Sports Room, (full body photo:1.4), (full body:1.4), (full shot:1.4), (full body shot:1.4), Thumbs Across the Neck Gesture, (8k,RAW photo,best quality,masterpiece,realistic,HDR:1.3), incredibly absurdres, ultra high resolution, (1girl,18 years old:1.5), teen, bishoujo, Delicate skin, baby face, extremely beautiful face, big eyes, pink blush, small mouth, cute smile, (short neck:1.3), Perky breasts, slim waist, flat belly, narrow ass, deep cleavage, <lora:aidmaRealisticPeoplePhotograph:0.8> aidmarealisticpeoplephotograph
```

### Negative prompt (recommended)
```
(worst quality:2),(low quality:2),(normal quality:2),Erotic,lowres,bad anatomy,(watermark),(sauteed tap),bad hands,normal quality,((monochrome)),((grayscale)),text,2 faces,cropped image,deformed hands,twisted fingers,long neck,extra limb,poorly drawn hands,missing limb,disfigured,blurry,bad anatomy,mutilated,surreal,extra fingers,distorted face,draft,grainy,watermark,moles,Pregnancy, big belly, bad_prompt_version2-neg, easynegative, negative_hand-neg, ng_deepnegative_v1_75t
```

### Tested combinations

**Combination 1 (Artistic):**
```
<lora:aidmaRealisticPeoplePhotograph:0.5-1.0>
<lora:NippleDiffusion-Flux:1.0>
<lora:Style_Santiago_Caruso:1.0>
<lora:Cinematic_Glamour_Photography_F1D:1.0>
<lora:Midjourney_V7_FLUX:0.1>
```

**Combination 2 (NSFW Realism):**
```
Base: FLUX Dev
<lora:XLabs_Flux_Realism:0.6>
<lora:NSFW_master:0.8>
<lora:aidmaRealisticPeoplePhotograph:0.5>
```

**Combination 3 (Film Stock):**
```
Base: FLUX Dev
<lora:XLabs_Flux_Realism:0.6>
<lora:NSFW_master:0.8>
<lora:Kodak_Portra_400_F1D:0.4>
<lora:aidmaRealisticPeoplePhotograph:0.6>
```

**Combination 4 (Cyberpunk + Film):**
```
Base: FLUX Dev
<lora:XLabs_Flux_Realism:0.4>
<lora:NSFW_master:0.4>
<lora:Cyberpunk_Anime_Style_Flux:0.3>
<lora:Kodak_Portra_400_F1D:0.6>
<lora:aidmaRealisticPeoplePhotograph:0.5>
```

### Keywords
- `aidmaRealisticPeoplePhotograph` - **REQUIRED** trigger word
- `photograph`
- `professional photo`
- `hyperdetailed photography`
- `RAW photo`
- `Canon eos 5d mark 4`
- `Depth of field`
- `realistic`

### Notes
- Use trigger word `aidmaRealisticPeoplePhotograph` for best results
- First in a series of hyper-realistic LoRAs
- Works great with NSFW_master and XLabs Realism
- Combine with film stock LoRAs for analog look
- Lower strength (0.5-0.6) when stacking multiple LoRAs

---

[← Back to Index](INDEX.md)

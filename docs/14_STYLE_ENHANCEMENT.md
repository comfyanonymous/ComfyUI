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
- [CumHereV1](#cumherev1) - Non-facial cum (clothes, hair, body, creampie)
- [Amsterdam Red Light District](#amsterdam-red-light-district) - RLD window rooms environment
- [Beautiful Girls' Faces - PrettyNova](#beautiful-girls-faces---prettynova) - Elegant diverse female faces
- [Naked/Nude for FLUX Sevenof9](#nakednude-for-flux-sevenof9) - Strong NSFW nude push
- [Flux Lustly.ai Uncensored v1](#flux-lustlyai-uncensored-v1) - Male and female nudity
- [Flux Skin Texture](#flux-skin-texture) - Removes plastic look, adds skin detail
- [Eros v06](#eros-v06) - Human nudity with accuracy and controllability
- [FLUX Naked Female](#flux-naked-female) - Realistic naked females with pubic hair
- [FLUX NSFW LoRA](#flux-nsfw-lora) - General NSFW for FLUX
- [Flux-Cameltoe NSFW](#flux-cameltoe-nsfw) - Cameltoe effect for clothed models
- [NSFW FLUX LoRA (AiArtV)](#nsfw-flux-lora-aiartv) - NSFW with trigger word
- [Cum and play with FLUX](#cum-and-play-with-flux) - Facial cumshot/semen effects
- [Cum On Feet - FLUX](#cum-on-feet---flux) - Cum on feet effect
- [Cum Facial / Cum on face - FLUX](#cum-facial--cum-on-face---flux) - Cum facial/bukkake effects
- [Facial cum massive FLUX](#facial-cum-massive-flux) - Massive facial cum with trigger word FCLHGE
- [Cumbubbles FLUX](#cumbubbles-flux) - Cum bubbles on mouth and nose
- [Cum on face - Non-Face Altering](#cum-on-face---non-face-altering) - Cum on face without altering character face
- [Bukkake / Realistic Cum Facial Flux](#bukkake--realistic-cum-facial-flux) - Realistic bukkake facials
- [Universal Cum Enhancer](#universal-cum-enhancer) - Enhances other cum LoRAs with texture/shape
- [FLUX Gone Wild](#flux-gone-wild) - Top-tier NSFW style enhancer

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

## CumHereV1

| Parameter | Value |
|-----------|-------|
| **File** | `CumHereV1.safetensors` |
| **Civitai** | https://civitai.com/models/730199/cumherev1-flux-cum-lora-clothes-hair-non-facials |
| **Trigger word** | None (understands "cum" naturally) |
| **Strength** | 0.8-1.0 |
| **Type** | Effect / NSFW |
| **Training** | 7h on RTX 4090, 4000 steps, 34 images |

### Description
Non-facial cumshot LoRA focused on realistic cum on clothes (sweaters, panties, swimsuits), breasts, tummy, hair, legs, feet, creampie, and more. Complements facial cum LoRAs. No trigger word needed - just describe what you want specifically.

### Target areas
- Clothes (sweaters, panties, swimsuits)
- Breasts and nipples
- Tummy/belly
- Hair
- Legs and feet
- Ass and panties
- Body (full coverage)
- Creampie/cum in pussy

### Usage tips
- Weight 1.0 works well, reduce to 0.8 if too much cum
- No need for "white cum" - already understands cum is white
- Be very specific: "There is cum on her ass and panties"
- Use "a lot of cum" for more quantity
- Not trained on facials but may randomly add them

### Tested resolutions
- 832 x 1216
- 896 x 1152
- 1024 x 1024

### Sample prompts

**Prompt 1 (Creampie library):**
```
Young nude skinny 25 year old woman smiling while on a mat spreading legs, creampie, cum in her pussy, she is wearing a silver necklace, sharp focus on her pussy, cum is dripping down her ass onto the floor. sunlight is coming in trough the window in a library. she has a slim naked body. She is holding a sign that says 'Cum inside me!' with a heart on it.
```

**Prompt 2 (Cowgirl cum covered):**
```
full body shot of a slim and slender young cowgirl with large breasts. she is a blonde with freckles. sitting, spreading her legs, showing her pussy, creampie, cum drip, cum in pussy. her face and body are covered in loads of cum, cum shot. she is wearing a typical classic cowboy outfit and hat, white boots with spurs. her cowboy clothes have been ripped and torn apart, revealing her body, panties pulled aside. she is leaning back with her back on the wall, head down, looking away. she is sad, embarrassed, lost, open mouth. in an empty pub in the wild west, after hours, dimmed light, shadows.. <lora:legspread-flux:0.5> <lora:CumHereV1:0.5>
```

**Prompt 3 (Snapchat style):**
```
<lora:[FLUX]Noisify:0.8>, low light, noise, grain, jpeg artifacts, night, A low-quality 2015 Snapchat photo depicts a smiling woman in bed wearing a bra with cum on her mouth, cum drips, <lora:CumHereV1:1>
```

**Prompt 4 (Egyptian palace):**
```
Naked slender Egyptian girl with small perky breasts is lying on her back, wearing intricate golden Egyptian jewelry, exhausted, looking at viewer, scenic palace in Ancient Egypt, there is cum all over her body
```

**Prompt 5 (Amsterdam red light):**
```
half bodyshot photography from street at window, girl displayed behind behind window, view through glass, reflections in window, laughing hysterically, girl is a shy and ashamed brunette girl with big boobs, choker, wearing a neon pink skirt and black see through tube top, big breasts, huge cleavage, wearing clear pleaser 10inch plattform heels, garter, fishnet thigh high stockings, she is sitting on red bed between two old men touching her thigh and breast:1.6, in red light district street in amsterdam, she has cum on her face and chest, bukkake, she feels sad, feeling uncomfortable and shocked, she has a lot of cum on her face, facial cumshot, vertical purple neon tube lights along windows, pink neon letters "happy hour bukkake", spotlight on girl
```

### Prompting examples
- `There is cum on her ass and panties`
- `There is cum all over the front of her swimsuit`
- `There is cum on her tummy`
- `There is a lot of cum on her breasts and nipples`
- `There is cum all over her body`
- `creampie, cum in her pussy, cum drip`

### Tested combinations

**Combination 1 (Creampie focus):**
```
Checkpoint: UltraReal Fine-Tune v2.0
<lora:Creampie_Flux:0.5>
<lora:CumHereV1:1>
```

**Combination 2 (Full cum scene):**
```
Checkpoint: STOIQO Afrodite FLUX
<lora:Perfect_Full_Round_Breasts:1>
<lora:Cum_On_Face_FLUX:1>
<lora:CumHereV1:1>
<lora:NippleDiffusion-Flux:1>
```

**Combination 3 (Legspread + creampie):**
```
Checkpoint: Real Horny Pro V3
<lora:CumHereV1:1>
<lora:Legspread_Flux:1>
```

**Combination 4 (Noisy/amateur):**
```
<lora:Noisify:0.8>
<lora:CumHereV1:1>
```

**Combination 5 (Skinny body):**
```
Base: FLUX Dev
<lora:CumHereV1:1>
<lora:Flux_Skinny_Thinspo_Petite:1>
```

**Combination 6 (Amateur style):**
```
Base: FLUX Dev
<lora:Amateur_Flux:0.7>
<lora:Huge_cumshot_facial:0.85>
<lora:CumHereV1:0.9>
```

### Compatible checkpoints
- flux1-dev-fp8
- flux1-dev-fp16
- flux1-schnell (adjust weight)
- UltraReal Fine-Tune
- STOIQO Afrodite
- Real Horny Pro V3
- Fluxmania Kreamania

### Notes
- Complements facial cum LoRAs (use both for full coverage)
- Trained on realistic images, anime/cartoon results are hit or miss
- Works well with legspread, creampie, and body type LoRAs
- Lower weight if cum looks unnatural

---

## Amsterdam Red Light District

| Parameter | Value |
|-----------|-------|
| **File** | `Rldstreet.safetensors` |
| **Civitai** | https://civitai.com/models/784975/amsterdam-redlight-district-outside-rooms |
| **Trigger word** | `rldstreet` |
| **Strength** | 1.0-2.0 |
| **CFG** | 7.5 |
| **Type** | Environment / Location |
| **Version** | V1 |

### Description
Amsterdam Red Light District street environment LoRA. Based on Office 52 outside rooms photos. Creates authentic RLD window room scenes with neon lighting, window displays, and street atmosphere.

### Environment features
- Window displays with girls
- Neon tube lights (purple, red, blue)
- Street view through glass
- Reflections in windows
- Red/pink lighting
- Signboards with neon accents

### Recommended settings
- **Weight:** 2.0 (high weight recommended)
- **CFG:** 7.5
- **Style:** `rldstreet style`

### Sample prompts

**Prompt 1 (Nerdy girl bukkake):**
```
view from above, Homely very nerdy girl with thin metallic black rimmed glasses, grey eyes, she looks very insecure and naive, ponytail light brown-blonde ponytail hair, black pvc halter top, large breasts, huge cleavage, neutral background, cum on face, cum facial. darkbrown lipstick, bukkake. cumshot, jizz she is giggling histerically (horse faced) but almost cries, very emotional
```

**Prompt 2 (Spanish woman neon city):**
```
This is a high-resolution photograph featuring A young brunette spanish woman posed. curly hair, She is wearing SGWN19clothing, a tight, cut-out pink bodysuit. squeezing breasts, thigh high boots. This photograph captures with backdrop of a city with neon light. there are a lot of signboard behind that use red and blue accent.
```

**Prompt 3 (Living room with students):**
```
half bodyshot photography of a brunette spanish girl with big boobs, wearing a pure pvc skirt with purple-black snake print, and black laced blouse, shiny dark lipstick, big breasts, huge cleavage, she is wearing a pair of hot black overknee boots, and the plattform base is 10inch, in living room sitting on a blue sofa in amsterdam, rldstreet style, girl sitting in between two horny male students in casual outfit, trying to kiss and touch
```

**Prompt 4 (Window bukkake scene):**
```
half bodyshot photography of a shy and ashamed brunette girl with big boobs, heavy makeup, dark eyeliner and eyeshadow, choker, wearing a black pleated skirt and black laced blouse, big breasts, huge cleavage, h33l wearing clear:1.6 pleaser 10inch plattform heels, garter, fishnet thigh high stockings, sitting on bed with crossed legs, behind a window in red light district street in amsterdam, rldstreet style, she is very shy and ashamed, she has cum on her face and chest, bukkake, she feels sad, feeling uncomfortable and shocked by old men starring at her through the window from street, feeling helpless and ashamed, she has a lot of cum on her face, facial cumshot, vertical purple neon tube lights along windows, spotlight on girl, girl surrounded by old men:1.6, old men are horny and touchy, talking to girl, old man trying to kiss her:1.6, old man reaches under her skirt:1.6
```

**Prompt 5 (Leopard print window):**
```
half bodyshot photography of a brunette girl with big boobs, wearing a pure pvc mini skirt with abstract black leopard print, and black lingerie blouse, big breasts, huge cleavage, she is wearing a pair of hot black overknee boots, the plattform base is 10inch posing behind a window in red light district street in amsterdam, rldstreet style, girl surrounded by old men, horny and touchy
```

### Keywords
- `rldstreet` - **REQUIRED** trigger word
- `rldstreet style`
- `behind a window in red light district street in amsterdam`
- `vertical purple neon tube lights along windows`
- `spotlight on girl`
- `view through glass`
- `reflections in window`

### Tested combinations

**Combination 1 (Cum facial + glasses):**
```
Base: FLUX Dev
<lora:Rldstreet:1>
<lora:Cum_Facial_FLUX:1.4>
<lora:cum_on_glasses_FLUX:1.3>
```

**Combination 2 (Bodysuit + cum):**
```
Base: FLUX Dev
<lora:Rldstreet:1>
<lora:Cut-out_Pink_Bodysuit:1>
<lora:Cum_on_face_FLUX:1>
```

**Combination 3 (Amateur + heels):**
```
Base: FLUX Dev
<lora:Amateur_Flux:0.4-0.7>
<lora:Pleaser_Brand_Shoes:0.9-1.2>
<lora:Rldstreet:1>
```

### Works well with
- Cum LoRAs (Cum on Face, CumHereV1)
- Pleaser Brand Shoes
- Amateur Flux
- Clothing LoRAs (bodysuits, PVC)
- Cum on glasses FLUX

### Notes
- Use high weight (2.0) for strong RLD atmosphere
- CFG 7.5 recommended
- Great for window display scenes
- Combines well with cum and clothing LoRAs
- Based on real Office 52 RLD rooms

---

## Beautiful Girls' Faces - PrettyNova

| Parameter | Value |
|-----------|-------|
| **File** | `PrettyNova_Beautiful_Faces.safetensors` |
| **Original filename** | `lora.TA_trained.safetensors` |
| **Civitai** | https://civitai.com/models/1154025/beautiful-girls-faces-nova-of-elegance-and-diversity-without-limits |
| **Trigger word** | `PrettyNova` |
| **Strength** | 0.95-1.0 |
| **Type** | Style / Face Enhancement |
| **Version** | v1.5 |

### Description
Nova LoRA offers a practical tool for generating beautiful and diverse female faces with precise details. Perfect for designers and artists seeking a blend of elegance and variety to meet their project needs. Brings creations to life with a unique touch of beauty.

### Features
- Beautiful, elegant female faces
- Diverse facial features
- Precise facial details
- Works well with various styles

### Sample prompts

**Prompt 1 (Perfection style combo):**
```
ultra detailed and realistic photograph of perfection style PrettyNova ultra realistic photograph highly detailed perfect composition <lora:perfection_style_v2d:1> <lora:PrettyNova_Beautiful_Faces:0.95>
```

**Prompt 2 (Fantasy dancer):**
```
She is the girl I dreamed of, hair platinum-blonde wild pigtails, dancer, Petite, Futuristic, Fantasy forest, Ravenous, You can tell she's not really human, She has every attribute of human in physical form, perfect, Alluring demure attractive beautiful, Action pose, Dancing, Dramatic lighting, Cinematic, Epic composition, PrettyNova, Perfect hand
```

### Keywords
- `PrettyNova` - **REQUIRED** trigger word
- `beautiful face`
- `elegant`
- `diverse`
- `precise details`
- `perfection style`

### Tested combinations

**Combination 1 (Perfection style):**
```
<lora:perfection_style_v2d:1>
<lora:PrettyNova_Beautiful_Faces:0.95>
```

**Combination 2 (Cyber + Hands):**
```
<lora:Cyber_Flux:0.25>
<lora:Hand_F1D_v2:1>
<lora:PrettyNova_Beautiful_Faces:1>
```

**Combination 3 (GarterBelt):**
```
<lora:GarterBeltFlux1.0:1>
<lora:PrettyNova_Beautiful_Faces:1>
```

### Notes
- Use trigger word `PrettyNova` for best face results
- Works great with Detailed Perfection Style LoRA
- Enhances facial beauty without altering overall style
- Good for fantasy, fashion, and portrait work

---

## Naked/Nude for FLUX Sevenof9

| Parameter | Value |
|-----------|-------|
| **File** | `Sevenof9_nude_FLUX_man_woman_v4.safetensors` |
| **Civitai** | https://civitai.com/models/660029/nakednudeforfluxsevenof9nsfw |
| **Trigger word** | None |
| **Strength** | 0.7-1.0 |
| **Type** | NSFW / Nudity Enhancement |
| **Version** | v4 |

### Description
Another naked FLUX LoRA that actually "pushes" nudity better than most alternatives. Works with both men and women. Uses joy-captioning for detailed descriptions. Some poses work better than others - not all anatomically correct (no sex poses).

### Recommended settings
- **Sampler:** EULER with Beta or Forge-Flux with Beta
- **Steps:** At least 25 steps
- **Strength:** 0.7-1.0

### Tips
- For big breasts, use "skinny woman" in the same prompt
- Never trained on "woman on escalator from behind" - works anyway
- Man's part works somewhat
- Use very detailed descriptions (joy-captioning style)

### Sample prompts

**Prompt 1 (Lavender field):**
```
Full length body shot using a nikon d850, 20 year old women has long corkscrew curly cinnamon hair with tightly coiled orange tips curly hair, pale skin, enchanting aqua marine eyes, full lips, cute nose. Naked with perfect large breasts, shaved pussy, medium labia, dark eyeliner and eyeshadow, limbal rings, in a field of lavender, sunny day, highly detailed, fashion photoshoot <lora:nude_FLUX_man_woman_v4:.7>
```

**Prompt 2 (Lake morning light):**
```
A photo of a woman standing in a serene lake, her slim, toned body illuminated by morning lighting that enhances her poised and striking stance. The lake's calm water ripples gently around her legs, adding depth to the scene. Her smooth caramel skin glows in the light, and her long, white-braided hair cascades down her back, complementing her sensuous grey eyes that gaze directly at the viewer. The dramatic angle highlights her perky medium breasts and topless confidence, capturing a natural yet powerful elegance. Subtle details, including pubic hair, add to the realistic portrayal of this full-body view. the background suggest a Wooden footbridge. <lora:nude_FLUX_man_woman_v4:1>
```

**Prompt 3 (Three oriental women ocean):**
```
masterpiece, three stunning oriental women, standing knee-deep in a tropical ocean, wet hair, wet skin, naked, middle girl blonde, detailed nipples, detailed areolae, no underwear, direct flirty gaze, sensual, erotic, posing. Perfect composition, perfect lighting, photorealistic, highly detailed, perfect hands, IMG_5150.CR2, analog film photo, cinematic film still, shallow depth of field, vignette, highly detailed, high budget Hollywood film, moody, epic, film grain, faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, stained, highly detailed, found footage, realist detail <lora:nude_FLUX_man_woman_v4:.7>
```

**Prompt 4 (Escalator from behind):**
```
low angle photo of a nude woman on an escalator, exposing her buttocks, one breast and looking back at camera. <lora:Sevenof9_nude_FLUX_man_woman_v4:1>
```

**Prompt 5 (Artistic dream mood):**
```
alone and abandoned like a lost soul in a blonde woman's body and devoid of will and feelings she dreams of love and hopes for a miracle that will turn back the time and brighten the gloomy clouds detailifier
```

**Prompt 6 (Red couch squatting):**
```
photo of A nude woman with a slender build, small breasts, and a shaved pubic area, squatting on a red couch, legs spread, hands over her head, revealing her vulva. She has long brown hair, fair skin, and a relaxed expression. The background shows a modern, well-lit room with a blue and white color scheme. <lora:Sevenof9_nude_FLUX_man_woman_v4:1>
```

**Prompt 7 (Webcam gamer girl):**
```
Professional photograph of naked webcam girl, sitting on her gamer chair in her bedroom legs spread wide showing off her hairy pussy, smiling at the viewer, a beautiful smile. She has long wavy hair, neon lights, cyberpunk bedroom <lora:nude_FLUX_man_woman_v4:.7>
```

**Prompt 8 (Avant-garde pixie cut):**
```
avant-garde, thrilling, and visually stunning high-fashion photograph of an ethereal, breathtakingly beautiful 20-year-old blonde woman with shorter pixiecut hairstyle, she is styled with bold, artistic flair, her body subtly and tastefully adorned with translucent fabrics, she has exposed nude naked perfect natural breasts boobs, cinematic mysterious and emotionally intense composition, set in a dramatic and abstract environment, a futuristic dreamscape, surreal wilderness, moody and dramatic lighting with deep shadows and sharp contrasts, her expression is fierce yet enigmatic, embodying confidence, vulnerability, and timeless beauty all at once, the highly detailed 16K RAW UHD arousing sensual explicit photo evokes the feeling of a revolutionary art piece, bold and unforgettable and emotionally resonant, pushing the limits of modern erotic visual storytelling, masterpiece by Loyd
```

**Prompt 9 (Grey couch blonde):**
```
photo of A nude adorable girl with a slim waist, huge natural breasts, and a shaved pubic area, squatting on a grey couch, legs spread, hands over her head, revealing her vulva. 18 yo young face, Hourglass figure, She has blond wavy hair, fair skin, normal amount of freckles and moles, a relaxed expression. Round face, small mouth and nose, no strong cheekbones, The background shows a modern, well-lit sun lit room with a red colour scheme. <lora:nude_FLUX_man_woman_v4:1>
```

**Prompt 10 (Norwegian supermodel):**
```
Realistic full-length DLSR photograph of a fit, slim, 175 cm height, extremely beautiful, sexy 22 year old Norwegian supermodel with medium very light blonde sidecut hair and striking light grey-blue eyes, freckles and realistic detailed skin texture including pores, she has a very pale skin color, body freckles and a few moles, a seductive confident look, smiling, nude with small perky breasts, light pink nipples and areolas, a white fashion necklace, a white silk shirt drapes over her shoulders. Her modern, stylish, chic modern Nordic style apartment is in the background, styled with vibrant Nordic accents, reflecting her hip personality, photographed in dramatic golden hour light. cinematic lighting
```

**Prompt 11 (Thong detailed):**
```
the image, a 1 woman, who appears to be in her early twenties, is standing with her full body facing the viewer, looking directly at the camera with a neutral expression. she has short, wavy brown hair and fair skin. her body is slim and athletic, with medium-sized breasts and a small waist. she is wearing a thong that is tied around her waist, accentuating her curves. the thong is red in color and has a knot detail at the front, adding a touch of elegance to her overall appearance. the lighting is soft and natural, casting gentle shadows on her body. the background is a simple beige wall with minimal details, and the overall atmosphere is minimalistic and intimate. breasts with perfect areolas and nipples, pubis, vulva, parted lips
```

**Prompt 12 (Shy Kitty quantum physics):**
```
incredibly realistic high resolution photorealistic UHD photograph of attractive blonde shy 19-year-old Kitty presenting her beautiful body, textless wordless photo capturing the following: why is it important that we seek to solve the mysteries of quantum physics? professional lighting
```

### Keywords
- `naked` / `nude`
- `shaved pussy` / `medium labia`
- `perfect large breasts` / `small perky breasts`
- `detailed nipples` / `detailed areolae`
- `vulva` / `pubic hair`
- `full body shot` / `full length body shot`
- `revealing her vulva`
- `legs spread`
- `topless`

### Tested combinations

**Combination 1 (Jib Mix + Nude):**
```
Checkpoint: Jib Mix Flux v8 - AccentuEight
<lora:nude_FLUX_man_woman_v4:0.7>
```

**Combination 2 (Mystic XXX + Nude):**
```
Checkpoint: FLUX Dev
<lora:nude_FLUX_man_woman_v4:0.9>
<lora:MysticXXX-v7:0.8>
```

**Combination 3 (Ultima + Detailifier):**
```
Checkpoint: FLUX Dev
<lora:Ultima_Flux:0.15>
<lora:Detailifier_Flux:0.55>
<lora:nude_FLUX_man_woman_v4:0.9>
```

**Combination 4 (Extreme Detailer):**
```
Checkpoint: FLUX Dev
<lora:nude_FLUX_man_woman_v4:0.85>
<lora:FLUX_Pro_1.1_Extreme_Detailer:0.25>
```

**Combination 5 (Nude Girls):**
```
Checkpoint: FLUX Dev
<lora:nude_FLUX_man_woman_v4:1>
<lora:Nude_Girls:1>
```

### Compatible checkpoints
- FLUX Dev
- Jib Mix Flux v8 - AccentuEight
- Flux Fusion V2

### Notes
- Use 25+ steps with EULER Beta or Forge-Flux Beta
- Works better than many other nude LoRAs for "pushing" nudity
- Joy-captioning style prompts work well (detailed descriptions)
- Some poses anatomically imperfect - experiment
- No sex poses trained
- For big breasts combine with "skinny woman"
- Works on both men and women

---

## Flux Lustly.ai Uncensored v1

| Parameter | Value |
|-----------|-------|
| **File** | `flux_lustly-ai_v1.safetensors` |
| **Civitai** | https://civitai.com/models/875879/flux-lustlyai-uncensored-v1-nsfw-lora-with-male-and-female-nudity |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | NSFW Unlock / Full Nudity |
| **Version** | v1 (Alpha) |

### Description
Full frontal nudity LoRA supporting both male AND female nudity. One of the few LoRAs that properly handles male anatomy (penis). Reliable on Flux Dev, promising on Schnell (slightly less stable).

### Key features
- Female nudity (full frontal)
- **Male nudity** (including erect penis) - rare feature
- Works on both Dev and Schnell models
- Cinematic and realistic styles

### Planned roadmap
- More diverse poses
- Dynamic interactions between individuals
- Various kinks

### Sample prompts

**Prompt 1 (Game of Thrones fantasy):**
```
Game of Thrones cinematic scene: a Night Watch knight, no pants, with a huge hard cock out, holding a giant sword in his hand, standing next to a naked red haired wildling young female, wearing only red medieval fur coat, flashing her boobs, in a medieval fantasy town, in the snow, huge live dragon flying in the air behind them, breathing fire
```

**Prompt 2 (Snowy forest lingerie):**
```
A young woman with long dark hair stands in a snowy forest, her seductive smile illuminated by soft, warm lighting. She wears a white fur coat with a large hood, draped over her shoulders, and a black lingerie set with thigh-high stockings. One hand rests on her hip, the other on her thigh, as she poses confidently. Snowflakes gently fall, creating a dreamy, ethereal atmosphere. The scene is framed with a shallow depth of field, emphasizing her expression and the intricate details of her attire. The lighting is moody and atmospheric, with a soft, diffused key light from the front and a subtle rim light from behind, casting a gentle glow on her skin. Film grain and a slightly desaturated color grading enhance the cinematic quality.
```

**Prompt 3 (Pool scene with male):**
```
Realistic high quality image of 55 years old blonde german woman, nude, curvy body, small perky boobs, high heels standing and talking with a naked 20 years old male, normal penis on side of interior swimming pool
```

**Prompt 4 (Backyard laundry):**
```
A young Russian woman with blonde hair, seen from the front, standing in a backyard hanging freshly washed clothes on a clothesline. She is nude and has wet body, and sunlight softly illuminates her and the surrounding garden. The scene captures a realistic, everyday moment with attention to natural lighting, textures of the fabric, and subtle expressions on her face. Background includes a wooden fence, green plants, and a few household items typical of a backyard.
```

**Prompt 5 (Beach hidden camera style):**
```
The image is taken from a hidden camera, captures a 19 years old girl, tall, covering boobs with a white towel, skinny body, clean shaved pussy, calm face, standing on sandy public beach. In the background are visible other nude women
```

### Keywords
- `nude` / `naked`
- `full frontal nudity`
- `penis` / `cock` / `hard cock`
- `flashing boobs`
- `shaved pussy`
- `wet body`
- Male/female anatomy terms

### Compatibility
- **Flux Dev** - Reliable, recommended
- **Flux Schnell** - Promising, slightly less stable
- Works with diffusers
- ComfyUI/Forge - community testing in progress

### Use cases
- Couples/male+female scenes
- Male nudity (rare capability)
- Fantasy/cinematic NSFW
- Realistic everyday nudity
- Hidden camera / voyeur style

### Notes
- One of few LoRAs supporting male anatomy
- Alpha version - more features planned
- Works well for cinematic and realistic styles
- Can generate couples/multi-person scenes
- Good for fantasy themes (Game of Thrones style)

---

## Flux Skin Texture

| Parameter | Value |
|-----------|-------|
| **File** | `Flux_Skin_Texture_V2.safetensors` |
| **Civitai** | https://civitai.com/models/1186433/flux-skin-texture |
| **Trigger word** | `skntxtr` (optional) |
| **Strength** | 0.6-1.0 |
| **Type** | Style / Enhancement / Realism |
| **Version** | v2.0 |

### Description
LoRA to improve the infamous plastic look we sometimes get with Flux.1-Dev. Trained on professional beauty shots and general photography where skin texture was prominent. Works with both men and women.

Improves:
- Skin texture and pores
- General photorealism
- Removes "plastic" Flux look
- Adds natural skin detail

### Sample prompts

**Prompt 1 (Malaysian woman bar):**
```
A close up photo of a young (Malaysian woman:1.4), nude side profile, her shirt is open in the middle exposing her small breasts for the viewer, she has a look of surprise and embarrassment on her face, winter day in the city bar, moody lighting, erotic photograph, sexy revealing pose, she is (topless:1.4), her hair is short and messy, expressive facial features and body language, she is leaning against the wall with her chest pressed out <lora:Flux_Skin_Texture_V2:1>
```

**Prompt 2 (Cowgirl on horse):**
```
A gorgeous Caucasian cowgirl with a supermodel physique, riding a horse bareback, back arched, with perfect posture, looking seductively at the viewer with a seductive, sultry and aroused yet confident expression. She has long, wavy blonde hair blowing in the wind, striking honey colored eyes, and luscious plump lips. Her skin is sun-kissed with visible tan lines, with a perfect detailed skin texture style, realism, detailed. She has a slim, toned body with defined abs and strong shoulders. She is wearing only a cowboy hat and boots, and a black and white striped shirt that is wide open, fully revealing her perky bare breasts and hard nipples, skntxtr, PerkyCTits, and a neatly trimmed triangle patch of pubic hair above her pussy. The setting is a dynamic ranch on a hot sunny day with a bright blue sky at the golden hour. 8k resolution, sharp focus.
```

**Prompt 3 (Arctic portrait with glasses):**
```
A hyper-realistic portrait. cinematic upper body photo, Dramatic Shadow with a Bold Look, A brunette woman, her hair neatly styled with subtle highlights that catch the light softly, beam of light on her, She wears round, silver-framed glasses perched elegantly on the bridge of her nose, excellent dynamic range, makeup, natural youthful glow. White modern clothes, straight white hair cascading down her shoulders, Her eyes are almond-shaped and striking blue, accentuated by thick eyeliner. arctic circle snow drift, snowing, snowflakes, cold, chill bumps, approaching perfection, dynamic, highly detailed, smooth, sharp focus, intricate details, shallow depth of field, vignette, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, (skin texture:1.4)
```

### Keywords
- `skntxtr` - optional trigger word
- `skin texture`
- `detailed skin`
- `realistic skin`
- `natural skin`
- `pores`
- `photorealistic`

### Tested combinations

**Combination 1 (PixelWave + Eros):**
```
Checkpoint: PixelWave FLUX.1-dev 03
<lora:Flux_Skin_Texture_V2:1>
<lora:nudes_v06:1>
```

**Combination 2 (CyberRealistic + Perky Tits):**
```
Checkpoint: CyberRealistic Flux v2.5
<lora:Flux_Skin_Texture_V2:0.8>
<lora:Perky_C_Tits:1>
```

**Combination 3 (XLabs Realism + Makeup):**
```
Checkpoint: FLUX Dev
<lora:XLabs_Flux_Realism:1>
<lora:Esc_Makeup:0.6>
<lora:Flux_Skin_Texture_V2:0.6>
<lora:Iphone_quality_FLUX:0.8>
```

### Compatible checkpoints
- FLUX Dev
- PixelWave FLUX.1-dev
- CyberRealistic Flux v2.5

### Notes
- Fixes the "plastic" Flux look
- Works with both male and female subjects
- Trained on professional beauty photography
- Best at 0.6-1.0 strength
- Combine with realism LoRAs for best results
- Optional trigger word `skntxtr` for stronger effect

---

## Eros v06

| Parameter | Value |
|-----------|-------|
| **File** | `eros_v06.safetensors` |
| **Civitai** | https://civitai.com/models/1063613/eros?modelVersionId=1364408 |
| **Trigger word** | `topless`, `nude`, `naked` |
| **Strength** | 0.8-1.0 |
| **Type** | CONCEPT / Nudity |

### Description
Reproduce human nudity with high accuracy and controllability on various body types, ethnicities, styles without shifting the base model too much.

### Recommended settings
- Steps: 20-40 (weird pose = more steps)
- CFG: 2.5-5
- Samplers: Euler/Simple, Beta, DDEIS/DDIM, Flux realistic

### Supported features
- **Poses:** standing, sitting, squatting, resting, lying, crouching, on all fours
- **Camera angles:** selfie, from behind, profile, top view, rear view, high/low angle
- **Close-up:** chest, lower body, pussy, breasts, buttocks, feet, hands
- **Breasts sizes:** tiny to enormous
- **Ethnicity:** Caucasian, Asian, Korean, Japanese, African, European, etc.
- **Skin color:** pale, fair, light, olive, medium, dark
- **Pubic hair:** hairy, trimmed, shaved

### Sample prompts

**Prompt 1 (Mountain nude):**
```
A full body photo of an 18 year old Russian woman, nude side profile, she has small pointy breasts which sag gently, she is standing on top on a mountain with the clouds below her, moody lighting, (she is wearing thigh-high stockings:1.2), erotic photograph, sexy revealing pose <lora:eros_v06:1>
```

**Prompt 2 (Forest path):**
```
The image shows a young woman standing on a rocky path in a forest. She is wearing a lower down gray dress, exposing her breasts, with a belt around her waist. She has shoulder-length blonde hair and is looking directly at the camera with a serious expression. <lora:eros_v06:1>
```

---

## FLUX Naked Female

| Parameter | Value |
|-----------|-------|
| **File** | `flux_naked_female_v1.safetensors` |
| **Original filename** | `b13313a36212c9637d8cc8f50a2ab96b.safetensors` |
| **Civitai** | https://civitai.com/models/938709/flux-naked-female |
| **Trigger word** | None |
| **Strength** | 1.0-1.45 |
| **Type** | CONCEPT / Nudity |

### Description
Generate realistic naked females with pubic hair. Different faces, hair colors, ethnicities.

### Known issues & solutions
- **Face out of frame:** Add hair color (Redhead, Brunette, Blonde)
- **Full body shot:** Add hair color + shoe type (e.g., "Full body shot, Woman, Blonde, white shoes")

### Sample prompts

**Prompt 1 (Crowd):**
```
Crowd of naked women, blonde, redhead, brunette, on street, celebrating, happy, celebrating party <lora:flux_naked_female_v1:1>
```

**Prompt 2 (Beach group):**
```
three (fully naked:2.5) women bright skin, very bright blond hair, different hairstyles, unique faces, scandinavian look, walking along a beach holding hands, (tall:1.4), slender, happy faces, smiling at viewer, (pubic hair, large labia clearly visible:1.2) <lora:flux_naked_female_v1:1.45>
```

---

## FLUX NSFW LoRA

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_NSFW.safetensors` |
| **Civitai** | https://civitai.com/models/742936/flux-nsfw |
| **Trigger word** | None |
| **Strength** | 0.8 |
| **Type** | CONCEPT / NSFW |

### Description
General NSFW content generation for FLUX.

### Sample prompts

**Prompt 1 (Bed):**
```
Naked woman, on bed, legs spread wide, pussy visible <lora:FLUX_NSFW:0.8>
```

**Prompt 2 (Korean woman):**
```
A 21-year-old South Korean woman walking confidently among the high-rise buildings, in 8K-quality. She has a great body, with huge breasts, a very slim waist. She is wearing a loose suit open enough to show her chest and navel. <lora:FLUX_NSFW:0.8>
```

---

## Flux-Cameltoe NSFW

| Parameter | Value |
|-----------|-------|
| **File** | `Flux-CamelToe.safetensors` |
| **Civitai** | https://civitai.com/models/748078/flux-cameltoe-nsfw |
| **Trigger word** | `cameltoe`, `v-shape` |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Clothing effect |

### Description
Cameltoe effect for clothed models. Works with Flux DEV and Flux Schnell.

### Usage tips
- For clothed models: use this LoRA alone
- For NSFW/Nude: combine with other NSFW LoRAs

### Sample prompts

**Prompt 1 (Pool):**
```
a full body photo of butifull young women next to the pool, she has thight white leggings on pulling a cameltoe in a v-shape. Her light pink tank top is partially covered with her long blond hair. <lora:Flux-CamelToe:1>
```

**Prompt 2 (Forest bikini):**
```
a photo of tow girls, one blond and one redhead doing a topless bikini photoshoot in the forrest at a green sunny spot. both of them have cameltoe pulling a v-shape on their random color bikini bottoms. <lora:Flux-CamelToe:1>
```

### Good combinations
- JK Perfect Breasts for Flux
- Cameltoe Panties (Flux)

---

## NSFW FLUX LoRA (AiArtV)

| Parameter | Value |
|-----------|-------|
| **File** | `nsfw_flux_lora_v1.safetensors` |
| **Civitai** | https://civitai.com/models/655753/nsfw-flux-lora |
| **Trigger word** | `AiArtV` |
| **Strength** | 0.5-1.0 |
| **Type** | CONCEPT / NSFW |

### Description
NSFW content generation for FLUX (experimental). Trained on 600 images, 18k steps.

### Recommended settings
- Sampler: euler
- Steps: 20
- CFG: 1

### Sample prompts

**Prompt 1 (Portrait):**
```
AiArtV, woman, open mouth, blue hair, earrings, teeth, choker, tongue, tongue out, mole, black choker, makeup, portrait, realistic, long tongue <lora:nsfw_flux_lora_v1:0.8>
```

**Prompt 2 (Changing room):**
```
18yo, light brown hair, long hair, low body fat, skinny, abs, medium breast, nude, naked, stunning girl, showing her pussy, in a changing room, petite, skinny, AiArtV <lora:nsfw_flux_lora_v1:0.8>
```

### Good combinations
- MysticXXX LoRA
- Huge cumshot facial LoRA
- Breast size slider

---

## Cum and play with FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `Semen_flux_lora_v2.safetensors` |
| **Civitai** | https://civitai.com/models/664287/cum-and-play-with-flux |
| **Trigger word** | None (describe effect in detail) |
| **Strength** | 0.35-1.05 |
| **Type** | CONCEPT / Cum effect |

### Description
Facial cumshot/semen effects for FLUX. Best results when describing the facial in detail.

### Best practice
Describe the effect explicitly:
> her face is covered with semen which is a thick, viscous white substance. The substance is smeared across her forehead, cheeks, nose, and chin, with some droplets on her lips.

### Sample prompts

**Prompt 1 (Brazilian):**
```
Raw, 8k, dramatic lighting, a profetonal photo of a cute young brasilian woman after giving a blowjob, large amounts of thick white liquid semen are in lines on her her face after a facial cumshot, shes kneeling on the floor looking up at the viewer <lora:Semen_flux_lora_v2:1>
```

**Prompt 2 (College selfie):**
```
A selfie photo of a female college student in her dorm room. She is taking a messy and gooey realistic semen facial cumshot across her face. She is dripping with cum, and a shy, surprised smile appears on her face as she blushes. <lora:Semen_flux_lora_v2:1>
```

### Good combinations
- Huge cumshot facial LoRA
- Boreal-FD (Boring Reality)
- Real Nipples and Areola Textures

---

## Cum On Feet - FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `Cum_On_Feet_-_FLUX_r1.safetensors` |
| **Civitai** | https://civitai.com/models/1583418/cum-on-feet-flux |
| **Trigger word** | `cum on feet`, `cumshot on feet` |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Cum effect |

### Description
Cum on feet effect for FLUX. Works better with just feet and no face.

### Sample prompts

**Prompt 1 (Closeup):**
```
closeup, rainbow nail polish, cum on feet, feet, cumshot on feet <lora:Cum_On_Feet_-_FLUX_r1:1>
```

**Prompt 2 (Chair):**
```
Asian woman sitting on a chair, showing feet to the viewer, cum on feet, feet, cumshot on feet <lora:Cum_On_Feet_-_FLUX_r1:1>
```

### Good combinations
- Slingback Pumps LoRA
- POV Sockjob, Shoejob, Footjob [Flux]

---

## Cum Facial / Cum on face - FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `Cum_Facial_-_FLUX-000009.safetensors` |
| **Civitai** | https://civitai.com/models/1240792/cum-facial-cum-on-face-flux |
| **Trigger word** | `cum facial`, `cum on face`, `bukkake` |
| **Strength** | 1.0-1.25 |
| **Type** | CONCEPT / Cum effect |

### Description
Cum facial/bukkake effects for FLUX. More visible effect than other cum LoRAs.

### Sample prompts

**Prompt 1 (Tanktop):**
```
Asian girl with a yellow tanktop with the text "Cumslut" and a black choker. On her knees with fishnet stockings. Looking up with open mouth, smiling, tongue out. Cum facial, cum on face, bukkake <lora:Cum_Facial_-_FLUX-000009:1>
```

**Prompt 2 (Pigtails):**
```
Cute blonde girl with braces and pigtails, facial, cum on face, cum facial. bukkake. thumb up, smile <lora:Cum_Facial_-_FLUX-000009:1>
```

**Prompt 3 (Group):**
```
4 smiling Asian girls next to each other, with huge cum facial, alot of cum on face, massive cumshot, bukkake <lora:Cum_Facial_-_FLUX-000009:1.25>
```

### Good combinations
- XLabs Flux Realism LoRA
- Perfect Full Round Breasts & Slim Waist
- UltraRealistic Lora Project

---

## Facial cum massive FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `facialcummassive_FLUX.safetensors` |
| **Original filename** | `9cd3fddff95e898ea5534c4eb6a6509b.safetensors` |
| **Civitai** | https://civitai.com/models/866273/facial-cum-massive-flux |
| **Trigger word** | `FCLHGE`, `Facial cum massive`, `cum on nose`, `cum on forehead`, `cum drips`, `bukkake` |
| **Strength** | 1.0-2.0 (up to 2 without subject LoRA) |
| **Clipskip** | 1 |
| **Type** | CONCEPT / Cum effect |

### Description
Massive facial cum effect LoRA for FLUX. Uses trigger word FCLHGE for activation. Produces heavy cum effects on face including nose, forehead, and dripping effects. Can use higher strength (up to 2.0) when not combining with subject LoRAs.

### Sample prompts

**Prompt 1 (Basic facial):**
```
FCLHGE, Facial cum massive, woman with cum on face, cum on nose, cum on forehead, cum drips <lora:facialcummassive_FLUX:1>
```

**Prompt 2 (Bukkake):**
```
FCLHGE, bukkake, massive facial, cum drips down face, cum on forehead, cum on nose <lora:facialcummassive_FLUX:1.5>
```

### Keywords
- `FCLHGE` - **TRIGGER WORD**
- `Facial cum massive`
- `cum on nose`
- `cum on forehead`
- `cum drips`
- `bukkake`
- `massive facial`

### Notes
- Use trigger word `FCLHGE` for best results
- Can increase strength to 2.0 when not using subject LoRAs
- Use Clipskip 1
- Combines well with other NSFW LoRAs

---

## Cumbubbles FLUX

| Parameter | Value |
|-----------|-------|
| **File** | `Cumbubbles_FLUX.safetensors` |
| **Civitai** | https://civitai.com/models/1406617/cumbubbles-flux |
| **Trigger word** | `cumbubbles` |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Cum effect |

### Description
Simple LoRA to add cum bubbles effect on mouth and nose. Creates frothy bubble effect with cum.

### Sample prompts

**Prompt 1 (Bukkake with bubbles):**
```
bukkake and cum everywhere, a woman with a white substance like cum on her face, closed eyes and raised foreskin, a woman with a lot of white stuff on her face and eyes, penis with cum fontain, gagging cum, face drenched in cum, a woman with a lot of white stuff on her face and nose like cum, cum drenched lips. <lora:Cumbubbles_FLUX:1> cumbubbles. bubbles on mouth and nose.
```

**Prompt 2 (Festival foam):**
```
cumbubbles The popular Japanese actress is enjoying the festival at a winter festival in Tokyo. She is caught in a joyous moment spraying herself with artificial snow made from cum foam. cum foam falls on her face, covering her cheeks, nose and lips, giving her a surprised yet amused look. She has her eyes closed and is smiling, as cum foam creates a frothy effect around her mouth and nose. The scene is vibrant and captures the atmosphere of the festival.
```

### Keywords
- `cumbubbles` - **TRIGGER WORD**
- `bubbles on mouth and nose`
- `cum foam`
- `frothy effect`

### Compatible checkpoints
- FLUX Dev
- AsianBeautyFlux

### Notes
- Use trigger word `cumbubbles` for activation
- Creates bubble/foam effect with cum
- Works well with bukkake prompts

---

## Cum on face - Non-Face Altering

| Parameter | Value |
|-----------|-------|
| **File** | `cumonfacelorav2.safetensors` |
| **Civitai** | https://civitai.com/models/858262/cum-on-face-flux-non-face-altering |
| **Trigger word** | `COF` |
| **Strength** | 0.8-1.0 |
| **Type** | CONCEPT / Cum effect |
| **Version** | v2.0 |

### Description
Cum on face LoRA that does NOT alter character faces - allows using with favorite character LoRAs. V2 retrained using De-distilled Flux with recaptioned dataset for better efficiency.

### Important: Distilled vs Dedistilled
- **Dedistilled models**: Full cum effect, works extremely well
- **Distilled models**: Less cum, need to max CFG and use all reinforcing words
- Recommended: Use Dedistilled models for best results

### Key features
- Non-face altering - safe for character LoRAs
- Works on face, body, mouth with Dedistilled
- V2 improved over V1

### Reinforcing words (important for Distilled)
```
she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue
```

### Alternative reinforcing words
```
white sticky cum on face, white sticky semen on face, white sticky sperm on face, face covered with white sticky cum
```

### Sample prompts

**Prompt 1 (Emo punk girl):**
```
<lora:cumonfacelorav2:1>COF, Above view of 18 years-old smiling emo punk girl with pink colored hairstyle, tongue out, above view, she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue. The scene happens in a rave during night
```

**Prompt 2 (Christmas market):**
```
Close-up Portrait of a very beautiful smiling woman called Elle wearing a super nice knitted white hat, knitted white gloves and a winter outfit, is standing on a christmas market and holding a Norwegian mulled wine mug in her hands. She has clear sticky cum with white reflections over her face. Her face is covered in cum after an enormous ejaculation. Christmas mood, love, fantasy, dreaming, cinematic <lora:cumonfacelorav2:1>
```

**Prompt 3 (Motorhead fan):**
```
A closeup photo of a gorgeous dark haired fan girl of the heavy metal band Motörhead is kneeling, mouth open, eyes closed. The point of view is from above her head, her face is towards the viewer. Her face is completely covered by cum, She has clear sticky cum with white reflections over her face, cum on nose, cum on lips, cum on chin, dripping cum, cum on eyes, cum on cheeks, cum on teeth, cum on forehead, cum dripping from cheeks, cum dripping from chin, cum on hair, front view, view from above, cum dripping from lips. Her face is almost completely covered by semen. Huge amount of sperm. She has a black "Motörhead Overkill" fan t-shirt on <lora:cumonfacelorav2:1>
```

**Prompt 4 (Satin maid):**
```
satin maid uniform, short-sleeve satin dress, white peterpan collar, white satin apron, skirtlift, a woman is lifting her dress to show her crotch area and vagina, COF, She has clear sticky cum with white reflections over her face, cum on nose <lora:cumonfacelorav2:1>
```

**Prompt 5 (Cumzilla style - works well):**
```
cumface woman with lots of white, thick, gooey cum all over and covering her face, cheeks, hair and forehead. The cum coats her face in a thick layer <lora:cumonfacelorav2:1>
```

### Keywords
- `COF` - **TRIGGER WORD**
- `clear sticky cum with white reflections`
- `dripping cum`
- `cum on forehead/cheeks/chin/eyes/lips/tongue`
- `cum dripping from chin`
- `face covered with cum`

### Prompting tip
Link reinforcing words with "with":
- GOOD: "Young woman face with sticky cum on face"
- BAD: "Young woman, sticky cum"

### Tested combinations
- MysticXXX LoRA
- Missionary POV LoRA
- Character LoRAs (Elle from Rick & Morty, etc.)
- Satin Maid LoRA
- Skirt Lift Concept LoRA
- Desi Espresso LoRA

### Compatible checkpoints
- FLUX Dev
- Fluxmania
- De-distilled Flux (RECOMMENDED)

### Notes
- Use trigger word `COF`
- Does NOT alter character face - safe for character combos
- Much better results with Dedistilled models
- For Distilled: max CFG, use ALL reinforcing words
- Can generate cum on other body parts with Dedistilled

---

## Bukkake / Realistic Cum Facial Flux

| Parameter | Value |
|-----------|-------|
| **File** | `Bukkake_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/1346321/bukkake-realistic-cum-facial-flux |
| **Trigger word** | None (use keywords) |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Cum effect |

### Description
Creates realistic bukkake facials for Flux. Works with single or multiple subjects.

### Sample prompts

**Prompt 1 (Single woman):**
```
A sexy woman is drenched in cum, she is smiling happily, bukkake, cum facial, cum on face, thick cum on face, semen on face, semen facial <lora:Bukkake_Flux:1>
```

**Prompt 2 (Two women):**
```
2 sexy women are drenched in cum, they are smiling happily, They don't look alike, bukkake, cum facial, cum on face, thick cum on face, semen on face, semen facial <lora:Bukkake_Flux:1>
```

**Prompt 3 (Ahegao bunny):**
```
a cinematic high def picture of a skinny slim cute cumface Woman age21, ahegao, ahegao face, tongue out, pink hair, bunny ears, bunny costume, (White thick choker with a pink plastic heart), lots of white sticky semen on her face, streams of semen pouring down her face and dripping down, cum on forehead cum on face, cum on tongue, pink neon light background, bedroom <lora:Bukkake_Flux:1>
```

### Keywords
- `bukkake`
- `cum facial`
- `cum on face`
- `thick cum on face`
- `semen on face`
- `semen facial`
- `drenched in cum`

### Notes
- No trigger word needed - use descriptive keywords
- Works with multiple subjects
- Good for realistic bukkake scenes

---

## Universal Cum Enhancer

| Parameter | Value |
|-----------|-------|
| **File** | `Cum_as_a_Concept_V1.safetensors` |
| **Civitai** | https://civitai.com/models/1494887/universal-cum-enhancer-cum-as-a-concept-flux |
| **Trigger word** | `cum on floor`, or just `cum` |
| **Strength** | 0.6-1.5 |
| **Type** | CONCEPT / Cum enhancer |
| **Version** | v1.0 |

### Description
Universal cum enhancer that boosts other cum LoRAs by adding better texture and shape. Designed to work alongside other NSFW/cum LoRAs for improved results. The word "cum" is the key trigger - add descriptors to match your needs.

### Key features
- Enhances other cum LoRAs
- Adds texture and shape to cum effects
- Works best with Flux Dev
- Adjustable strength (0.6-1.5)

### Known limitations
- Sometimes generates less cum than expected - tweak prompts
- Quality may dip if prompts stray too far from trained data
- Works best when combined with other cum LoRAs

### Sample prompts

**Prompt 1 (Snapchat style with Noisify):**
```
a low light 2015 snapchat quality realistic iphone photo taken from above with her mouth slightly open, in a dark bedroom with the lights off, her face is covered in long strands of stringy white semen and cum. There are thin sticky strands stretching from her chin to small tiny globs on her neck and chest. the strands of cum and semen are shot diagonally across her face, in almost parallel strands. There is no top or clothing visible. She looks like she has been roughed up, her hair is messy and her makeup is smudged, her eyeliner is running. low light, noise, film grain. The photo is taken from above, as if the viewer is positioned on top of her as she is lying down. <lora:Cum_as_a_Concept_V1:1>
```

**Prompt 2 (Close-up with realism):**
```
An image of a close-up shot of a Caucasian woman's face with cum on it. She is wearing a camisole. Her eyes are closed, and her expression is neutral. The background is minimal, showing a light-colored floor and part of a wall. The lighting is bright, suggesting daytime. <lora:Cum_as_a_Concept_V1:1> <lora:flux_realism_lora:1>
```

### Negative prompt (recommended)
```
ugly face, mutated hands, low res, blurry face, pumped body, athletic body, black and white, text, big head
```

### Keywords
- `cum` - **KEY TRIGGER**
- `cum on floor`
- `stringy white semen`
- `strands of cum`
- `sticky strands`

### Tested combinations
- Noisify LoRA
- Character LoRAs (Anna, etc.)
- Flux Realism LoRA

### Notes
- Use as enhancer WITH other cum LoRAs for best results
- Adjust weight 0.6-1.5 depending on desired intensity
- Stick close to trained prompts for quality

---

## FLUX Gone Wild

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_Gone_Wild.safetensors` |
| **Original filename** | `FLUX_Gone_Wild-000003.safetensors` |
| **Civitai** | https://civitai.com/models/1188155/flux-gone-wild |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | STYLE / NSFW Enhancement |
| **Version** | v1.0 |

### Description
One of the best FLUX NSFW models available. Enhances overall NSFW quality and explicit content generation. Works great with Instagram model and character LoRAs.

### Key features
- High quality NSFW generation
- Works well with other LoRAs
- Good for explicit content (masturbation, nudity)
- Snapchat/Instagram selfie style support

### Sample prompts

**Prompt 1 (Bathroom mirror selfie):**
```
Perfect Eyes, perspective, score_9, score_8_up, score_7_up, masterpiece, high quality, realistic, detailed, (photorealistic:1.4), navel focus, Natural skin), 1girl, 25year old, slut, gorgeous girl, (cute girl), tanned, makeup, long haircut straight hair, ((black thick hair:1.7)), suntan skin, (hourglass body), natural breast, athletic girl, big lips, soft lips, big mouth, nipples, supermodel body, slutty face, big eyes, black eyes, gorgeous perfect face, big natural breasts, (((((naked:1.2, breasts revealed, nipples revealed)))) ((((A Snapchat selfie with pink iPhone in the mirror of a beautiful woman with tanned skin taking a selfie while masturbating in the bathroom, fingering vagina, pussy revealed)))). The overall mood is playful and theatrical, with a focus on performance art. The lighting emphasizes the performer, creating a dynamic and engaging composition. masterwork, masterpiece, best quality, detailed, depth of field, high detail, best quality, very aesthetic, 8k, dynamic pose, depth of field, dynamic angle, instagirl <lora:FLUX_Gone_Wild:1>
```

**Prompt 2 (With RNAT and Instagram):**
```
Perfect Eyes, perspective, score_9, score_8_up, score_7_up, masterpiece, high quality, realistic, detailed, (photorealistic:1.4), navel focus, Natural skin), 1girl, 25year old, slut, gorgeous girl, (cute girl), tanned, makeup, long haircut straight hair, ((black thick hair:1.7)), suntan skin, (hourglass body), natural breast, athletic girl, big lips, soft lips, big mouth, nipples, supermodel body, slutty face, big eyes, black eyes, gorgeous perfect face, big natural breasts, (((((naked:1.2, breasts revealed, nipples revealed)))) ((((A Snapchat selfie with pink iPhone in the mirror of a beautiful woman with tanned skin taking a selfie while masturbating in the bathroom, fingering vagina, pussy revealed)))). masterwork, masterpiece, best quality, detailed, depth of field, high detail, best quality, very aesthetic, 8k, dynamic pose, depth of field, dynamic angle, instagirl, Ona_model, RNAT, Mirror selfie, holding cellphone <lora:FLUX_Gone_Wild:1>
```

### Keywords
- `score_9`, `score_8_up`, `score_7_up`
- `masterpiece`, `high quality`, `realistic`
- `photorealistic`
- `naked`, `breasts revealed`, `nipples revealed`
- `Snapchat selfie`, `mirror selfie`
- `fingering vagina`, `pussy revealed`
- `instagirl`

### Tested combinations
- Real Nipples and Areola Textures (RNAT) LoRA
- Instagram Girl LoRA
- Ona Instagram Model LoRA
- Mirror Selfie FLUX LoRA

### Compatible checkpoints
- FLUX Dev
- Jib Mix Flux

### Notes
- One of the best NSFW LoRAs available
- Works great with Instagram/selfie style LoRAs
- Good for explicit masturbation scenes
- Combine with RNAT for realistic nipples

---

[← Back to Index](INDEX.md)

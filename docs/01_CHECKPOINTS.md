# 1. Checkpoints

[← Back to Index](INDEX.md)

Base models and diffusion checkpoints for FLUX-based generation.

---

## Table of Contents
- [FLUX1 Dev GGUF F16](#flux1-dev-gguf-f16)
- [GGUF FLUX.1 Dev Q8](#gguf-flux1-dev-q8)
- [Flux ArtFusion 4-steps](#flux-artfusion-4-steps)
- [Real Horny Pro V3](#real-horny-pro-v3)
- [Jib Mix Flux](#jib-mix-flux)
- [8 Steps CreArt-Hyper-Flux-Dev](#8-steps-creart-hyper-flux-dev)
- [Flux1-DedistilledMixTuned v4.0](#flux1-dedistilledmixtuned-v40)
- [Flux.1 Dev Asian FP16](#flux1-dev-asian-fp16)
- [Fluxcstasy v1](#fluxcstasy-v1)
- [FluxUnchained NF4](#fluxunchained-nf4)
- [SlimNudeGirls Flux FP8 NSFW](#slimnudegirls-flux-fp8-nsfw)
- [Flux Real NSFW | Lluminartis](#flux-real-nsfw--lluminartis)
- [Fluxed Up NSFW Checkpoint (Nunchaku INT4)](#fluxed-up-nsfw-checkpoint-nunchaku-int4)

---

## FLUX1 Dev GGUF F16

| Parameter | Value |
|-----------|-------|
| **File** | `flux1-dev-F16.gguf` |
| **Location** | `models\unet\` |
| **Size** | 22.1 GB |
| **Civitai** | https://civitai.com/models/flux1DevGGUFF16_f16 |

### Sample prompt
```
pretty girl sitting in the grass next to a lake
```

---

## GGUF FLUX.1 Dev Q8

| Parameter | Value |
|-----------|-------|
| **File** | `ggufFLUX1DevQ8ModelUsed_v10.gguf` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/1452850?modelVersionId=1642703 |
| **Name** | GGUF FLUX.1 Dev Q8 model used in InfiniteYOU Workflow |

---

## Flux ArtFusion 4-steps

| Parameter | Value |
|-----------|-------|
| **File** | `fluxArtfusion4Steps_v12.safetensors` |
| **Location** | `models\unet\` |
| **Size** | 16 GB |
| **Civitai** | https://civitai.com/models/641214/flux-artfusion-4-steps |
| **Recommended settings** | 4 steps, CFG 1, Sampler LCM |

### Description
Very fast model based on Flux.1 Schnell. Batch of 4 images in less than a minute on 4070 12GB. Version 1.2 has better colors, details and contrast.

### Sample prompts

**Prompt 1 (Shower):**
```
naked woman is in the shower, she is singing and dancing
```
Settings: Steps: 4, CFG scale: 1, Sampler: LCM

**Prompt 2 (Roman bath):**
```
There are wonderful studio lights and various reflections,her thin face is very delicate,
18yo beautiful cute girl with blushing and female orgasm.skin naturally showing the texture of blood vessels and detailed pale skin,the sidelight outlines her sexy body curves,
cowboy shot,--,<lora:AGirl_type3_F1:0.3>,<lora:BustyWomen-v3:0.3>,(full body:1.2),sagging breasts apart,puffy long nipples,
--,
<lora:body harness lingerie_f1:1.08>,she wearing white body harness lingerie and lace footwear,
--,She is in the center of the picture. Apart from her, all the other people in this ancient Roman open-air bath are ancient Roman men. It is beautifully decorated, the pool is steaming, and in the distance is the magnificent Roman Colosseum.
```

### Recommended workflows
- GonzaLomo Flux Refiner Workflow v3.0
- Flux.1 S Workflow with Lightning upscale (for portraits)

---

## Real Horny Pro V3

| Parameter | Value |
|-----------|-------|
| **File** | `realHornyProV3_realHornyProV2NF4.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/684924/real-horny-pro-v3?modelVersionId=968460 |
| **Type** | Unet (NF4) - requires VAE and CLIP |

### Description
Specialized NSFW model with improved realism and LoRA compatibility. NF4 version requires less VRAM and runs faster, but with some quality loss. "Asian Cuties" version prefers Asian women.

### Sample prompts

**Prompt 1 (Nun painting):**
```
oil painting of a a lecherous nun in an open cassock, full body, wide angle, A 25-year-old hot beautiful seductive alluring sensual woman with perfect face and perfect well-endowed hourglass figure prefect breast red lips, white skin, black hair, muscle body, (green glowing shiny eyes), mysterious cathedral in background, looking at the center camera, painting style, beautiful detailed intricate insanely detailed, <lora:Eldritch_Paint_Sketch_for_Flux_1.0.5:1.2>
```

---

## Jib Mix Flux

| Parameter | Value |
|-----------|-------|
| **Files** | `jibMixFlux_v8Accentueight.safetensors`, `jibMixFlux_v85Consisteight.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/686814/jib-mix-flux |
| **Type** | Diffusion Model (FLUX merge) |
| **Versions** | v12, v8.5 ConsistEight, v8 AccentuEight (installed), v8-Flash SVDQuant-4bit, v7.8, v7.2, v6.1, v5 |

### Description
FLUX Dev trained on SDXL dataset with merged LoRAs, correcting anatomy censorship and excessive bokeh/blurred backgrounds.

**Version differences:**
- **v12** - Merge with SRPO model, cleaner looking, may need grain LoRAs for amateur look
- **v8.5 ConsistEight** - More consistent outputs
- **v8 AccentuEight** - Much better skin texture than v7, without bad Flux Lines of v6. NSFW slightly lacking - use Jibs Flux Nipple Fix LoRA
- **v8-Flash SVDQuant-4bit** - Super fast (5s on 3090, 2.5s on 4090, 0.8s on 5090 at 10 steps). Requires Nunchaku
- **v7.8 Clear Text Focus** - Better readable text, less nipple flashing through clothes
- **v7.2 Pixel Heaven** - Fixes Flux Lines, better drawn/concept ability, tones down freckles
- **v6.1 Real Pix Fixed** - Most detailed backgrounds, fixed hand issues

### Recommended settings
- **Guidance scale:** 2.5-3.5 (or 2 for version 2)
- **Sampler:** dpmpp_2m
- **Scheduler:** Sgm_uniform, Beta or Custom Stigmas
- **CFG:** 2.8
- **Steps:** 8-14 (with Hyper), 20-40 (standard)
- **Resolution:** ~1280x1344 recommended

### Sample prompts

**Prompt 1 (Swedish woman bar):**
```
An incredibly sexy blond 21 year old Swedish woman in a laced latex corset and very high heels sits in a provocative pose sat on a wooden bar stool, legs spread wide, knees up, busy bar at night in the background, a fan ruffles her hair.
```

**Prompt 2 (Beach volley group):**
```
naked playing beach volley group of s, realistic, water, blonde, huge perky breasts, skinny, slender, narrow waist, naked
```

**Prompt 3 (Webcam gamer girl):**
```
Professional photograph of naked webcam girl, sitting on her gamer chair in her bedroom legs spread wide showing off her hairy pussy, smiling at the viewer, a beautiful smile. She has long wavy hair, neon lights, cyberpunk bedroom <lora:nude_FLUX_man_woman_v4:.7>
```

**Prompt 4 (Leather punk):**
```
Fallon leans against a brick wall tagged with local band flyers and graffiti, rainwater darkening her long black hair. She's nude beneath an oversized leather jacket, fishnets torn around her thighs. Her Doc Martens stomp cigarette butts into the concrete. Vintage erotic art, RAW photo, Kodak Portra 800
```

**Prompt 5 (Futuristic nude crowd):**
```
(completely nude:1.1), futuristic interior, crowded room, brutalist architecture, concrete plates, very hazy room, indoor waterfall, indirect light, diffuse light, sharp photo, professional photography, 4k, vogue photography, feet style, perfect feet, soles, toes, BREAK. , (background is filled with nude athletic females:1.1), female fitness models in the background, BREAK. 1 is very skinny, she is resting on a glowing cube, athletic, (very lean:1.2), she spreads her legs, leg up, tan, beautiful vagina, very long hair, medium breasts, slim waist, erect nipples
```

**Prompt 6 (Spanish influencer):**
```
sexy naked coach influencer tanned spanish 1 on a terrace, brunette, tan lines, skinny, narrow waist, wide hips, large breasts
```

**Prompt 7 (Ice skating):**
```
A ((totally naked short 18 year old blonde slim women)), ice skating at an indoor skate rink, shot on Kodak Portra 400 film with a Hasselblad 500C camera, moody lighting, ultra fine detail, cinematic. zavy-rghflt, (flat chest) smiling at the camera, shaved pussy.
```

**Prompt 8 (Forest birch trees):**
```
Full-length shot of a young woman girl with a slim figure, small breasts and visible pubic hair, standing against a backdrop of birch trees in a sunlit forest. She is wearing only a black short-sleeved top that partially covers her torso and white sneakers. The background features lush greenery and diffused sunlight shining through the trees.
```

**Prompt 9 (Tattoo fantasy):**
```
RAW full body analogue photo of a beautiful 25 years old woman, ((20yo Yakuza)), ultra-seductive, full tattooed, top soft light, highly detailed, (((Oleg Dou photography style))), full body photo, ((Nude)), (((tattoos))), film grain, Fujifilm XT3, 8k uhd
```

**Prompt 10 (Lace lingerie couch):**
```
A photo-realistic shoot from the side about a young woman in black lace lingerie lounging on a white couch in a softly lit room. a 20-year-old woman with fair skin and long, wavy brown hair, lying on her stomach, wearing a black lace bra and matching panties. she has blue eyes and is looking directly at the camera with a sultry expression.
```

### Tested combinations

**Combination 1 (Sevenof9 Nude):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:nude_FLUX_man_woman_v4:0.7>
```

**Combination 2 (Mystic XXX):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:MysticXXX-v4:1>
```

**Combination 3 (MoreFace + Hands):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:MoreFaceV2-lora:1>
<lora:Hand_F1D_v2:1>
```

**Combination 4 (PussyDiffusion):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:MoreFaceV2-lora:1>
<lora:pussydiffusion-f1:1>
```

**Combination 5 (Curvy Natural):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:Curvy_Natural:0.6>
```

**Combination 6 (Small Breasts + Hands):**
```
Checkpoint: Jib Mix Flux v8 AccentuEight
<lora:breast-size2:-1> or <lora:breast-size2:-2>
<lora:Hand_F1D_v2:1>
```

### Negative prompt template
```
Orange peel skin, hairy face, ((Red spots on face)), ((red freckles)), ((shiny Skin)), ((bad hands)), ((cartoon)), ((bokeh)), Deviantart, jpeg, (worst quality, low quality), (watermark, signature), (blur, blurry, grainy), morbid, ugly, mutated, poorly lit, bad shadow, cropped, out of frame, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D, 3D Game:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth:1.3), ((bokeh))
```

### Notes
- NSFW capabilities may be reduced in some versions - use Jibs Flux Nipple Fix LoRA
- SVDQuant-4bit version requires Nunchaku for fast generation
- v8 AccentuEight has best skin texture balance
- v6 has most detailed backgrounds
- For hand issues at low step count, increase steps or use Hyper Flux LoRA at <0.10
- Works great with Sevenof9 Nude, MysticXXX, MoreFace LoRAs

---

## 8 Steps CreArt-Hyper-Flux-Dev

| Parameter | Value |
|-----------|-------|
| **File** | `8StepsCreartHyperFlux_v26HyperDevFp8Unet.safetensors` |
| **Location** | `models\diffusion_models\` |
| **Civitai** | https://civitai.com/models/699688?modelVersionId=930403 |
| **Type** | Unet FP8 - requires CLIP L, T5XXL and VAE |
| **Base** | ByteDance Hyper-SD merged with multiple LoRAs |

### Description
Ultimate version of Hyper Flux Dev merged with ByteDance Hyper 8 steps LoRA. Includes merges with MoreFace LoRA, SkinDetails LoRA, and Real-lora for enhanced realism. Excellent for NSFW with improved versatility and artistic capabilities.

### Version history
- **v4.0:** Better realism, versatility, nudity
- **v3.0:** More LoRA merged, better lighting, fixed vertical frame
- **v2.7:** Merged with MoreFace LoRA - more realism, versatility
- **v2.6:** Merged with SkinDetails LoRA - more realism

### Recommended settings
- **Steps:** 8-10 (recommended: 10)
- **Guidance:** 3-3.5
- **Sampler:** Euler
- **Scheduler:** Beta
- **Optional:** Detail Daemon for enhanced details

### Sample prompts

**Prompt 1 (Leather outfit):**
```
a woman looking at viewer sexy pose full body shot wearing black leather outfit with thigh-high boots, Crimped hair, Gray, Swimwear High-waisted bikini with a ruffle top and matching bottoms, establishing shot, cafeteria, (masterpiece best quality ultra-detailed best shadow amazing realistic picture), Fujifilm XT3, -color- lighting
```

**Prompt 2 (Kyoto lingerie):**
```
split lighting, Canon RF, (masterpiece best quality ultra-detailed best shadow amazing realistic picture), ancient city of Kyoto is a treasure trove of traditional Japanese culture and architecture, with temples, shrines, and tea houses, dressed in a lace shift dress with a high neckline and pearl accessories, Jet black, Wavy hair, a woman in a lingerie sexy pose
```

**Prompt 3 (Cinematic female):**
```
a beautiful cinematic sexy female, Long ponytail, Ginger hair, dressed in a crochet bikini top with high-waisted shorts and sandals, professionally shot, Skate park, detailed masterpiece most beautiful artwork in the world Ultrarealistic, Nikon d850, back-light, <lora:morefaceV2-lora:1>
```

**Prompt 4 (Korean/Nigerian figure):**
```
A provocative [korean|nigerian] figure draped in sheer fabrics, illuminated by soft, warm light, surrounded by deep crimson hues; the atmosphere exudes sultry allure, desire, and tantalizing intimacy, capturing an erotic, captivating moment.
```

### Notes
- Works great with MoreFace LoRA (already partially merged)
- GGUF Q4_0 and BnB NF4 versions also available
- ComfyUI NF4 node: https://github.com/DenkingOfficial/ComfyUI_UNet_bitsandbytes_NF4
- HuggingFace Hyper-SD: https://huggingface.co/ByteDance/Hyper-SD

---

## Flux1-DedistilledMixTuned v4.0

| Parameter | Value |
|-----------|-------|
| **File** | `flux1_v40Fp8.safetensors` |
| **Location** | `models\diffusion_models\` |
| **Civitai** | https://civitai.com/models/941929/flux1-dedistilledmixtuned |
| **Type** | Diffusion Model (FP8) - requires CLIP and VAE |
| **Base** | SRPO + Krea + Dev merged |

### Description
V4.0 Pure base model integrating realism from SRPO, artistry from Krea, with excellent texture and LoRA compatibility. Extremely realistic and delicate - maintains details even at 8M pixels. Optimized for portraits, oriental faces, composition, and lighting.

### Key features
- Excellent NSFW/SFW LoRA compatibility
- Great prompt restoration - use LLM-enhanced or structured prompts
- Good artistic expression and style diversity
- Optimized for oriental face shapes and ethnicity diversity

### Recommended settings
- **Sampler:** deis+simple or euler+beta
- **More noise:** ddim/dpm_2/dpmpp_2+beta/sgm_uniform
- **More detail:** heunpp2+ddim_uniform
- **Steps:** 20-30
- **Upscaler:** UltimateSDUpscale or TTP
- **Film effects:** add LUT (35mm/AGAF/Kodak)

### Sample prompts

**Prompt 1 (Black girl interior):**
```
A medium close range photograph taken at a low angle, with the camera angle below the level of the human eye, taken sideways. The background is a dim modern interior, with light coming from the right side, soft and slightly dim. The subject is a black girl with big waves, high skull, long hair, sitting by a gray bed, wearing black diamond patterned black yoga pants, legs straight and raised, facing left, slightly looking up at the camera.
```

**Prompt 2 (Beach influencer):**
```
photo of a beautiful attractive woman, pretty face, Instagram influencer, topless, natural breasts, nipples, narrow waist, pussy, wide hips, posing sexy on beach, tropical ocean, towering palms background, sensual vibes, youthful look, epic dramatic sunset, natural lighting, sharp focus
```

**Prompt 3 (Portrait photography):**
```
portrait photography of Professional DSLR photo, photorealism, ultra-detailed, 8K resolution photography, capturing emotion, personality, flattering lighting, professional, engaging, compelling composition
```

### Negative prompt template
```
bad image, bad photo, bad hand, bad finger, logo, Backlight, worst quality, low resolution, distorted, twisted, watermark
```

### Notes
- V4.0 is the recommended version (pure base model)
- V3.0-Krea has weaker portrait LoRA compatibility
- Works well with film effect LoRAs (35mm, Kodak)

---

## Flux.1 Dev Asian FP16

| Parameter | Value |
|-----------|-------|
| **File** | `flux1DevAsian_v10FP16.safetensors` |
| **Location** | `models\diffusion_models\` |
| **Civitai** | https://civitai.com/models/672618/flux1devasian |
| **Type** | Diffusion Model (FP16, ~22GB) - requires VAE, CLIP, T5XXL |
| **Base** | FLUX.1 Dev + Asian LoRA merged |

### Description
Experimental merged model with retrained Asian LoRA. Optimized for Asian facial features with aesthetic tweaks. FP16 version offers better quality and fewer limb issues than FP8. Compatible with other Flux.1 Dev LoRAs (may need weight adjustments).

### Key features
- Good variety of Asian facial features
- Aesthetic optimizations for portraits
- Compatible with Flux Dev LoRAs
- FP16 = better precision, less limb distortion

### Known issues
- Slightly increased overall brightness
- Minor limb accuracy degradation (less in FP16)
- May affect composition compared to original Dev

### Requirements
- VRAM: >=24GB recommended for FP16
- Requires: VAE (ae.sft), CLIP (clip_l), T5XXL

### Recommended settings
- **Sampler:** euler
- **Scheduler:** simple
- **Steps:** 25
- **Guidance:** 4
- **ModelSamplingFlux:** max_shift 1.2, base_shift 0.5

### Sample prompts

**Prompt 1 (Nightclub):**
```
A dimly lit urban nightclub scene with a glowing red and pink neon sign. A confident woman stands at the bar, leaning on a stool, wearing an oversized black shirt open to reveal lace lingerie. The room is filled with haze and colorful lights, highlighting the modern and edgy atmosphere.
```

**Prompt 2 (Erotic scene):**
```
Asian girl with thick thighs, long straight hair, smooth, glowing skin, and a soft, dreamy expression, is being embraced tenderly by a dark-skinned man in a dimly lit room. Neon lights cast an ethereal, warm, and colourful glow, creating an erotic haze. Shallow depth of field, subtle colour grading. Warm, Intimate Atmosphere.
```

**Prompt 3 (Pinup bar):**
```
raw photo, from below, front side view, 1girl, 20 yr old, stunningly beautiful girl, pinup pose, sexy round eyes, realistic long black hair, hair parted in center, lounge, jazz club, bar, nice black dress, up skirt view, legs slightly spread, slender, petite, indigo lighting, at night time, cowboy shot, sitting at bar, shadows
```

### Notes
- FP16 is recommended over FP8 for better quality
- FP8 (~11GB) available for lower VRAM systems
- May need to adjust LoRA weights when combining

---

## Fluxcstasy v1

| Parameter | Value |
|-----------|-------|
| **File** | `fluxcstasyV1Fp16Fp8NF4_fp16V10.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/1310785/fluxcstasy-v1-fp16-and-fp8-and-nf4-and-gguf-q8q6q5 |
| **Type** | Diffusion Model (FP16) - requires VAE and CLIP |
| **Versions** | FP16, FP8, NF4, GGUF Q8/Q6/Q5 |

### Description
Fluxcstasy produces great quality images with sexy-looking breasts and moves away from the original FLUX model's boring artificial faces. It greatly corrects cleft chins, making faces realistic and beautiful. The model has an excellent analog photography feel with nice skin textures. Versatile for sensual erotic photos, casual everyday images, and artistic photography.

### Key features
- Corrects cleft chin issues
- Realistic, beautiful faces
- Analog photography aesthetic
- Nice skin texture details
- Versatile style range (erotic to artistic)

### Sample prompts

**Prompt 1 (Snow cabin):**
```
Thin atthletic build. Beautiful girl wearing a cropped unzipped sweatshirt, open jacket, cleavage. Glamourous hair. Detailed eyes and face. Soft features. Morning in the snow. Standing in front of large cabin with lots of windows. Photorealistic, Natural atmospheric lighting, intricate details, 35mm photograph, film, professional, 4k visuals, highly detailed, elegant, studio quality.
```

**Prompt 2 (College girl):**
```
Thin atthletic build. 1girl. Wearing knee high socks, short plaid skirt, crop top shirt and open sweatshirt. Standing in a college dorm hallway. Beautiful detailed eyes. Subdued eyeliner. Light make-up. Studio lighting. Photorealistic, Natural atmospheric lighting, intricate details, 35mm photograph, film, professional, 4k visuals, highly detailed, elegant, studio quality.
```

**Prompt 3 (Shopping blonde):**
```
Blonde Hair, thin, athletic. 1girl. Cute. Beautiful detailed eyes eyes. Full lips. Hair styled in a sideswept casual. Wearing track pants and tight short sleeved cropped hoodie. Athletic body. Shopping at an outdoor market. Photorealistic, Natural atmospheric lighting, intricate details, 35mm photograph, film, professional, 4k visuals, highly detailed, elegant, studio quality.
```

**Prompt 4 (Latex luxury):**
```
pale young woman, wearing only latex overknee socks and latex overarm gloves, in a luxurious mansion, very extravagance
```

**Prompt 5 (Vintage portrait):**
```
The image is a vintage portrait of a young woman. She stands with her torso slightly turned to the side, with her arms crossed over her chest. Her very small breasts are exposed for all to see, and her nipples are visible. She has square glasses. She has a serious expression on her face, and her eyes are looking straight at the camera. The background is completely dark, which makes the woman the center of the image. very skinny woman, naked breasts, flat chest, areola
```

**Prompt 6 (B&W artistic):**
```
The image is a black and white photograph of a young woman sitting on floor, legs folded under, hands behind head. She is completely naked, with her body facing the camera and her arms resting on her head. Her hair is styled in loose waves and she is looking directly at the camera with a serious expression. The background is a plain wall, and the overall mood of the image is somber and contemplative. A Astounding Tour de Force, high-contrast black-and-white portrait of a breathtaking young very skinny woman, style by Harry Callahan, her realistic skin texture captured in exquisite detail, her naked breast breasts, hard breasts, small breasts, dark areola, beautiful_and_aesthetic Small and rounded nipples, her realistic Elegant taupe eyes with a touch of shimmer and a wine-colored lip captured on her face. Her hairstyle is Voluminous Half-Updo. Shot with a Fujica 35 Automagic on SFL UN54 film, pushed to ISO 3200 for pronounced grain film and deep, inky blacks. The scene using a special High Angle (from behind) view Close shot and Detailed. The Diffused Lighting evokes a timeless, classic Candlelit, reminiscent of Time-worn photo but with a Industrial, hyper-detailed cinematic aesthetic. The macro lens with eye focus. The f/1.4 aperture creates an ethereal Clear Scene, softly blurring the background into dreamy Out-of-Focus Area, newton, corbjin.
```

**Prompt 7 (Sci-fi cosplay):**
```
Still Frame, cinematic, dramatic, Artistic Image: Caroline Munro as Stella Star in the movie Starcrash 1978, wearing a black leather bikini with studded accessories, walking on a futuristic alien planet, oil painting, expressionism style, rich colors with high contrast, close-up shot, inspired by Edvard Munch, 85mm lens, aperture f/4.0, ISO 400, shutter speed 1/125, high resolution.
```

### Notes
- Multiple versions available: FP16 (best quality), FP8 (smaller), NF4, GGUF Q8/Q6/Q5
- Great for analog/film photography style
- Works well with detailed natural language prompts

---

## FluxUnchained NF4

| Parameter | Value |
|-----------|-------|
| **File** | `fluxunchainedNF4_fluxunchainedV11NF4.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/663307/fluxunchained-nf4 |
| **Type** | Diffusion Model (NF4) - requires VAE and CLIP |
| **Original author** | SocialGuitarist |

### Description
NF4 version of FluxUnchained model by SocialGuitarist. Optimized for lower VRAM GPUs (works well on 3080 10GB). NF4 format provides good inference speed with quality similar to Q4/Q5 but faster execution.

### Required components
To use this UNET-only version, add the following encoders:
- `clip_l.safetensors`
- `t5xxl_fp16.safetensors` or `t5xxl_fp8_e4m3fn.safetensors`
- `ae.safetensors`

### Sample prompts

**Prompt 1 (Indian latex):**
```
Photo of an 30 year old (Indian woman)+++. She is petite and pretty. She is only earing a big latex collar around her neck and thigh high (latex boots)++. On her chest is a big tattoos with text "FUCK DOLL" in clear black font. She is kneeling down with her back straight and arms behind her back, looking up towards the camera pov. Her body is completely visible in the photo and the background is a indian home.
```

**Prompt 2 (Transparent saree):**
```
Photo of a 30 year old Indian++ female with a fair complexion with long hair. Female is wearing a (transparent black latex saree)++ with a latex pallu draped over her naked boobs and nipples. She has multiple (queen of spades) tattoos on her body. The photo is a shot in an indian home, and portrait is shot from thigh up.
```

### Keywords
- `Indian woman`
- `latex`
- `transparent`
- `tattoos`
- `pov`

### Notes
- UNET-only version saves disk space
- Requires external text encoders (CLIP, T5XXL) and VAE
- Optimized for GPUs with limited VRAM
- Faster than Q4/Q5 GGUF with similar quality
- All Flux.1 Dev license terms apply

---

## SlimNudeGirls Flux FP8 NSFW

| Parameter | Value |
|-----------|-------|
| **File** | `slimnudegirlsFluxFp8Nsfw_v099ReleaseCandidate.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/855390/slimnudegirls-flux-fp8-nsfw-release-candidate |
| **Type** | Unet (FP8) - requires VAE and CLIP |
| **Version** | v0.9.9 Release Candidate |

### Description
Specialized checkpoint for slim nude/topless girls. Merge of Flux base model with specially trained LoRAs. Created because the author couldn't find a Flux checkpoint creating slim nude girls with tiny breasts. This model creates erotic images but no sexual pictures.

### Required components
- Flux VAE: `ae.safetensors`
- Flux Text Encoders: `clip_l.safetensors` and `t5xxl_fp8_e4m3fn.safetensors`

### Key features
- Optimized for slim body types with tiny breasts
- Trained on poses: standing, sitting, kneeling, squatting, lying, on all fours
- Supports fully/mostly shaved or hairy vagina
- Better compatible with riding, sitting and lying positions (v0.9.9)

### Version hints (v0.9.9)
- This version might be a bit strong
- When using LoRAs, you might need to increase LoRA strength by 10%

### Prompting tips
- Use `slim nude/topless girl` or `slim nude/topless woman`
- For difficult situations, enforce with:
  - `with visible vagina/breasts`
  - `with fully/mostly shaved/hairy visible vagina`
  - Use weight syntax: `(nude:1.3)`

### Sample prompts

**Prompt 1 (Garter belt cafe):**
```
front view, gartbt, (slim nude girl) with black stockings and garters and high heels, standing in a crowded street cafe, serving drinks, necklace, short brunette wild hair style <lora:GarterBeltFlux1.0:1.2>
```

**Prompt 2 (Mall walking):**
```
(slim 25yo nude girl) in boots walking in crowded mall. holds hair. mkpotail. long blonde hair. (visible fully shaved pussy) <lora:MakePonytailFlux:1.1>
```

### Recommended use cases
- Slim female models with cars/automotive
- Fashion model style workflows
- Nude photography with petite body types
- LoRA sample generation (non-nude training images)

### Tested combinations
```
<lora:GarterBeltFlux1.0:1.2>
<lora:MakePonytailFlux:1.1>
```

### Notes
- Early preview version - further enhancements planned
- Not designed for sexual content, only erotic/nude
- Great for workflows with slim models (automotive, fashion)
- When combining with LoRAs, increase LoRA strength by ~10%

---

## Flux Real NSFW | Lluminartis

| Parameter | Value |
|-----------|-------|
| **File** | `fluxRealNSFW_v20.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/1316180/flux-real-nsfw-or-lluminartis |
| **Type** | BASE MODEL (FLUX) |
| **Version** | 2.0 |

### Description
Version 2 is not very good for explicit NSFW poses, but it does produce good nudes.

### Recommended settings
- Steps: 20
- CFG: 1.0
- Sampler: euler
- Scheduler: normal
- Denoise: 1.00

### Sample prompts

**Prompt 1 (Beach):**
```
A confident completely naked young woman standing ankle-deep in crystal-clear ocean waves, wearing fashionable oversized sunglasses. Her natural, genuine smile reflects joy and contentment. Golden sunlight creates a warm backlight effect, highlighting her silhouette. <lora:tiny_waist_flux_v1:0.25> <lora:Big_Boobs:0.2>
```

**Prompt 2 (Minimalist room):**
```
A nude, slender woman with small breasts and visible genitals stands confidently, hands behind her head, in a modern, minimalist room.
```

**Prompt 3 (Evening dress):**
```
A stunning 45-year-old woman, dressed in a short cold shoulder velvet dress with a deep cleavage trimmed with fur, matching elbow gloves and knee-high strapped stiletto obsidian glass platform sandals, leans on the railing of the embankment and looks erotique straight at the viewer. The soft light of the setting sun creates an atmosphere of coziness.
```

### Notes
- Version 2 works well for nudes but not explicit poses
- Good for natural-looking erotic photography

---

## Fluxed Up NSFW Checkpoint (Nunchaku INT4)

| Parameter | Value |
|-----------|-------|
| **File** | `fluxedUpFluxNSFWCheckpoint_51INT4.safetensors` |
| **Location** | `models\unet\` |
| **Civitai** | https://civitai.com/models/1967499/fluxed-up-flux-nsfw-checkpoint-nunchaku-int4fp4 |
| **Type** | BASE MODEL (FLUX) |
| **Version** | 5.1 INT4 (Nunchaku) |
| **Format** | Nunchaku NVFP4 |

### Description
Nunchaku INT4 version of the Fluxed Up 5.1 model. Some quality issues - legs can get weird sometimes. Can use with turbo LoRA for faster generation.

### Recommended settings

**With Turbo LoRA:**
- Sampler: Euler/Simple
- Steps: 8

**Without Turbo LoRA:**
- Sampler: DPM++ 2M/Beta
- Steps: 30

### Sample prompts

**Prompt 1 (Coffee shop):**
```
A professional realistic photo of a woman. The woman has K-beauty straight brows, Rounded (diamond:0.25) face, (puffy cheek:0.3), happy, auburn Illusion Ponytail hair, red eyes, overlined lips with subtle outline, portrait shot, upper body. The woman is happy and smiling. A (solo:1.4) woman slouching, wrapping her arms around her legs, surprised, legs together, pussy peek, holding one leg. Viewed from above, warm coffee shop window seat
```

**Prompt 2 (Shadow Muse penthouse):**
```
Shadow Muse, A soft-focus, hyper-realistic shot of a 29-year-old supermodel with ivory skin, ash-blonde hair in a low bun, green eyes sultry with intent. She wears a sheer black lace bustier, revealing her busty figure, paired with a velvet choker. Leaning against a frosted glass partition in a dimly lit penthouse, city lights blurred, her gaze piercing, ultra-detailed, mysterious allure.
```

**Prompt 3 (Winter Eastern European):**
```
photo of a young white eastern european woman, mid-20s. She has straight, shoulder-length dark hair with purple highlights and big perky breast. Light snowflakes rest on her hair, suggesting a cold, snowy day. Blue-gray eyes are prominent, minimal makeup visible. She is outdoors, urban background with a brick wall and bare trees, winter season is evident, her face very underexposed and shadowed
```

**Prompt 4 (Redhead gamer):**
```
Redhead AI Babe, Rua Aingeal, Red Angel, Freckles, Curly Red Hair, Ginger, Green Eyes, Irish Redhead, Fair Skin Tone, Silky red Hair, Light green Eyes, brushing long wavy thick red hair with a hairbrush, large pale bare breasts, athletic, fit, colorful striped sweatpants, topless, Scr33nLum, indoors, shadows, dim monitor lighting, old bedroom, looking at gaming monitor, sitting on gaming chair, bored, messy room, Interior, relaxing, brushing hair, lounging
```

### Known issues
- Legs can sometimes look weird
- Reduced smoothing grids for faster processing

### Notes
- Nunchaku INT4 format for lower VRAM usage
- Works with turbo LoRA for 8-step generation
- Good for NSFW content generation

---

[← Back to Index](INDEX.md)

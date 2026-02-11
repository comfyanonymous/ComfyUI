# Persephone [Flux NSFW/SFW]

[Back to CHECKPOINTS Index](INDEX.md)

## Overview

A very strong NSFW Flux checkpoint with a distinctive structured prompt format using "YOUR CONTEXT" and "YOUR PHOTO" sections. Supports both SFW and NSFW content with diverse ethnicity, camera model specification, and weighted concept mixing for blending multiple scene descriptions.

## File Information

- **Filename:** persephoneFluxNSFWSFW_20FP8.safetensors
- **Location:** models\unet\
- **Format:** SafeTensors FP8
- **Size:** ~11.6 GB
- **Version:** 2.0
- **Base Model:** Flux.1 Dev
- **Creator:** 6tZ

## Statistics

- **Downloads:** 12,088
- **Rating:** 661
- **Tips:** 1,001,610
- **Comments:** 15
- **NSFW Level:** 31/32
- **Score:** ⭐⭐⭐

## Links

- **Civitai:** https://civitai.com/models/1775002/persephone-flux-nsfwsfw

## Model Type

UNet-only NSFW/SFW model - no VAE or CLIP baked in. Requires separate VAE (flux_vae.safetensors) and CLIP (Dual Clip Loader). Works in ComfyUI with standard UNETLoader.

## Description

Persephone is a strong NSFW Flux checkpoint by creator 6tZ that supports both SFW and NSFW content generation. The model uses a distinctive two-section prompt structure with "YOUR CONTEXT:" defining the photographer style/mood and "YOUR PHOTO:" or "YOUR PHOTOGRAPH:" describing the actual scene. Every prompt specifies a real camera model (Sony, Canon, Nikon, Kodak, Olympus, etc.) and uses `(film grain:1.2)` as a consistent quality enhancer.

Key differentiators include:
- **YOUR CONTEXT / YOUR PHOTO prompt structure** - separates artistic intent from scene description, allowing precise control over photographic style
- **Camera model prompting** - every prompt specifies a real camera model, influencing the rendering aesthetic (DSLR vs mirrorless vs vintage film cameras)
- **Diverse ethnicity support** - European, Asian, African, Barbudan, Surinamese, Malaysian, Syrian, and more
- **Weighted concept mixing** - uses `(content:weight)` format to blend multiple scenes or concepts in a single prompt (e.g., 0.6/0.4 splits for combining two different NSFW scenarios)

Each sample image on Civitai contains the used workflow for easy reproduction.

## Key Features

- Full NSFW and SFW capability in a single checkpoint
- Structured prompt format (YOUR CONTEXT / YOUR PHOTO)
- Camera model specification influences output aesthetic
- Weighted concept mixing with (content:weight) syntax
- Diverse ethnicity and nationality support
- Film grain aesthetic built into prompt structure
- Multiple photography styles (Hollywood, fashion, retro, aerial, amateur, etc.)
- FP8 format fits 12 GB VRAM
- 1M+ community tips indicating high quality

## Prompt Structure

Persephone uses a unique two-section prompt format that separates the photographer's artistic intent from the scene description.

### Section 1: YOUR CONTEXT (Photographer Style)

Defines the photographer persona, mood, and artistic style. This section controls the overall aesthetic.

```
YOUR CONTEXT: You are a [photographer type] who [style description].
Your photographs exhibit [quality attributes]. YOUR PHOTO:
```

**Example photographer types used in samples:**
- Hollywood filmmaker making a high-budget film
- iPhone camera amateur photograph (candid, unposed, real life)
- Photographer who appreciates classic Kodak Portra film aesthetic
- Cinematographer who works in dark movies (film noir, Ektachrome palette)
- Photographer who prefers serene and tranquil images (minimalist, high contrast)
- Photographer inspired by orthochromatic and colorized films
- Aerial photographer using wide-angle drone cams
- Color specialist photographer (teal and orange palette)
- Fashion photographer creating ultra-bright high-key photos

### Section 2: Subject Description

Describes the person, clothing, age, ethnicity, hair, and accessories.

```
The photo is a high-resolution [style] photo of [age/ethnicity] woman with [hair color] [hair style] hair.
She is wearing [clothing]. [optional: glasses/accessories]
```

**Photography style modifiers:** epic, cinematic, modern, retro, professional, realistic, fantasy, amateur, stylish

### Section 3: NSFW Content / Pose

The actual scene description, actions, poses, and NSFW content. Supports weighted mixing:

```
(Scene description A:0.6). (Scene description B:0.4)
```

This allows blending two different concepts in a single generation with controllable emphasis.

### Section 4: Scene & Camera

```
The scene is captured in a [location] with detailed background details in the environment.
The award-winning photo is a close-up (film grain:1.2) photo shot with a [camera model].
The photograph is highly detailed.
```

### Camera Models Used in Examples

- Sony a6400 Mirrorless Camera
- Sony a6600 Mirrorless Camera
- Sony a7 III Mirrorless Camera
- Sony a7 IV Mirrorless Camera
- Sony a7R IV Mirrorless Camera
- Nikon D3500 DSLR Camera
- Nikon Z5 Mirrorless Camera
- Nikon Z7 II Mirrorless Camera
- Nikon Z fc Mirrorless Camera
- Nikon F2 Camera
- Canon EOS 5D Mark IV DSLR Camera
- Canon EOS 90D DSLR Camera
- Canon EOS R6 Mirrorless Camera
- Canon EOS RP Mirrorless Camera
- Canon EOS Rebel T8i DSLR Camera
- Kodak No 3A Folding Pocket Camera
- Kodak 35 Camera
- Kodak PIXPRO AZ241 16.15MP Digital Camera
- Panasonic Lumix G7 Mirrorless Camera
- Panasonic Lumix GH5 II Mirrorless Camera
- Fujifilm X-H2S Mirrorless Camera
- Olympus OM-1 Camera
- Olympus XA2 Camera
- Contax G1/G2 Camera
- Yashica T4 Camera
- Polaroid SX-70 Alpha Camera
- Pentax K1000 Camera
- Mamiya 7 II Camera
- Graflex Century Graphic 2x3 Camera
- Lubitel 2

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Sampler** | DPM++ 2M SDE (dpmpp_2m_sde) or dpmpp_2m_sde_gpu_beta |
| **Scheduler** | Beta |
| **Steps** | 32 |
| **CFG/Guidance** | 4.0 (Distilled CFG Scale) |
| **VAE** | flux_vae.safetensors (not baked in) |
| **CLIP** | Not baked in - use Dual Clip Loader |

## Version History

| Version | Format | Size | Downloads |
|---------|--------|------|----------:|
| 2.0 FP16 | FP16 | 23.2 GB | 456 |
| **2.0 FP8** | **FP8** | **11.6 GB** | **197** |
| 1.2 FP16 | FP16 | 23.2 GB | 2,401 |
| 1.1 FP8 | FP8 | 11.6 GB | 1,138 |
| 1.1 FP16 | FP16 | 23.2 GB | 917 |
| 1.0 FP16 | FP16 | 23.2 GB | 2,180 |
| 1.0 Q8-GGUF | GGUF | 12.4 GB | 1,092 |
| 1.0 Q6_K | GGUF | 9.6 GB | 2,557 |
| 1.0 Q4_K_S | GGUF | 6.6 GB | 1,150 |

**Bold** = installed version

## VRAM Requirements

| Format | VRAM | File Size | Installed |
|--------|:----:|:---------:|:---------:|
| FP16 | 20-24 GB | 23.2 GB | - |
| **FP8** | **12 GB** | **11.6 GB** | **2.0** |
| Q8-GGUF | 12-14 GB | 12.4 GB | - |
| Q6_K GGUF | 10-12 GB | 9.6 GB | - |
| Q4_K_S GGUF | 8-10 GB | 6.6 GB | - |

## Sample Prompts

### SFW / Softcore

**Prompt 1 (Hollywood filmmaker - snowy village):**
```
YOUR CONTEXT: You are a Hollywood filmmaker making a high-budget film.
The photo is a high-resolution epic cinematic modern photo of  an adult 40y-o european woman with light blonde Retro Pin-Up Waves hair. She is wearing Racer stripe pants and Surf suit.
in a dynamic sexy pose, posing for the viewer
The scene is captured in a in a snowy village square, where lanterns light cobblestone paths location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a6400 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 272541744110476, Distilled CFG: 4.0

**Prompt 2 (Kodak Portra - marshland):**
```
YOUR CONTEXT: You are a photographer who appreciates the classic aesthetic of Kodak Portra film.
The photo is a high-resolution  professional photo of  an adult 20y-o european woman with dark brown Runway-Ready Blowout hair. She is wearing Henley shirt and pvc stockings. She is wearing glasses.
in a dynamic sexy pose, posing for the viewer
The scene is captured in a marshland with tall grasses location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Kodak No 3A Folding Pocket Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 746946502567479, Distilled CFG: 4.0

**Prompt 3 (Serene minimalist - arcade):**
```
YOUR CONTEXT: You are a photographer who prefers serene and tranquil images. Your photographs exhibit attractive and spicy content, where everyone is sexy and provocative, with high contrast and clean, minimalist compositions that emphasize empty space to highlight negative space. YOUR PHOTO:
The photo is a high-resolution  retro photo of   african woman with blonde Classic Side Braid hair. She is wearing Spaghetti strap top and metal chain suspender stockings. She is wearing glasses.
in a dynamic sexy pose, posing for the viewer
The scene is captured in a in a noisy arcade with game machines location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Olympus OM-1 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 514922542576473, Distilled CFG: 4.0

**Prompt 4 (Aerial drone - vintage subway):**
```
YOUR CONTEXT: You are an aerial photographer who enjoys using wide-angle drone cams. Your photographs exhibit attractive and spicy content, where everyone is sexy and provocative, with panoramic scenes captured from afar, high up with elevated perspectives and intense colors. YOUR PHOTO:
The photo is a high-resolution  realistic photo of   african woman with copper Tinsel-Threaded Braid hair. She is wearing latex encasement suit. She is wearing glasses.
in a dynamic sexy pose, posing for the viewer
The scene is captured in a in a vintage subway station, surrounded by echoes and quiet after the last train location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS 5D Mark IV DSLR Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 631812368514865, Distilled CFG: 4.0

**Prompt 5 (Fashion high-key - fortune teller):**
```
YOUR CONTEXT: You are a fashion photographer who creates ultra-bright photos for luxury productions. Your photograph showcases attractive and spicy content, where everyone is sexy and provocative, with a high-key composition shot flooded with soft studio lighting. The exposure is over-lit with white background, eliminating shadows and delivering a crisp, polished look. Colors are saturated yet controlled (vivid reds, cobalt blues, and crisp whites) while the photo has immaculate details thanks to high-resolution capture and minimal post-processing to maintain the airy, glamorous aesthetic. YOUR PHOTOGRAPH:
The photo is a high-resolution epic fantasy photo of   caucasian woman with gray Windswept Glam Curls hair. She is wearing real-shackle prison latex tights.
A mysterious, raven-haired fortune teller sits cross-legged on a vibrant, ornate rug, surrounded by a dazzling array of peculiar trinkets and relics at a mystical flea market that appears to exist in a dreamlike state, where a maze of antique mirrors seems to stretch on forever, reflecting fragmented realities and distorted echoes of the fortune teller's own enigmatic presence, her eyes gleaming with an otherworldly intensity as she gazes into a crystal ball, her long fingers weaving an intricate pattern in the air as if conducting an unseen symphony, the mirrors around her casting a dizzying kaleidoscope of reflections that seem to ripple and undulate like the surface of a pond, each one revealing a different facet of her mystical persona, from the wispy tendrils of smoke curling from the tip of her ornate, gemstone-tipped cigarette holder to the delicate, filigreed patterns that dance across the surface of her dark, velvet cloak, which appears to be woven from the very shadows themselves, as the fortune teller's gaze draws the viewer deeper into the heart of the maze, where the boundaries between reality and illusion begin to blur and dissolve.
The scene is captured in a in a small music practice room, upright piano, stool, and acoustic panels location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Yashica T4 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 15913303501499, Distilled CFG: 4.0

**Prompt 6 (Amateur - zen kimono seductive):**
```
The image is a high-resolution amateur photography of  asian woman with dark brown Satin-Smooth Locks hair. She is wearing zen buddhist kimono - a minimalist, fitted kimono made of soft, breathable fabric, often featuring simple patterns that embrace calm colors, designed to be practical and elegant while honoring Zen principles of simplicity and mindfulness. and genital-stapled latex thigh-highs. She is wearing glasses.
seductive pose
The scene is captured in a calm lake shore at dawn with a pair of wooden canoes location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon Z7 II Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 369004350339131, Distilled CFG: 4.0

**Prompt 7 (Retro - Surinamese nipple reveal):**
```
The image is a high-resolution retro photography of a 35y-o Surinamese woman with Green straight  hair. She is wearing mini skirt and Muscle tee.
She is wearing a Orange, off-the-shoulder blouse that is partially pulled down, exposing her chest and nipples.
The scene is captured in a romantic rooftop garden, with a wrought-iron table and a view of the stars above location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a6600 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 329025924348307, Distilled CFG: 4.0

**Prompt 8 (Retro - film noir lofi nude):**
```
The image is a high-resolution retro photography of  african woman with Red Loose Spiral Curls hair. She is wearing t-shirt and runproof fetish stockings.
film potrait , girl surrounded with bubbles detailed, dramatic lighting shadow (lofi, analog-style)ck-and-white photograph of a naked woman captured in a sensual, the woman appears to be of caucasian , with fair skin and light blonde hair, her mouth slightly open, black shirt, the background is a solid, dark black, which contrasts sharply with her skin tone and highlights, making her the focal point of the image, the lighting is soft and diffused, highlighting the contours of her body and creating a sense intimacy, the overall mood of the photograph is intense and erotic, emphasizing the natural beauty and sensuality of the subject,
The scene is captured in a wooden beam rafters in an abandoned mill, with sunlight streaming through the cracks location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Mamiya 7 II Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 773846091358334, Distilled CFG: 4.0

**Prompt 9 (Stylish - mermaid bustier reveal):**
```
The image is a high-resolution stylish photography of a 30y-o asian woman with Salt and Pepper (Gray) Gilded Waves hair. She is wearing mermaid-inspired satin bustier.
The woman is in the process of lifting her auburn Playsuit, revealing her bare breasts with prominent nipples.
The scene is captured in a on an oceanfront lounge, where the sound of waves brings calm location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon F2 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 241093013123665, Distilled CFG: 4.0

### Oral / Cum

**Prompt 10 (iPhone amateur - bathroom bustier):**
```
YOUR CONTEXT:  This is a photograph taken with an iPhone camera. Amateur photograph. Candid photograph. Casual photograph. Unfiltered photograph. Unposed photograph. Real life photograph. 2010s. Natural light. Subtle shadows. Taken for Facebook. Taken for Flickr. Taken for Instagram. Taken for OnlyFans. Taken for Reddit. Taken for Snapchat. YOUR PHOTO:
The photo is a high-resolution epic cinematic retro photo of   U.S. woman with ash blonde Feathered Bangs hair. She is wearing stocking and delicate lace bustier with thin straps.
She is looking up towards the viewer. Her mouth is gaping wide open. Her mouth if filled with a creamy, white semi-transparent substance that appears to be milk or a similar liquid.
The scene is captured in a softly flickering candles on the bathroom counter next to a filled bathtub with bubbles location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon D3500 DSLR Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 296278024844967, Distilled CFG: 4.0

**Prompt 11 (Retro - facial cum close-up):**
```
The image is a high-resolution retro photography of a 18y-o european woman with Mahogany Sleek Mid-Part  hair. She is wearing caftan and bathing suit.
A close-up photograph of a woman. Her eyes are closed, and her mouth is wide open, with her tongue extended and coated in a white, viscous semi-transparent wet drippy substance that appears to be semen. The texture of the substance is glossy and slightly translucent. Her facial expression is one of sexual arousal and satisfaction.
The scene is captured in a labyrinthine tunnels beneath a ruined city, with glowing runes along the walls location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Panasonic Lumix GH5 II Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 891064465770818, Distilled CFG: 4.0

**Prompt 12 (Fantasy - blowjob):**
```
The image is a high-resolution fantasy photography of  caucasian woman with Dark Chestnut Curly Ponytail hair. She is wearing Slacks and croptop.
Blowjob, fellatio, giving head. A young woman with long, dark brown hair and a pale complexion kneels on a light-colored floor in a well-lit room, wearing a black bra with red lace trim. Her face is tilted upwards as she sucks a man's erect penis. The man stands with his feet shoulder-width apart. His penis is visible, and his pubic hair. A silver necklace with a small pendant hangs around her neck. The man's face is not visible. The scene is lit with natural light and in a medium-angle shot. Kissing penis head while touching testicles.
The scene is captured in a empty tech desk in a forgotten lab on an asteroid, papers scattered location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS 5D Mark IV DSLR Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 1113379968042651, Distilled CFG: 4.0

**Prompt 13 (Stylish - deepthroat pool):**
```
The image is a high-resolution stylish photography of a 35y-o caucasian woman with dirty blonde Cascading Curls hair. She is wearing Qipao (Cheongsam) and anal milking full-body encasement.
(Male POV: Her expression is one of surprise and shock, (with her mouth extremely wide open:1.5) and eyes wide spread staring openly. Her jaws are spread and unhinged from this, her mouth is insanely opened. Another's gigantic thick and huge and extremely long penis is visible and is shoved into the open mouth of the woman. All the way inside the mouth. There is wet saliva and spit all around from this insertion and mouth fisting on the woman, penetrating her mouth hole with the arm. The penis is all the way inside. Her throat is bulging from the penis being on the inside. Her tongue is underneath and to the side. There is saliva and cum dripping down onto her chest and clevage. There is pools of cum all inside her mouth and on her tongue. Her mouth is filled with wet slimey cum.:0.6). (Her nipples are prominently visible, with the left nipple erect and the right nipple also erect. Both nipples are a light pink color. There are small, scattered freckles on her skin, adding a natural, youthful touch. :0.4)
The scene is captured in a in a swimming pool location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a6600 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 792862671307972, Distilled CFG: 4.0

**Prompt 14 (Amateur - cum in mouth):**
```
The image is a high-resolution amateur photography of a 18y-o asian woman with auburn short  hair. She is wearing Sporty skirt and sheer pantyhose. She is wearing glasses.
She is looking up towards the viewer. Her mouth is gaping wide open. Her mouth if filled with a creamy, white semi-transparent substance that appears to be milk or a similar liquid. Viewed from above, she is looking up. (Her entire mouth and throat is filled to the brim with white semen inside her mouth:1.6), it can be seen inside her throat from above. The liquid is all inside her mouth. (Her eyes are staring wide open:1.4), she is extremely happy and pleased with herself. Her tongue is completely covered in the white liquid semen, as it drips into her gaping mouth hole. Her eyes are closed
The scene is captured in a bedroom in a futuristic city apartment, with soft blue lighting and a floating bed location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon Z5 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 771821336992466, Distilled CFG: 4.0

**Prompt 15 (Realistic - creampie spread legs):**
```
The image is a high-resolution realistic photography of a 18y-o caucasian woman with dark blonde Supermodel Blowout hair. She is wearing Fencing skirt and Graphic tee with cargo pants.
The woman is lying on her back with her legs spread wide apart. She is positioned at the center of the image, looking directly at the camera with a slightly open mouth, showing a hint of a smile. The woman's pubic area is visible, showing a natural amount of pubic hair, and her labia are spread apart, revealing her vaginal opening. There is a visible semen stain on her inner thighs and around her vulva, indicating recent sexual activity.
The scene is captured in a in a sky-high glass-walled elevator, watching the city location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Lubitel 2. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 85085971150336, Distilled CFG: 4.0

### Anal

**Prompt 16 (Amateur - gaping anus rear view):**
```
The image is a high-resolution amateur photography of  asian woman with natural blonde Subtle Balayage Waves hair. She is wearing ukrainian vyshyvanka - a form-fitting embroidered blouse paired with a high-waisted skirt, showcasing the midriff and legs, featuring detailed floral patterns that emphasize femininity while celebrating Ukrainian traditions. and fitted and sheer black bodysuit layered under a classic blazer, styled with a plaid pencil skirt and sexy patent leather pumps..
Low angle, rear view. The focus is on her bare buttocks and vulva. Her anus is gaping widely and is open.
The scene is captured in a in a candlelit bistro corner, cozy and filled with the aroma of freshly baked bread location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a7 IV Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 943006752429896, Distilled CFG: 4.0

**Prompt 17 (Realistic - buttplug):**
```
The image is a high-resolution realistic photography of  african woman with Indigo cropped pixie hair. She is wearing Leaf print shorts and Cape.
The woman is on all fours with the butt towards the viewer and she is turning her head to look at the camera with a blushing face. (In the image is a Yellow buttplug, a plastic sex toy dildo.:1.4) It's inserted into her asshole. The woman looks seductively at the viewer.
The scene is captured in a in a candlelit castle courtyard, creating an aura of romance location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Kodak PIXPRO AZ241 16.15MP Digital Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 222833976647810, Distilled CFG: 4.0

**Prompt 18 (Professional - Pakistani jora anilingus):**
```
The image is a high-resolution professional photography of  caucasian woman with Auburn messy  hair. She is wearing Pakistani jora - a stunning outfit that blends traditional and modern elements, usually featuring a fitted tunic with a high slit and a flowing skirt, richly adorned with sequence and embroidery, showcasing the silhouette while celebrating vibrant cultural expression. and seamed pantyhose. She is wearing glasses.
The woman in the foreground is kneeling, with her face close to the other woman's anus. She is licking the other woman's, with her tongue extended and her eyes staring wide open. She is showing happiness and her mouth and tongue is licking the other woman.
The scene is captured in a dimly lit hallway with framed family photos and a soft carpet leading to a bedroom location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Kodak 35 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 17779575925631, Distilled CFG: 4.0

**Prompt 19 (Amateur - Malaysian finger insertion):**
```
The image is a high-resolution amateur photography of  Malaysian woman with ash blonde Gatsby Waves hair. She is wearing collared dress and serafuku. She is wearing glasses.
A finger is being slowly inserted into her anus and asshole. Viewed from behind. Eye contact. Sexy seductive eyes. Spread buttcheeks. She is bending over, leaning forward and to the side. Hand is on her butt, pointing to her anus. The finger is slowly inserted all the way into her asshole.
The scene is captured in a in a converted storage room, turned into a hidden little retreat location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a7R IV Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 82186977184614, Distilled CFG: 4.0

### Positions

**Prompt 20 (Serene minimalist - doggystyle POV):**
```
YOUR CONTEXT: You are a photographer who prefers serene and tranquil images. Your photographs exhibit attractive and spicy content, where everyone is sexy and provocative, with high contrast and clean, minimalist compositions that emphasize empty space to highlight negative space. YOUR PHOTO:
The photo is a high-resolution epic professional photo of  an adult 30y-o asian woman with Blonde Platinum Blonde Blowout hair. She is wearing Sweatpants and skin-tone tights.
A realistic image is desired from a male point of view, focusing on a woman who is completely nude, on all fours, with her long, brown hair tied up in a ponytail. She is barefoot and wearing jewelry, adding to her sensual appearance. The woman has a tattoo on her thigh, and is facing away from the viewer, with her ass and vaginal area visible. The scene suggests a sexual encounter, with the woman in a doggystyle position, and the man's penis, and male pubic hair implied to be present. The image is taken from a point of view that is from above, with the woman's body the solo focus, capturing her intimate moments in a realistic and detailed manner. The setting of the bed sheet adds to the intimate atmosphere of the depiction. The woman's nudity is uncensored, and the image is meant to be a realistic and unobstructed view of the scene. The man's presence is implied, but not the primary focus, with the woman's body and actions taking center stage.
The scene is captured in a airy, sunlit conservatory with lush plants, and a table set for tea location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a7 III Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 143088914432593, Distilled CFG: 4.0

**Prompt 21 (Professional - nude athlete on bed):**
```
The image is a high-resolution professional photography of a 18y-o european woman with red Romantic Braided Crown hair. She is wearing High-waisted shorts and Tap pants.
A nude female athlete with light brown wavy short hair sits on a bed in a bright and airy room, wearing a blue tie hanging loose around her neck. Her face looks at the camera with a grin. Her eyes are a bright blue and her skin is pale with freckles. She has a skinny body and petite build and her nipples are visible. Her breasts are small and perky, with pink nipples. Her pubic area is visible, and her vagina is exposed, with a small patch of blonde pubic hair over it. A blue satin sash is wrapped around her slim narrow waist. Her thin athletic legs are spread wide apart. The scene is lit with soft natural light pouring in through a window, and the image is rendered in a realistic and detailed style, with a focus on the woman's anatomy and expression.,
The scene is captured in a at a classic movie theater location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Pentax K1000 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 568782649120790, Distilled CFG: 4.0

**Prompt 22 (Professional - Barbudan nipslip):**
```
The image is a high-resolution professional photography of  Barbudan woman with dark blonde Runway-Ready Blowout hair. She is wearing white socks and Striped jumpsuit.
(Low angle, rear view. The focus is on her bare buttocks and vulva. Her anus is gaping widely and is open.:0.6). (She accidentally reveals a nipslip. She is caught by surprise and blushes severely by shame. Her face shows a reaction of shock and fear from being caught.:0.4)
The scene is captured in a in a candlelit secret garden, where flickering candles reveal hidden nooks location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon Z7 II Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 955195906221222, Distilled CFG: 4.0

**Prompt 23 (Professional - Syrian woman mixed poses):**
```
The image is a high-resolution professional photography of a 45y-o Syrian woman with ash blonde Soft Fluffy Curls hair. She is wearing Longline fishtail skirt and dungarees.
(Low angle, rear view. The focus is on her bare buttocks and vulva. Her anus is gaping widely and is open.:0.6). (She is sitting on the Train car roof topless with her breasts out. Her nipples are perky and hard and her large areolas are of a lighter skin tone.:0.4)
The scene is captured in a at an art nouveau cafe window, overlooking a bustling cobblestone street location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS Rebel T8i DSLR Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 222312456664763, Distilled CFG: 4.0

**Prompt 24 (Fantasy - missionary bed):**
```
The image is a high-resolution fantasy photography of  asian woman with red Side-Swept Curls hair. She is wearing princess costume and clothes pull. She is wearing glasses.
A realistic image is desired from a male point of view, focusing on a woman who is lying on her back on a bed, with her legs up and spread, showcasing her body. She has long, blonde hair and blue eyes that seem to be looking directly at the viewer. Her small breasts are visible, with a pink and blue bikini top underneath a black fishnet top, which is lifted up, exposing her breasts. The woman's mouth is open, and her lips are visible, forming an "o face" expression. She is bottomless, with her female pubic hair visible, and the image highlights her vaginal area and pussy. The scene suggests a sexual encounter, with the implication of a male presence, including a penis and erection, potentially in a missionary position. The woman's body is relaxed on the bed, with a pillow underneath her, and her veins are visible, adding to the realistic depiction of the scene. The solo focus remains on the woman, capturing her intimate moments in a realistic and detailed manner.
The scene is captured in a in a silk-wrapped canopy bed, a lavish suite with a bed cocooned in silky drapes location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon Z fc Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 521810137451945, Distilled CFG: 4.0

**Prompt 25 (Fantasy - Victorian hands and knees):**
```
The image is a high-resolution fantasy photography of a 18y-o caucasian woman with gray Milkmaid Updo hair. She is wearing crisp cotton Victorian top with cyber accents and thigh-high bondage boots. She is wearing glasses.
She is positioned on her hands and knees, facing away from the camera, which focuses on her bare buttocks and vulva.
The scene is captured in a on a French vineyard's estate, where grapevines stretch to the horizon location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Nikon Z fc Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 297184999563768, Distilled CFG: 4.0

**Prompt 26 (Realistic - space capsule hands and knees):**
```
The image is a high-resolution realistic photography of  european woman with Olive Knot Bun hair. She is wearing tunic and ripped tights. She is wearing glasses.
She is positioned on her hands and knees, facing away from the camera, which focuses on her bare buttocks and vulva.
The scene is captured in a space capsule hatch, slightly open, revealing a simple, minimal interior location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Graflex Century Graphic 2x3 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 774406075031940, Distilled CFG: 4.0

**Prompt 27 (Fantasy - genital close-up grassland):**
```
The image is a high-resolution fantasy photography of  african woman with ash brown Dreamy Half-Up Curls hair. She is wearing Sleek athleisure set with a sporty vibe  .
The focus is on her genital area, which is prominently displayed in the center of the image. Her skin is light-toned and smooth, with a few faint blemishes and natural body hair visible around the pubic area which has some pubic hairs.
The scene is captured in a rustic cabin in open grassland with a smoking chimney location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS R6 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 707327235114070, Distilled CFG: 4.0

**Prompt 28 (Retro - panty pull spread):**
```
The image is a high-resolution retro photography of a 20y-o african woman with auburn twist-out hair. She is wearing Fun patterned leggings with an oversized sweater  .
(Her hands are placed on her buttocks, spreading them apart to further expose her genitalia. Viewed from front. A very extreme close-up:0.6). (Her panties are being pulled down as she undresses while bending forward. She reveals her partially shaved vulva.:0.4)
The scene is captured in a at a seaside cafe under stars, where the waves form a soft melody nearby location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Olympus XA2 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 139882755064072, Distilled CFG: 4.0

### Lesbian

**Prompt 29 (Stylish - lesbian scene):**
```
The image is a high-resolution stylish photography of  caucasian woman with chestnut Pulled-Back Loose Waves hair. She is wearing strappy satin lingerie with side slits. She is wearing glasses.
(explicit image featuring two women engaging in a sexual act. They are positioned closely together with their faces nearly touching, and their tongues extended to lick the other woman's asshole. The woman has her hands on the other woman's vagina, spreading it apart to expose her genitalia. She is groping her and grabbing her body roughly but passionately. The women look completely different from each other and are wearing different clothes, different hairstyles and different faces. The other woman has Bronze Flipped Ends hair. She is wearing Silk blouse with high-waisted jeans  . She is wearing glasses.:0.6). (The woman is positioned on all fours, facing away from the camera, with her back arched, highlighting her curvaceous figure. Her skin is light and smooth, with a slight sheen, possibly from oil or sweat. Her hands are on the ground as she is all bent over. caught by surprise as she almost falls over, shocked face:0.4)
The scene is captured in a in a bustling market square, where the scent of spices and flowers fills the air location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Sony a7R IV Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 211372697202114, Distilled CFG: 4.0

**Prompt 30 (Professional - lesbian cunnilingus):**
```
The image is a high-resolution professional photography of  european woman with Cyan Chestnut Waves hair. She is wearing sari sarong and waist-high pleated skirt with a cropped sweatshirt featuring an oversized fit, paired with stylish slip-on trainers for a casual chic look..
This is a high-resolution photograph depicting two women engaged in an intimate act on a bed. Both women are completely nude, revealing their genitalia. The woman on the left is licking the other woman's labia. The woman on the right has her legs spread wide apart, exposing their genitals fully.
The scene is captured in a barren cliffside overlooking ocean with a distant lighthouse location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS 90D DSLR Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 856807537786019, Distilled CFG: 4.0

### Toys / Masturbation

**Prompt 31 (Orthochromatic - steampunk masturbation):**
```
YOUR CONTEXT: You are a photographer inspired by the look of early photographic processes, specifically orthochromatic and colorized films. Your photograph showcases attractive and spicy content, where everyone is sexy and provocative, with a scene with a high-contrast, blue-sensitive aesthetic. Reds appear very dark, while blues and greens are bright. The image is characterized by a stark, graphic quality. Post-processing focuses on enhancing the contrast and emphasizing the tonal separation. YOUR PHOTOGRAPH:
The photo is a high-resolution epic modern photo of   caucasian woman with medium brown shag hair. She is wearing Long-sleeve tee with a tiered skirt  .
A petite Black woman in steampunk inventor cosplay with goggles and corset gears masturbates intensely using a large brass vibrating prop on a workbench in a Victorian laboratory set, legs spread wide on the table as the device thrusts into her vagina while her other hand circles her clitoris, arousal fluid pooling on polished wood. Corset unlaced to expose small breasts. Warm gas lamp clusters on the bench provide intimate golden overhead light that highlights mechanical details and glistening mucosal surfaces, while cool blue Tesla coil effects in the background add electric side accents reflecting off brass and sweat. Style: Steampunk workshop erotic photography with gaslamp and coil contrast. Mood: Inventive frenzy, mechanical ecstasy, Victorian ingenuity lust.
The scene is captured in a from a glass control tower with radars location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Canon EOS RP Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 1061037357897233, Distilled CFG: 4.0

**Prompt 32 (Fashion high-key - gamer girl selfie):**
```
YOUR CONTEXT: You are a fashion photographer who creates ultra-bright photos for luxury productions. Your photograph showcases attractive and spicy content, where everyone is sexy and provocative, with a high-key composition shot flooded with soft studio lighting. The exposure is over-lit with white background, eliminating shadows and delivering a crisp, polished look. Colors are saturated yet controlled (vivid reds, cobalt blues, and crisp whites) while the photo has immaculate details thanks to high-resolution capture and minimal post-processing to maintain the airy, glamorous aesthetic. YOUR PHOTOGRAPH:
The photo is a high-resolution epic cinematic amateur photo of  an adult 45y-o caucasian woman with ash brown Dewy Waves hair. She is wearing white socks and Animal print dress.
A fit 22 year old woman's point of view in a snapchat mirror selfie.  She's masturbating and the text reads "Let's play!"  She's sitting in a gamer chair in front of colorful gamer PC with her legs spread, and her hand is covering her crotch. She's holding a pink phone and wearing headphones with pink cat ears.
The scene is captured in a in a walk-in closet turned reading nook, with dim lights and cozy throws location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Olympus XA2 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 915396256313682, Distilled CFG: 4.0

**Prompt 33 (Stylish - cucumber insertion):**
```
The image is a high-resolution stylish photography of a 18y-o caucasian woman with Purple Retro Bouffant hair. She is wearing fitted navy blazer worn over a lace-trimmed white blouse, paired with a cheeky mini skirt and thigh-high boots for a striking contrast. and germany: tall and athletic with fair skin, long blond hair styled in a neat braid, and blue eyes; beautifully adorned in a traditional dirndl, featuring a fitted bodice and a flowing skirt in vibrant colours, complemented with fashionable lace-up shoes.. She is wearing glasses.
She is nude. The image is taken from an extremely low angle, focusing on her genital area. Her legs are spread wide apart, and a large, green cucumber is inserted into her vagina pussy.
The scene is captured in a space capsule hatch, slightly open, revealing a simple, minimal interior location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Polaroid SX-70 Alpha Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 574767309257590, Distilled CFG: 4.0

### Artistic / Creative

**Prompt 34 (Dark cinematographer - spa BDSM):**
```
YOUR CONTEXT: You are a cinematographer who works in dark movies. Your photographs exhibit attractive and spicy content, where everyone is sexy and provocative, with intense side lighting and gobo-crafted patterns to sculpt deep, sharply defined shadows, along with a muted Ektachrome palette that evokes film noir. YOUR PHOTO:
The photo is a high-resolution  epic photo of   asian woman with black Butterfly Cut hair. She is wearing unbuttoned shorts and blouse.
(multiple women), 3 women, panties, pantyhose, skirt, kneeling, Bondage all women, dynamic pose, dynamic angle, Embarrassment, Blush, Closed eyes, Tears, Breath, bdsm, skin dentation, pain, (2women, jewelry, nipples, closed eyes, earrings, large breasts, grabbing another's breast, grabbing from behind, passion, colorful, vibrant, dramatic lighting, moaning face, enjoying pleasure, blushing, open mouth tongue out, croptop, wearing lace bodystocking:0.5)
The scene is captured in a on a spa day for couples location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Panasonic Lumix G7 Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 816646538748489, Distilled CFG: 4.0

**Prompt 35 (Teal orange color specialist - submarine threesome):**
```
YOUR CONTEXT: You are a color specialist photographer. Your photograph showcases attractive and spicy content, where everyone is sexy and provocative, with a satured teal and orange colors. The image features cool teal tones in the shadows and highlights, balanced by warm orange tones in the midtones and skin tones, creating a vibrant photography with high contrast and vivid colors. YOUR PHOTOGRAPH:
The photo is a high-resolution epic fantasy photo of   asian woman with Brown Long Blunt Cut hair. She is wearing stocking and sweater. She is wearing glasses.
(A man receives simultaneous attention from two women in the narrow cockpit of a vintage submarine during dive. The women—one East Asian with pixie cut, one Middle Eastern with long braid—kneel on either side, one performing deep fellatio while the other licks and sucks his testicles, his uniform pants lowered as he grips control panels. Uniform jackets are open exposing breasts pressed against his thighs, pre-ejaculate and saliva coating his erect penis. Dim red instrument lighting from dashboard panels bathes the scene in deep crimson tones, creating moody highlights on wet skin and fluids while green gauge glows add contrasting accents reflecting off brass and sweat. Style: Military vessel erotic photography with control room lighting. Mood: Claustrophobic intensity, submerged urgency, mechanical tension.:0.6). (A muscular Black man in knight armor cosplay with chainmail and sword props takes a petite East Asian woman dressed as a captured elf princess in flowing silver gown from behind on the stone floor of a medieval dungeon set. His thick erect penis penetrates her vaginally in deep strokes, arousal coating the shaft visibly with each withdrawal. Gown torn and bunched at waist, armor plates lifted. Cool blue dungeon LED torches provide low side lighting that outlines chain shadows and highlights sweat on skin, while a single warm overhead brazier effect casts orange glow on the thrusting hips and dripping fluids. Style: Medieval fantasy dungeon cosplay erotic photography with torch and brazier contrast. Mood: Chivalric conquest, elven captivity, iron-bound desire.:0.4)
The scene is captured in a on whispering desert dunes, where the wind carries secrets of ages past location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Contax G1/G2 Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: DPM++ 2M SDE, Seed: 538796884564087, Distilled CFG: 4.0

**Prompt 36 (Epic - wet body abandoned spacecraft):**
```
The image is a high-resolution epic photography of  african woman with blonde Curly Ponytail hair. She is wearing geta and milking machine tethered pantyhose. She is wearing glasses.
(Her skin and hair are (glistening with water droplets:1.4) and are (soaking wet from dripping water:1.3), (she is wet and is covered in dripping and running water:1.5). She is naked and the water is gushing across her naked body in droplets and streams.:0.6). (Focusing on her genital area, which is prominently displayed in the foreground. She has her legs spread wide apart, revealing her shaved vulva and anus. The skin around these areas is glistening, possibly from oil or moisture.:0.4)
The scene is captured in a glowing stones lining a narrow corridor in an abandoned spacecraft location with detailed background details in the environment. The award-winning photo is a close-up (film grain:1.2) photo shot with a Fujifilm X-H2S Mirrorless Camera. The photograph is highly detailed.
```
Settings: Steps: 32, Sampler: dpmpp_2m_sde_gpu_beta, Seed: 49251913475478, Distilled CFG: 4.0

## Strengths

- 1M+ tips indicating exceptional community confidence
- Unique YOUR CONTEXT / YOUR PHOTO prompt structure for precise style control
- Wide range of photographer personas (Hollywood, fashion, retro, aerial, amateur, dark cinema)
- Extensive camera model diversity influences rendering aesthetics
- Broad ethnicity and nationality support (European, Asian, African, Barbudan, Surinamese, Malaysian, Syrian)
- Weighted concept mixing allows blending multiple scenes (0.6/0.4, 0.5/0.5 splits)
- Both SFW and NSFW in one model
- FP8 format fits 12 GB VRAM
- GGUF versions available for lower VRAM systems (down to Q4_K_S at 6.6 GB)
- Each Civitai sample image includes embedded workflow

## Limitations

- No VAE or CLIP baked in - must load both separately
- 32 steps required - higher than some FLUX models
- Prompt structure is specific and verbose - requires following the YOUR CONTEXT/YOUR PHOTO format for best results
- Relatively new model (12K downloads vs 48K-82K for established competitors)
- Two samplers to choose from (dpmpp_2m_sde vs dpmpp_2m_sde_gpu_beta) - results may vary

## Notes

- UNet model - requires separate VAE (flux_vae.safetensors) and CLIP (Dual Clip Loader)
- ComfyUI recommended workflow included in every sample image on Civitai
- The YOUR CONTEXT section is optional but strongly recommended for style control
- `(film grain:1.2)` in the scene section is used in every sample prompt and adds realism
- Weight syntax like `(content:0.6)` works for concept mixing - higher weight = stronger influence
- Camera model choice affects rendering: vintage cameras (Kodak, Mamiya, Graflex) produce different aesthetics than modern mirrorless (Sony, Canon)
- Supports age specification (e.g., "adult 40y-o", "18y-o", "35y-o")
- Both sampler variants (DPM++ 2M SDE and dpmpp_2m_sde_gpu_beta) work well at 32 steps with Distilled CFG 4.0
- Tags: porn, nsfw, sfw, flux, base model

---

**Category:** NSFW Specialized
**Last Updated:** 2026-02-11

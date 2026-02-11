# PussyDiffusion FLUX - Prompt Examples Reference

**LoRA:** `pussydiffusion-f1.safetensors`
**Civitai:** https://civitai.com/models/983498/pussydiffusion-flux

---

## Technical Settings (CRITICAL)

### General Guidelines
- **Strength:** 0.7-1.5 (depending on scenario)
  - txt2img: 0.2-0.6 (lower strength)
  - inpaint: 0.7-1.5 (higher strength for better results)
  - Typically: 0.5-0.75 for balanced results
- **Steps:** 20-50
- **CFG:** 1-5
- **Scheduler:** Beta (0.6/0.6) or Simple
- **Best for:** INPAINT (but works in txt2img too)

### Important Notes
- ⚠️ **May break anatomy sometimes** - test different strengths
- ⚠️ **Negative weight adds clothes** (LOL - interesting effect)
- ⚠️ **Known issue:** Sometimes generates almost translucent horizontal pattern in image
- ✓ Flux sometimes doesn't follow trigger words - may need multiple attempts
- ✓ Rear view works - add "showing her ass from the back"

---

## Trigger Words

| Trigger | Description | Details |
|---------|-------------|---------|
| `bush pussy` | Shaved with hair on top / landing strip | Clean shaved with trimmed pubic hair on top |
| `butterfly pussy` | Large labia (big lips) | Prominent outer/inner lips |
| `hairy pussy` | Hairy pussy/vagina | Natural, unshaved |
| `shaved pussy` | Shaved, cameltoe style (innie) | Completely smooth, tucked appearance |
| `asian pussy` | Dark lips on fair skin | Contrast between skin tone and labia color |
| `pink pussy` | Pink delicate pussy | Light pink, delicate appearance |

**Combining Triggers:**
- You can use multiple triggers together: `pink pussy, shaved pussy`
- Mix anatomy types: `butterfly pussy, hairy pussy`

**Rear Views:**
- Add "showing her ass from the back" or similar
- Trained on rear views for all trigger types above

---

## Example 1: French Brunette Bed (Shaved Pussy)

```
A Raw Hires photograph of a fair-skinned stark naked white 20 year old French Brunette woman with shoulder-length, wavy brown hair, She has a warm, friendly smile and is looking at viewer with her beautiful Green eyes. She is lying on a white bed. She has small to medium breasts, a slim physique with breathtaking hourglass curves, and is posing with her legs spread apart, revealing her (shaved pussy) and vulva. <lora:pussydiffusion-f1:0.5> <lora:nipplediffusion-f1:0.5>
```

**Settings:**
- Steps: 50
- CFG: 5
- Size: 832x1216
- LoRA Strength: 0.5

**Trigger Used:** `shaved pussy`

**Why It Works:**
- Natural language description with "Raw Hires photograph"
- Specific ethnicity and features (French, green eyes, brown hair)
- Body type details (hourglass curves, small to medium breasts)
- Clear pose description (lying, legs spread)
- Combined with NippleDiffusion for full anatomy

---

## Example 2: Cute Blonde Pool (Shaved Pussy)

```
A photo focus on a very cute, innocent, adorable, naked blonde 18 year old girl who is petite, with small breasts. She is relaxed, and her legs are spread, showing her shaved pussy, labia, and clitoris. She is on a pool chair in a pink girly pool room. She is excited. She is wearing a cute pink belt, a cute pink choker <lora:pussydiffusion-f1:1>
```

**Settings:**
- Steps: 50
- CFG: 3.5
- Size: 832x1216
- LoRA Strength: 1.0

**Trigger Used:** `shaved pussy`

**Why It Works:**
- Multiple personality descriptors (cute, innocent, adorable)
- Age and body type specified (18, petite, small breasts)
- Anatomical detail keywords (labia, clitoris)
- Environment details (pool chair, pink pool room)
- Emotion/mood (relaxed, excited)
- Accessories (belt, choker) add realism

---

## Example 3: Bengali Watercolor (Asian/Hairy Pussy)

```
(((nsfw))) naked bengali woman, (water colour) (best quality) (intricate details) (8k) (HDR) (cinematic lighting) (sharp focus) ((detailed and realistic face:1)), ((painted in the style of edwin lord weeks), athletic, sitting on doorstep in kolkata, traditional jewelry, legs apart, asian pussy, hairy pussy <lora:pussydiffusion-f1:1>
```

**Settings:**
- Steps: 35
- CFG: 5.5
- Size: 832x1216
- LoRA Strength: 1.0

**Trigger Used:** `asian pussy, hairy pussy` (combined)

**Why It Works:**
- Artistic style reference (watercolour, edwin lord weeks)
- Quality boosters (8k, HDR, intricate details)
- Cultural specificity (Bengali, Kolkata, traditional jewelry)
- Multiple triggers combined
- NSFW tag to unlock
- Emphasis markers (((nsfw))), ((face:1))

---

## Example 4: Latina Fantasy Throne (Bush Pussy)

```
raw photo, stunningly beautiful 20 yr old latina naked woman supermodel, full body view, viewed from below, Fantasy theme, magical, mythical scene, medieval setting, colorful, realistic, detailed, smooth clear caramel skin, long flowing straight blonde hair, large round emerald-green eyes, sexy long eyelashes, all around winged eyeliner, super glossy brown lips, closed mouth, ruby-diamond choker, ruby-diamond dangling earrings, gold bracelets, gold ankle bracelets, nicely shaped sexy caramel legs, bare feet, pink pussy is visible, well manicured pubic hair, brown pubic hair, pink pussy, relaxed sensuously sitting on an overstuffed couch in a medieval ornate throne room, legs spread towards viewer exposing her vagina, lit by firelight <lora:pussydiffusion-f1:0.9>
```

**Settings:**
- Steps: 30
- CFG: 1
- Sampler: Euler
- Size: 1024x1280
- Schedule: Beta (0.6/0.6)
- Distilled CFG: 3.5
- LoRA Strength: 0.9

**Trigger Used:** `bush pussy, pink pussy` (implied through "well manicured pubic hair")

**Why It Works:**
- Camera angle specified (viewed from below)
- Fantasy theme with medieval setting
- Extensive detail on facial features (eyes, lashes, eyeliner, lips)
- Jewelry/accessories detailed (choker, earrings, bracelets)
- Skin tone description (smooth clear caramel)
- Lighting source (firelight)
- Multiple pubic hair descriptors
- Beta scheduler for better quality

---

## Example 5: Irish Redhead Penthouse (Pink Pussy)

```
A Raw Ultra High-resolution photoshoot of an Irish Redhead model named Rua Aingeal (Red Angel), a fair-skinned 22 year old petite woman with striking green eyes and long, wavy red hair. Her face is covered in freckles, particularly prominent on her cheeks and nose. The lighting is soft, highlighting her skin texture and the natural beauty of her freckles, She is Stark Naked, Fully Nude for this Photoshoot, posing with her arms raised and sitting with only one knee on ground while sitting with her other sole on the ground. Her beautiful pink pussy is exposed. She has dark nipples with more contrast to the natural skin and long beautiful puffy nipples. She has the most breathtakingly beautiful smile on her face. The photo was taken in a cozy minimalist urban penthouse with an wooden floor. <lora:pussydiffusion-f1:0.6> <lora:nipplediffusion-f1:0.6>
```

**Settings:**
- Steps: 50
- CFG: 5
- Size: 832x1216
- LoRA Strength: 0.6

**Trigger Used:** `pink pussy`

**Why It Works:**
- Character name adds consistency (Rua Aingeal)
- Detailed facial features (freckles, green eyes)
- Lighting description (soft, highlighting texture)
- Specific pose (arms raised, one knee down)
- Nipple details combined with PussyDiffusion
- Environment details (minimalist penthouse, wooden floor)
- Emotion (breathtakingly beautiful smile)

---

## Example 6: Ukrainian Goddess Lake

```
raw photo, dramatic light, dramatic pose, dramatic angle, stunningly beautiful naked 18 yr old ukranian goddess, sitting on a large rock beside a lake, realistic smooth caramel skin, slim body, perky small caramel breasts with pink nipples, lake, sexy long eyelashes, all around winged eyeliner, realistic long golden brown braided hair, dripping wet hair, dripping wet skin, dripping water, bathing in lake, washing hair, washing body, head bent to one side rinsing out wet hair, full body view, golden brown pubic hair, sensuous large round golden-hazel eyes, facing viewer, nicely shaped caramel legs, bare feet, legs spread toward viewer <lora:pussydiffusion-f1:0.5> <lora:nipples:0.5> <lora:NSFW_master:0.45>
```

**Settings:**
- Steps: 30
- CFG: 1
- Sampler: Euler
- Size: 1024x1280
- Schedule: Beta (0.6/0.6)
- Distilled CFG: 3.5
- LoRA Strength: 0.5

**Trigger Used:** Implicit through `golden brown pubic hair`

**Why It Works:**
- Triple "dramatic" emphasis (light, pose, angle)
- Water effects (dripping wet hair, skin, water)
- Action (bathing, washing, rinsing)
- Dynamic pose (head bent to one side)
- Color coordination (golden brown hair + pubic hair)
- Three LoRAs balanced (pussy + nipples + NSFW master)
- Natural outdoor setting

---

## Example 7: Hotel Brunette (Pink Pussy)

```
Realistic photo of a sexy young, fit, slender 28-year-old Caucasian brunette sitting on a hotel bed, gazing at the camera. She has a youthful, heart-shaped faces with lips that aren't overly plump, cheekbones that aren't too prominent, and and chin with no cleft. Her hair is in a messy ponytail. She wear pink lipstick, with her makeup subtle, cute and girlish. She has pale, creamy alabaster skin. She is naked with medium saggy, a flat stomach, and a pink pussy, legs spread. She wears a coy, seductive smile on her pretty faces, lips slightly parted, her green eyes gazing alluringly, with a sense of sensual invitation. she has a black digital watch on her wrist and a pierced navel. In the background, out of focus, the lights of the city at night can be seen glistening with bokeh. <lora:pussydiffusion-f1:0.75>
```

**Settings:**
- Steps: 25
- CFG: 5
- Size: 832x1216
- LoRA Strength: 0.75

**Trigger Used:** `pink pussy`

**Why It Works:**
- Negative descriptors (lips NOT overly plump, cheekbones NOT too prominent)
- Natural imperfections (medium saggy breasts, messy ponytail)
- Small details (black digital watch, pierced navel)
- Background with depth (city lights, bokeh)
- Emotional expression (coy, seductive smile)
- Eye contact (gazing at camera, gazing alluringly)

---

## Example 8: Inpaint - Panties Pulled Down (Special)

```
The image is a high-resolution photograph featuring a close-up, low-angle perspective of a woman's lower body. She is positioned on her hands and knees on a wooden floor, with her legs spread apart. Her skin tone is light and smooth. She is wearing a pair of pale pink, lace-trimmed panties with a small black bow at the center of the waistband. The panties are being pulled down, revealing her vulva. The lighting is bright and natural, suggesting daylight coming from an unseen source, possibly a window in the background. <lora:pussydiffusion-f1:0.75>
```

**Settings:**
- Steps: 20
- CFG: 1
- Sampler: Euler
- Schedule: Simple
- Distilled CFG: 3.5
- **Denoising: 0.75**
- **Inpaint area: Only masked**
- LoRA Strength: 0.75

**Trigger Used:** None (inpaint focused)

**Why It Works (INPAINT SPECIFIC):**
- Technical camera description (close-up, low-angle)
- Specific pose (hands and knees, legs spread)
- Clothing detail (pale pink, lace-trimmed, black bow)
- Action (panties being pulled down)
- Lighting source (bright, natural, daylight, window)
- Surface detail (wooden floor)
- **Denoising 0.75** for inpaint strength
- Lower steps (20) sufficient for inpaint

---

## Prompt Patterns That Work Well

### 1. Photography Studio Style
**Pattern:**
```
[Raw/Ultra High-resolution] [photoshoot/photograph] of [ethnicity] [age] [body type] woman
[facial features] [hair description] [pose] [anatomy details] [trigger words]
[environment] [lighting] <lora:pussydiffusion-f1:[strength]>
```

**Examples:**
- "Raw Hires photograph of a fair-skinned French Brunette woman..."
- "Ultra High-resolution photoshoot of an Irish Redhead model..."

**Keywords:** Raw, Hires, photoshoot, photograph, model

### 2. Natural/Outdoor Scene
**Pattern:**
```
raw photo, [dramatic descriptors] [age] [ethnicity] [body type]
[location] [weather/water effects] [hair] [pubic hair color]
[pose] [legs spread] [lighting] <lora:pussydiffusion-f1:[strength]>
```

**Examples:**
- "stunningly beautiful naked Ukrainian goddess, sitting on rock beside lake, dripping wet..."
- "naked bengali woman, sitting on doorstep in kolkata..."

**Keywords:** dramatic, goddess, dripping wet, outdoor, natural

### 3. Fantasy/Medieval Theme
**Pattern:**
```
[Fantasy theme] [age] [ethnicity] naked woman [body type]
[medieval setting] [detailed clothing/jewelry] [skin description]
[facial features] [pubic hair style] [pose] [firelight/magical lighting]
<lora:pussydiffusion-f1:[strength]>
```

**Examples:**
- "Fantasy theme, magical, mythical scene, medieval setting, latina naked woman, ruby-diamond choker..."

**Keywords:** fantasy, magical, mythical, medieval, throne, firelight

### 4. Intimate Indoor Setting
**Pattern:**
```
Realistic photo, [age] [ethnicity] [body type] sitting on [furniture]
[facial features] [makeup] [hair] [pink pussy trigger]
[emotional expression] [small details: watch, jewelry, piercings]
[background with bokeh] <lora:pussydiffusion-f1:[strength]>
```

**Examples:**
- "28-year-old Caucasian brunette sitting on hotel bed, pink lipstick, subtle makeup..."

**Keywords:** realistic photo, hotel, bed, bokeh, city lights, gazing

### 5. Artistic Style
**Pattern:**
```
[art style: watercolour, painting] (quality boosters) [artist name]
[ethnicity] [pose] [cultural details] [trigger words]
<lora:pussydiffusion-f1:[strength]>
```

**Examples:**
- "(water colour) painted in the style of edwin lord weeks, bengali woman..."

**Keywords:** watercolour, painted, style of [artist], artistic

### 6. Inpaint-Specific Pattern
**Pattern:**
```
The image is a [high-resolution] photograph featuring [camera angle]
[body position] on [surface]. [Clothing description] being [action: pulled down, removed]
revealing [anatomy]. Lighting is [bright/natural] from [source].
<lora:pussydiffusion-f1:0.7-1.5>
```

**Settings:** Denoising 0.5-0.75, Inpaint area: Only masked

---

## Common Keywords in Successful Prompts

### Anatomy Keywords:
- `shaved pussy`, `pink pussy`, `bush pussy`, `hairy pussy`, `asian pussy`, `butterfly pussy`
- `vulva`, `labia`, `clitoris`, `vagina`
- `legs spread`, `legs apart`, `legs spread toward viewer`
- `pubic hair`, `trimmed pubic hair`, `well manicured pubic hair`, `[color] pubic hair`

### Body Parts:
- `small breasts`, `medium breasts`, `perky breasts`, `saggy breasts`
- `nipples`, `dark nipples`, `pink nipples`, `puffy nipples`
- `flat stomach`, `hourglass curves`, `slim body`, `petite`, `athletic`
- `caramel skin`, `fair skin`, `pale skin`, `alabaster skin`

### Pose & Position:
- `sitting`, `lying`, `hands and knees`, `legs spread`, `legs apart`
- `arms raised`, `posing`, `relaxed`, `sensuously sitting`
- `facing viewer`, `gazing at camera`, `looking at viewer`
- `viewed from below`, `low-angle`, `close-up`, `full body view`

### Facial Features:
- `green eyes`, `hazel eyes`, `emerald-green eyes`
- `long eyelashes`, `winged eyeliner`, `pink lipstick`, `glossy lips`
- `freckles`, `heart-shaped face`, `youthful face`
- `smile`, `seductive smile`, `breathtaking smile`, `coy smile`

### Hair:
- `wavy brown hair`, `long red hair`, `blonde hair`, `braided hair`
- `shoulder-length`, `long flowing`, `messy ponytail`
- `dripping wet hair` (for water scenes)

### Environment:
- `white bed`, `hotel bed`, `pool chair`, `overstuffed couch`, `rock beside lake`
- `medieval throne room`, `urban penthouse`, `doorstep in kolkata`
- `wooden floor`, `pink pool room`, `minimalist`

### Lighting:
- `soft lighting`, `dramatic light`, `firelight`, `daylight`, `bright and natural`
- `highlighting skin texture`, `lit by firelight`, `golden hour`
- `bokeh` (background blur)

### Clothing/Accessories:
- `stark naked`, `fully nude`, `topless`, `bottomless`
- `panties being pulled down`, `pink belt`, `choker`
- `jewelry`, `earrings`, `bracelets`, `watch`, `pierced navel`

### Quality Boosters:
- `raw photo`, `realistic`, `detailed`, `high-resolution`, `ultra high-resolution`
- `(best quality)`, `(8k)`, `(HDR)`, `(intricate details)`, `(sharp focus)`
- `stunningly beautiful`, `breathtaking`

### Emotions/Mood:
- `relaxed`, `excited`, `sensual`, `seductive`, `innocent`, `adorable`, `cute`
- `warm smile`, `friendly smile`, `coy smile`
- `gazing alluringly`, `sense of sensual invitation`

---

## Tested LoRA Combinations

### Combination 1: Full Anatomy (Best)
```
<lora:pussydiffusion-f1:0.5> <lora:nipplediffusion-f1:0.5>
```
**Use case:** Complete female anatomy enhancement (pussy + nipples)
**Strength:** Balanced 0.5/0.5 works best

### Combination 2: NSFW Unlock
```
<lora:pussydiffusion-f1:0.6> <lora:MysticXXX-v6:0.7>
```
**Use case:** Strong NSFW content generation
**Strength:** Higher MysticXXX for NSFW unlock

### Combination 3: Triple Anatomy Stack
```
<lora:pussydiffusion-f1:0.5> <lora:nipples:0.5> <lora:NSFW_master:0.45>
```
**Use case:** Maximum anatomical detail with NSFW enhancement
**Strength:** Balanced across three LoRAs

### Combination 4: Innocent/Cute Style
```
<lora:pussydiffusion-f1:0.2> <lora:midjourney_whisper_innocent_eyes_v01:0.5> <lora:Jib_Flux_Nipple_Fix_v2:0.4>
```
**Use case:** Innocent, cute faces with subtle anatomy
**Strength:** Lower pussy (0.2) for subtlety, higher eyes for character

### Combination 5: Multi-LoRA Low Strength
```
<lora:pussydiffusion-f1:0.2> <lora:MysticXXX:1.0> <lora:xxx:0.2>
```
**Use case:** Multiple LoRAs where you want MysticXXX to dominate
**Strength:** Low pussy strength to avoid conflicts

---

## Strength Guidelines by Use Case

| Use Case | Strength | Notes |
|----------|----------|-------|
| **txt2img (subtle)** | 0.2-0.4 | Light anatomical influence |
| **txt2img (standard)** | 0.5-0.75 | Balanced, typical use |
| **txt2img (strong)** | 0.8-1.0 | Strong anatomical control |
| **Inpaint (moderate)** | 0.7-0.9 | Good for refinement |
| **Inpaint (strong)** | 1.0-1.5 | Maximum control |
| **Multi-LoRA** | 0.2-0.5 | Lower to avoid conflicts |
| **Solo LoRA** | 0.6-1.0 | Higher when used alone |

---

## Sampler & Scheduler Recommendations

### Best Samplers
1. **Euler** - Most consistent, works with Beta scheduler
2. **DPM++ 2M** - Good quality, slightly slower
3. **DEIS** - Fast, good for iteration

### Best Schedulers
1. **Beta (0.6/0.6)** - Highest quality, recommended for finals
2. **Simple** - Fast, good for testing
3. **Normal** - Balanced default

### CFG Settings
- **CFG 1-3:** Best for high-detail, photorealistic (with Distilled CFG 3.5)
- **CFG 3.5-5:** Balanced, follows prompts well
- **CFG 5.5+:** Stronger prompt adherence (may oversaturate)

---

## Common Issues & Solutions

### Issue 1: Translucent Horizontal Pattern
**Symptoms:** Almost transparent horizontal lines/pattern in image
**Solution:**
- Lower LoRA strength (try 0.3-0.5)
- Change sampler (try Euler if using DPM++)
- Add quality negative prompts
- Try different seed

### Issue 2: Broken Anatomy
**Symptoms:** Malformed or unrealistic anatomy
**Solution:**
- Use inpaint instead of txt2img
- Lower strength to 0.4-0.6
- Add more anatomical keywords (vulva, labia)
- Combine with anatomical LoRAs (NippleDiffusion)
- Use Beta scheduler

### Issue 3: Trigger Words Not Working
**Symptoms:** Flux ignores trigger words (shaved pussy, pink pussy, etc.)
**Solution:**
- Increase LoRA strength
- Add emphasis: `(shaved pussy:1.2)` or `shaved pussy, smooth`
- Use multiple related keywords
- Try different trigger combinations
- Regenerate with different seed

### Issue 4: Clothes Appearing
**Symptoms:** Unwanted clothing in nude generation
**Solution:**
- Add "naked", "nude", "fully nude", "stark naked" to prompt
- Negative prompt: "clothes, clothed, dressed, underwear"
- **Experiment:** Try NEGATIVE LoRA weight (adds clothes - interesting!)
- Increase NSFW LoRA strength (MysticXXX)

### Issue 5: Wrong Pubic Hair Style
**Symptoms:** Gets hairy when you want shaved, or vice versa
**Solution:**
- Use correct trigger word explicitly
- Add descriptive keywords: "completely smooth", "well-groomed"
- For shaved: "smooth pussy, no pubic hair, waxed"
- For hairy: "natural pubic hair, untrimmed, full bush"
- Increase strength to 0.7-1.0

---

## Advanced Techniques

### 1. Rear View Generation
**Add to prompt:**
- "showing her ass from the back"
- "view from behind"
- "rear view, bent over"
- "doggy style position, from behind"

**Works with all triggers:** bush pussy, shaved pussy, pink pussy, etc.

### 2. Inpainting Workflow
**Best practice:**
1. Generate base image without PussyDiffusion (or low strength 0.2)
2. Mask the genital area
3. Inpaint with PussyDiffusion at 0.7-1.5 strength
4. Use "Only masked" inpaint area
5. Denoising: 0.5-0.75
6. Steps: 20-30 (fewer needed for inpaint)

### 3. Combining Multiple Triggers
**Examples:**
- `pink pussy, shaved pussy` - pink and smooth
- `butterfly pussy, hairy pussy` - large lips with hair
- `bush pussy, pink pussy` - landing strip with pink coloring

### 4. Negative Weight for Clothes (Experimental)
**How it works:**
- Use negative LoRA strength: `<lora:pussydiffusion-f1:-0.5>`
- Adds clothing to the image
- Interesting effect for creative exploration
- Unstable, unpredictable results

### 5. Multi-Pass Generation
**Workflow:**
1. **Pass 1:** Generate with low strength (0.3) for composition
2. **Pass 2:** Inpaint genital area with high strength (1.0)
3. **Pass 3:** Inpaint face/details with face LoRAs
4. **Pass 4:** Upscale with anatomical LoRAs enabled

---

## Ethnicity-Specific Tips

### Asian
- Trigger: `asian pussy` (dark lips on fair skin)
- Add: "asian woman", "japanese", "chinese", "korean", "bengali"
- Skin: "fair skin", "pale skin", "porcelain skin"

### Latina
- Add: "latina", "hispanic", "caramel skin", "tan skin"
- Pubic hair: "brown pubic hair", "dark pubic hair"
- Combine with: `pink pussy` or `bush pussy`

### Caucasian/European
- Trigger: `pink pussy` (most common)
- Add: "fair-skinned", "pale", "alabaster skin", "Irish", "French"
- Pubic hair: "blonde pubic hair", "red pubic hair", "ginger"

### Middle Eastern/Mediterranean
- Add: "olive skin", "mediterranean", "tan"
- Combine with: `butterfly pussy` (fuller lips)
- Pubic hair: "dark pubic hair", "thick pubic hair"

---

## Photography Style Tips

### Raw/Realistic Photography
**Keywords:** raw photo, realistic photo, photograph, DSLR
**Settings:** CFG 3.5-5, Euler sampler, Beta scheduler

### High-Fashion Photoshoot
**Keywords:** photoshoot, model, studio, professional lighting, soft lighting
**Settings:** Steps 40-50, CFG 5, high resolution

### Artistic/Painted Style
**Keywords:** watercolour, painted in the style of [artist], artistic
**Settings:** CFG 5.5+, emphasis on art style in prompt

### Amateur/Candid
**Keywords:** amateur photo, iphone photo, candid, natural
**Settings:** Lower CFG (1-3), add grain/imperfections

### Vintage/Film
**Keywords:** film photograph, vintage, 35mm, film grain
**Settings:** Add film-specific keywords, slight desaturation

---

## Resolution Recommendations

| Aspect Ratio | Resolution | Best For |
|--------------|------------|----------|
| **Portrait (tall)** | 832x1216 | Standing, sitting portraits |
| **Portrait (taller)** | 1024x1280 | Full body, legs visible |
| **Square** | 1024x1024 | Centered composition |
| **Landscape** | 1216x832 | Lying down, wide scenes |
| **Close-up** | 768x1024 | Genital focus, inpaint |

---

## Recommended Workflow

### For Beginners
1. Start with **txt2img, strength 0.5**
2. Use one clear trigger word: `shaved pussy` or `pink pussy`
3. Simple prompt with ethnicity + age + pose + trigger
4. Euler sampler, Simple scheduler, CFG 3.5
5. 30 steps, 832x1216 resolution

### For Advanced Users
1. **Multi-LoRA stack:** PussyDiffusion + NippleDiffusion + MysticXXX
2. **Lower strengths** (0.3-0.5) when combining multiple LoRAs
3. **Beta scheduler (0.6/0.6)** for final renders
4. **Inpaint** for refinement (denoising 0.75, strength 0.8-1.2)
5. **50 steps**, higher resolution (1024x1280+)

### For Inpainting
1. Generate base with **low strength (0.2)** or no pussy LoRA
2. Mask genital area carefully
3. Inpaint with **strength 0.7-1.5**
4. **Denoising 0.5-0.75** depending on how much change needed
5. **20-30 steps** (fewer for inpaint)
6. Scheduler: Simple or Beta
7. Inpaint area: **Only masked**

---

*Last updated: 2026-02-10*

[← Back to LoRA Documentation](pussydiffusion.md)

# Nightclub Girl

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1379 |
| **👍** | 122 |
| **Tips** | 10 |

| Parameter | Value |
|-----------|-------|
| **File** | `Nightclub_Girl.safetensors` |
| **Original filename** | `ClubVibeGirls_v3_Flux_512_fix-000010.safetensors` |
| **Civitai** | https://civitai.com/models/1810229/nightclub-girl-or |
| **Trigger word** | `nc_girls`, `nc_scene`, `nc_heels` |
| **Strength** | < 0.8 recommended |
| **Type** | CONCEPT / Scene |
| **Compatibility** | FLUX |

### Description

Complex LoRA for nightclub scenes featuring interactions between background crowds, spatial elements, and foreground girls. Includes multiple scene types (DJ booth, bar, dance floor, elevated platforms), various poses (dancing, squatting, kneeling, pole dancing), and clothing styles. **Strength < 0.8 recommended** to avoid body merging issues.

### Key features

- Multiple scene types (DJ booth, bar, sofa area, elevated platform)
- Various action poses (dancing, squatting, kneeling, pole dance)
- Nightclub clothing styles (bikinis, bodysuits, dresses)
- Clear platform heels option
- Multi-girl control support
- Background crowd interactions

### Trigger words

| Trigger | Purpose |
|---------|---------|
| `nc_girls` | Activates girl generation |
| `nc_scene` | Activates nightclub scene |
| `nc_heels` | Activates clear platform heels |

### Scene prompts

**DJ Booth + Dance Floor:**
```
dance floor,dj booth,nc_scene
```

**Bar Area:**
```
bar area,bar counter,nc_scene
```

**Dance Floor Stage:**
```
dance floor,stage,nc_scene
```

**Sofa Area:**
```
lounge area, sofa area,white sofa,black side table,nc_scene,moody atmosphere
```

**Elevated Platform:**
```
nightclub,nigtclub's stage,stage, elevated square platform, (very high square platform:1.3),people in background gathered ((below the elevated platform):1.4), surrounding it and ((looking at girl):1.4), at the base of the stage.nc_scene
```

**Diagonal Performing Stage:**
```
Dynamic shot,looking at performing on a ((diagonal stage edge):1.4) that extends across the frame. The audience is visible on the left side of the frame, below the stage, while the girl is prominently featured on the right side, above the stage.nc_scene
```

### Action prompts

**Dance:**
```
dancing,(seductive pose:1.2), (sensual dance:1.3), (provocative:1.1), dynamic pose, energetic, (arms outstretched:1.2), hands reaching out
```

**Half-Squat:**
```
standing,half-squat, (bending knees, knees bent, spread legs,low stance, dynamic pose,leaning forward):0.6, (legs bent:0.7),hands on knee
```

**Deep-Squat:**
```
squatting, full-squat, deep squat, low squat, ass to ground, sitting on heels, (buttocks below knees:1.2), (very low:1.1), dynamic pose
```

**Single Knee:**
```
kneeling, single knee kneeling, hand on knee, one knee down, (one knee on ground:1.2), (leg bent:1.1), dynamic pose
```

**Full Knee:**
```
kneeling,( full kneeling, both knees down):1.3, (knees on ground:1.2), (sitting on calves:1.1), hand on thigh, static pose
```

**Pole Dance:**
```
(pole dancing:1.3), (on pole:1.2), (dance pole:1.1), flexible,graceful,holding pole rod
```

**Posing with liquor:**
```
posing, group photo, looking at viewer, sultry expression, holding liquor bottle
```

### Clothing prompts

**Red Cutout Dress:**
```
(red cutout dress:1.3), (red cutout mini dress:1.2), (one-shoulder dress:1.1), (choker:1.0)
```

**Gem Bikini:**
```
(gemstones with chains bikini:1.4), (blue gems:1.2), (minimal clothing:1.2), revealing outfit
```

**Gold Bodysuit:**
```
(gold bodysuit:1.4), (shiny clothing:1.3), (latex clothing:1.2), (lace stockings:1.3), (white stockings:1.2), (garter straps:1.1)
```

**Neon Green:**
```
(neon green clothing:1.4), (fluorescent green:1.3), (long sleeve crop top:1.2), (bikini bottom:1.1),underboob
```

**Fishnet Bodysuit:**
```
(fishnet bodysuit:1.4), (mesh clothing:1.3), (cutout bodysuit:1.2), (black clothing:1.1)
```

**Sparkling Dress:**
```
(sparkling dress:1.4), (rhinestone dress:1.3), (cutout dress:1.6), (one-shoulder dress:1.2), (black dress:1.1)
```

**White Mini Dress:**
```
(white dress:1.4), (mini dress:1.3), (bodycon dress:1.2), (shiny clothing:1.1), (silk dress:1.1), (satin dress:1.1), spaghetti strap, frilled trim, choker, white gloves
```

**Rhinestone Bikini:**
```
(bejeweled bikini:1.4), (rhinestone bikini:1.3), (sparkling clothing:1.2)
```

**Black Bodysuit:**
```
(black bodysuit:1.3), (shiny clothing:1.2), (latex clothing:1.1)
```

### Footwear prompts

**Clear Platform Heels:**
```
wearing nc_heels,clear platform high heels with clear straps, the heels' straps are visible
```

**Boots:**
```
wearing ankle boots
wearing knee-high boots
```

**Hidden shoes:**
```
shoes not visible
```

### Multi-girl control

```
3girls total in foreground, (1girl from front:1.4),2girls from behind
3girls total in foreground, 2girls squatting, 1girl standing
```

### Recommended settings

- **Steps:** 4-30
- **CFG:** 1-5.5
- **Sampler:** Euler
- **Size:** 832x1216, 1216x832
- **Strength:** < 0.8

### Sample prompts

**Prompt 1 (Bar counter deep squat):**
```
nc_girls, nc_scene, 1girl, solo, 1girl total in foreground, (squatting on bar counter:1.3), (deep squat:1.2), (full-squat:1.1), (buttocks below knees:1.1), (on high bar counter:1.2), (hands on butt:1.3), (grabbing butt:1.2), (spread legs:1.1), (from behind:1.3), (from side:1.2), (low angle:1.1), (pole dancing:1.3), (on pole:1.2), (dance pole:1.1), flexible, graceful, strong, clubwear, (bikini:1.2), panties, underwear, (revealing outfit:1.2), thighhighs, long hair, black hair, nightclub, bar area, indoors, red light, spotlight, disco lights, neon lit, stage lights, people in background, crowd, blurred background, bottle, cup, masterpiece, ultra detailed, photorealistic, volumetric lighting, red theme
```
Negative: `deformed hands, deformed feet, extra limbs, extra legs, bad anatomy, missing limbs, poorly drawn hands, poorly drawn feet, disfigured limbs, blurry, low quality, bad quality, worst quality, lowres,deformed knees,bumpy knees,red or pink dot on breasts,nude,naked,nipples`
Settings: Steps: 4, CFG scale: 1, Sampler: Euler

**Prompt 2 (Rhinestone bikini dancing):**
```
(clear platform high heels with clear straps:1.1), (provocative:1.1), (revealing outfit:1.2), (seductive pose:1.2), (sensual dance:1.3), 1girl total in foreground, from side and from behind, ass, back shot, bracelet, brown hair, (bejeweled bikini:1.4), (rhinestone bikini:1.3), (sparkling clothing:1.2), (gold bikini:1.1), dancing, disco lights, dj booth, dynamic pose, energetic, high heels, in nc_scene, jewelry, long hair, looking at viewer, masterpiece, nc_girls, neon lit, photorealistic, realistic, solo, spotlight, stage, stage lights, the heels' straps are visible, ultra detailed, wearing nc_heels, low angle, squatting, half-squat, slight squat, bending knees, knees bent, low stance, (legs bent:1.1), blue and yellow theme
```
Settings: Steps: 28, CFG scale: 3.5, Size: 832x1216

**Prompt 3 (3 girls group photo):**
```
(clear platform high heels with clear straps:1.1), (revealing outfit:1.2), 3girls total in foreground, 2girls from front, 1girl from side and behind, ass, bag, belt, black background, black bra, black gloves, black hair, black panties, blonde hair, bra, breasts, brown hair, cleavage, club corner, clubwear, dark background, dim lighting, earrings, elbow gloves, gloves, group photo, hat, high heels, holding, holding liquor bottle, in nc_scene, jewelry, long hair, looking at viewer, masterpiece, medium breasts, moody atmosphere, navel, nc_girls, panties, photo area, photorealistic, posing, realistic, sandals, skirt, smile, standing, sultry expression, the heels' straps are visible, toenails, ultra detailed, underwear, wearing nc_heels
```
Negative: `deformed hands, deformed feet, extra limbs, extra legs, bad anatomy, missing limbs, poorly drawn hands, poorly drawn feet, disfigured limbs, blurry, low quality, bad quality, worst quality, lowres,deformed knees,bumpy knees,red or pink dot on breasts,nude,naked,nipples`
Settings: Steps: 4, CFG scale: 1, Sampler: Euler

**Prompt 4 (Elevated platform kneeling):**
```
nightclub,nigtclub's stage,stage,((elevated platform:1.4)),
nc_girls,(1girl total, solo):1.4, from front and from side,from below,low angle, the girl kneeling on elevated platform while people under the platform and watching her, breasts, black hair, cleavage,ponytail,medium breasts,
clubwear,(revealing outfit:1.2),(white short dress:1.1), (white mini dress), (white silk choker),
looking at viewer, kneeling, full kneeling, both knees down, (knees on stage:1.2), (sitting on calves:1.1), static pose,
(provocative:1.1) (seductive pose:1.2),energetic,
the heels' straps are visible, ultra detailed, wearing nc_heels
disco lights, neon lit,spotlight,stage lights, photorealistic, realistic,masterpiece, ultra detailed
```
Negative: `deformed hands, deformed feet, extra limbs, extra legs, bad anatomy, missing limbs, poorly drawn hands, poorly drawn feet, disfigured limbs, blurry, low quality, bad quality, worst quality, lowres,deformed knees,bumpy knees,nc_heels,heels`
Settings: Steps: 4, CFG scale: 1, Sampler: Euler

**Prompt 5 (Pole dance on bar counter):**
```
(bare midriff:1.1), (on pole:1.3), (pole dancer:1.2), (revealing outfit:1.2), 1girl, bar area, girl is squatting on bar counter,grabbing the pole and pole dancing. bottle, bracelet, breasts, brown hair, clubwear, crop top, crowd, disco lights, earrings, flexible, graceful, in nc_scene, jewelry, long hair, looking at viewer, masterpiece, medium breasts, navel, navel piercing, nc_girls, neon lit, panties, people in background, photorealistic, piercing, pole, pole dance, pole dancing, shirt, solo, solo focus, stage lights, stripper, stripper pole, strong, ultra detailed, underboob, underwear, wristwatch,shoes not visible
```
Settings: Steps: 27, CFG scale: 3.5, Size: 1216x832

### Keywords

- `nc_girls` - **TRIGGER** for girls
- `nc_scene` - **TRIGGER** for scene
- `nc_heels` - **TRIGGER** for clear platform heels
- `nightclub`
- `clubwear`
- `disco lights`
- `neon lit`
- `spotlight`
- `stage lights`
- `people in background`
- `crowd`
- `elevated platform`
- `dance floor`
- `dj booth`
- `bar area`
- `pole dancing`
- `revealing outfit`
- `photorealistic`

### Notes

- **Strength < 0.8** recommended to avoid body merging issues
- Complex LoRA with scene, pose, and clothing concepts combined
- Use negative prompt for better anatomy results
- Clear platform heels are embedded in V1-alt version
- For Regional Prompting, use detailed multi-girl control
- Background crowd/people interactions require careful prompting
- Elevated platform scenes need extra prompt reinforcement
- Training: 250 images, batch size 4, 10 epochs, network_dim 32

---

[← Back to Index](INDEX.md)

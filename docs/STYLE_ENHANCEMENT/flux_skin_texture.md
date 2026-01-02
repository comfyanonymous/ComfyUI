# Flux Skin Texture

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 2244 |
| **👍** | 99 |
| **Tips** | 100 |

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

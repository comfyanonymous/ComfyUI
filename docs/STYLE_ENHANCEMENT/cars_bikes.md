# Cars & Bikes

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1569 |
| **👍** | 92 |
| **Tips** | 10 |

| Parameter | Value |
|-----------|-------|
| **File** | `cars__Bikes.safetensors` |
| **Civitai** | https://civitai.com/models/552837/cars-and-bikes-fluxsdxl |
| **Trigger word** | None |
| **Strength** | 0.6-1.1 |
| **Type** | Enhancement / Vehicles |
| **Compatibility** | FLUX, SDXL |

### Description
LoRA designed to enhance generation of realistic cars and bikes. Focuses on replicating specific brands and models more accurately. Trained on 50 high-quality vehicle images. Addresses common issue of non-specific or unrealistic vehicle designs.

### Recommended settings
- **CFG:** 1.5-3.5
- **Steps:** 20-30
- **Best results:** Specify brand, model, year, and unique features

### Sample prompts

**Prompt 1 (Futuristic sports car):**
```
Ultra-detailed, photorealistic digital photograph of a futuristic silver sports coupe with sleek, aerodynamic lines and closed gull-wing doors, parked on a glossy wet surface at dusk. Bright LED headlights cast crisp beams across the puddles, and dramatic studio-style lighting highlights the car's sculpted body panels. Shot as if with an 85 mm lens at f/1.8 for shallow depth of field and rich bokeh, 8K resolution, masterpiece. <lora:cars__Bikes:1>
```

**Prompt 2 (Car girl garage):**
```
Close-up shot of a car enthusiast, 'Car Girl', tinkering with her prized possession in a well-lit garage setting. Her messy blonde hair is tied back in a ponytail as she works underneath the hood of a sleek, black sports car, wrench and socket in hand. The composition is tight, focusing on Car Girl's determined expression and the intricate details of the engine. <lora:cars__Bikes:1>
```

**Prompt 3 (F1 racer NSFW):**
```
Formula 1 female racer in sleek sexy suit, Erotically sexy posed beside high-performance car on track, gazing with determination; metallic car reflects ambient lights, textured asphalt adds depth; pit crew and grandstands blurred in background, creating dynamic speed and anticipation; cinematic low-angle shot, ultra-realistic details, immersive depth, 4K resolution <lora:NSFW_master:0.7> <lora:cars__Bikes:0.6>
```

**Prompt 4 (Asian girl in Rolls Royce):**
```
realism, realistic girl, asian girl, sitting in a car, inside a rolls royce, sexy pose, naked breasts, black dress, white mink coat, glasses, pantyhose, photorealism, real girl, photo, dynamic pose, dynamic angle, hand near face, sexy look, 8k, hdr, realistic face, realistic skin <lora:cars__Bikes:1>
```

**Prompt 5 (BMW night drive):**
```
2019 bmw m5 wagon, outdoors at night, blue scheme, driving a car, blue flames surrounding, cinematic lighting, dark, no text, bmw e30, top down lighting, journalistic photography <lora:cars__Bikes:1>
```

**Prompt 6 (Night city chase):**
```
Create a vibrant scene of a sleek, futuristic sports car speeding through a bustling city at night. The city should be illuminated by neon lights reflecting off wet pavement. Capture the motion blur of the car's movement, the dynamic angles of skyscrapers, and include other traffic as blurred silhouettes in the background <lora:cars__Bikes:1.1>
```

**Prompt 7 (War-torn chase scene):**
```
Create a cinematic scene set in a chaotic, war-torn urban environment engulfed in flames and thick smoke. The foreground features a sleek, red high-performance race car with glowing blue headlights, speeding down a wet, reflective street. Behind it, a second dark-colored race car follows closely. Above, a military helicopter hovers. The background showcases towering buildings partially destroyed, with orange and yellow flames erupting from windows. <lora:cars__Bikes:0.75>
```

### Keywords
- Vehicle brands: `BMW`, `Rolls Royce`, `Ferrari`, etc.
- `sports car` / `race car`
- `motorcycle` / `bike`
- `high-performance`
- `aerodynamic`
- `sleek`
- `gull-wing doors`
- `LED headlights`

### Tested combinations

**Combination 1 (NSFW with car):**
```
<lora:cars__Bikes:0.6>
<lora:NSFW_master:0.7>
<lora:aidmaMJ6.1-FLUX-v0.5:0.6>
```

**Combination 2 (Hyperrealism):**
```
<lora:cars__Bikes:1>
<lora:aidmaHyperrealismv0.3:0.7>
<lora:aidmaImageUpraderv0.3:0.7>
```

**Combination 3 (Cinematic action):**
```
<lora:cars__Bikes:0.75>
<lora:FluxDFaeTasticDetails:0.75>
<lora:aidmaMJ6.1-FLUX-v0.5:0.35>
<lora:Phlux_V1:0.85>
```

**Combination 4 (Realistic interior):**
```
<lora:cars__Bikes:1>
<lora:Real_Nipples_Areola_Textures:1>
<lora:Hand_F1D_v2:1>
<lora:Skin_Texture_F1D:1>
```

### Use cases
- Specific car brand/model generation
- Racing scenes
- Car photography
- Automotive concept art
- Car + model scenes (NSFW capable)
- Night driving scenes
- Action chase sequences

### Notes
- Specify brand, model, year for best accuracy
- Works with both FLUX and SDXL
- Use img2img or inpainting to refine vehicle details
- Lower strength (0.6) when combining with NSFW LoRAs
- Higher strength (1.0-1.1) for dominant vehicle focus
- Combines well with detail enhancers and MJ style LoRAs

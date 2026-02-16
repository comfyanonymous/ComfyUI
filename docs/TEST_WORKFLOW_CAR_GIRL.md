# Test Workflow: Sexy Car Girl

## Concept
Sexy brunette in luxury lingerie (or nude) leaning against sports car.

## Character requirements
- Slim, sexy body
- Brunette hair
- Brown eyes
- Navy blue nails
- Luxury lingerie or nude

---

## Recommended LoRAs

| LoRA | Strength | Purpose |
|------|----------|---------|
| `FictiveCharacter1_flux_lora_v1` | 0.9-1.0 | Base character (ohwx) |
| `cars__Bikes` | 0.8-1.0 | Realistic car |
| `kicezaesthetics_V1` | 0.8 | Detail/color |
| `aidmaMJ6.1-FLUX-v0.5` | 0.5 | MJ style quality |
| `flux_realism_lora` | 0.5 | Realism boost |

---

## Test Prompts

### Prompt 1: Luxury Lingerie + Ferrari
```
Ultra-detailed photorealistic photograph of ohwx woman, sexy slim brunette with brown eyes, navy blue painted nails, leaning seductively against a red Ferrari 488 GTB. She is wearing black lace luxury lingerie set (bra and thong), garter belt with stockings. Her long dark hair flows over shoulders. Soft golden hour lighting, wet pavement reflections, cinematic bokeh background. Shot with 85mm f/1.4 lens, 8K resolution, masterpiece quality.
<lora:FictiveCharacter1_flux_lora_v1:1>
<lora:cars__Bikes:0.9>
<lora:kicezaesthetics_V1:0.8>
<lora:aidmaMJ6.1-FLUX-v0.5:0.5>
```

### Prompt 2: Nude + Lamborghini
```
Professional erotic photography of ohwx woman, stunning slim brunette model with brown eyes and navy blue manicure, completely nude, strategically posing against a white Lamborghini Huracan. Her athletic body curves highlighted by studio lighting. Dark wavy hair, seductive gaze at camera. Luxury garage setting with dramatic shadows. Ultra-realistic skin texture, perfect anatomy, 8K cinematic quality.
<lora:FictiveCharacter1_flux_lora_v1:1>
<lora:cars__Bikes:1>
<lora:MysticXXX-v6:0.7>
<lora:flux_realism_lora:0.6>
```

### Prompt 3: Sheer Robe + Porsche
```
Glamour photograph of ohwx woman, elegant slim brunette with brown eyes, navy blue nail polish, wearing sheer black silk robe barely covering her body, no underwear visible beneath. She leans provocatively on hood of silver Porsche 911 GT3. Nighttime city backdrop with neon reflections. Wet street, moody lighting, fashion magazine style. Hair styled in loose waves.
<lora:FictiveCharacter1_flux_lora_v1:0.95>
<lora:cars__Bikes:0.85>
<lora:FLUX_sexy_clothes_v3_Sevenof9:0.8>
<lora:kicezaesthetics_V1:0.7>
```

### Prompt 4: Topless + BMW M4
```
Candid amateur style photo of ohwx woman, topless slim brunette girlfriend with brown eyes, navy painted nails, sitting on hood of black BMW M4 Competition. She covers breasts playfully with hands. Wearing only unbuttoned denim shorts. Natural outdoor lighting, parking lot setting. Realistic skin texture, subtle freckles, genuine smile.
<lora:FictiveCharacter1_flux_lora_v1:1>
<lora:cars__Bikes:0.9>
<lora:amateur:0.8>
<lora:Cute_Belly_Button_by_Sarcastic_TOFU:0.6>
```

---

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 30-40 |
| **CFG** | 1-3 |
| **Guidance** | 2.5-3.5 |
| **Sampler** | Euler / DPM++ 2M |
| **Size** | 832x1216 or 896x1152 |
| **Upscale** | 1.5x with Remacri |

---

## Character Keywords
```
ohwx woman, slim brunette, brown eyes, navy blue nails, navy blue manicure, dark wavy hair, seductive, sexy, athletic body
```

## Car Keywords
```
sports car, Ferrari, Lamborghini, Porsche, BMW M4, luxury car, supercar, exotic car, wet pavement, reflections
```

## Lingerie Keywords
```
black lace lingerie, luxury lingerie, garter belt, stockings, sheer, silk robe, thong, bra
```

---

## Testing Notes

1. **Start with Prompt 1** - balanced lingerie + car
2. **Test different car brands** - some work better than others
3. **Adjust character LoRA strength** - 0.9-1.1 range
4. **Try with/without MJ style** - affects overall look
5. **Compare upscalers** - Remacri vs UltraSharp

## A/B Testing Ideas
- Same seed, different LoRA combinations
- Same prompt, different car brands
- Lingerie vs nude versions
- Day vs night lighting

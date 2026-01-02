# Landing Strip V2

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `Landing_Strip_V2.safetensors` |
| **Original filename** | `landingstrip.safetensors` |
| **Civitai** | https://civitai.com/models/1522417/landing-strip-pubic-hair-flux |
| **Trigger word** | `landingstrip` |
| **Strength** | 0.1-0.75 |
| **Type** | STYLE |
| **Compatibility** | FLUX |

## Description

Clean, simple landing strip pubic hair effect. Very flexible strength range - use 0.1 for subtle effect, 0.7+ for prominent. Works great with skin detail LoRAs like Skintastic and Detailed_imperfect_skin.

## Key features

- Simple trigger word `landingstrip`
- Very flexible strength (0.1-0.75)
- Clean landing strip effect
- Works well with skin detail LoRAs
- Good for artistic/fashion nude photography

## Recommended settings

- **Steps:** 32-40
- **CFG:** 1-3.5
- **Sampler:** Euler
- **Scheduler:** Simple
- **Strength:** 0.1-0.75

## Sample prompts

**Prompt 1 (Supermodel):**
```
cinematic film still supermodel standing, blue eyes, windblown hair, landingstrip, birkenstocks, woman, landingstrip, perfect hands. intricate details, highly detailed masterpiece, cinematic, detailed, realistic, hyper realistic, ultra detailed, 8k, realistic enhanced detail, masterpiece, high-definition, 4k 8k, score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up
```
Settings: Steps: 40, CFG: 3.5, Sampler: Euler

**Prompt 2 (Italian Dolomites):**
```
Skin imperfections, Freckles, moles, natural blemishes, nipples, landingstrip, Wide cinematic view high in the Italian Dolomites at a bright sunny day.
A famous beautiful 31-year-old Italian female TV presenter poses dynamically for a nude charity calendar.
athletically built but slightly skinny, natural medium saggy breasts with correct anatomical separation and realistic weight, toned abs, nicely trimmed pubic hair, obviously happy with a radiant smile, correct anatomical proportions, natural joint alignment, fingers anatomically accurate, no fused digits, no warped torso.
She wears good walking shoes, a jumper tied around her waist, her luxurious long hair waves powerfully in the strong mountain wind.
Shot on Leica M10 Monochrom with Noctilux 50 mm f/0.95, ultra-real skin pores, subsurface scattering, analog film grain, vibrant natural color photography, full color image, no black and white, rich color palette. 8K masterpiece.
<lora:Landing_Strip_V2:0.75> <lora:Detailed_imperfect_skin:0.15> <lora:aps_nipples:0.6>
```
Settings: Steps: 32, CFG: 1, Sampler: Euler simple, Size: 896x1344

**Prompt 3 (Italian painter Rosa):**
```
nipples, landingstrip, natural blemishes, stretch marks, NBClub, braless, unsupported breasts, skntstc, Cinematic three-quarter portrait of Rosa, a slightly curvy and firm Italian painter in her late 30s, standing barefoot in front of a weathered turquoise door in Ostuni, Apulia. Vertical 3:4 aspect ratio, framed from knees to above the head.
Her hair is tied up in a messy bun, with a few strands escaping, with textured waves and individual flyaway strands. Expression: playful, self-assured, with a hint of mischief.
<lora:Landing_Strip_V2:0.25> <lora:Skintastic:1.0> <lora:aps_nipples:0.55>
```
Settings: Steps: 32, CFG: 1, Sampler: Euler simple, Size: 992x1328

**Prompt 4 (Venice gondola):**
```
Skin imperfections, Freckles, moles, natural blemishes, Rough skin texture, Acne, stretch marks, nipples, landingstrip, Dense fog over a quiet Venetian canal at dawn, a gondola glides silently past, distant church bells echo.
A very popular beautiful 24-year-old female singer poses dynamically for a nude charity calendar while standing gracefully inside a small wooden gondola.
athletically built with surgically enlarged breasts with correct anatomical separation and realistic weight, toned abs, light blonde landing strip, subtle labia visible, obviously happy with a radiant smile.
correct anatomical proportions, natural joint alignment, fingers anatomically accurate, no fused digits, no warped torso.
She wears a long wide transparent dress slipped down over her shoulders, with a dramatic split revealing her legs, her luxurious platinum blonde, tightly curled hair waves powerfully in the cool morning breeze.
Shot on Leica M10 Monochrom with Noctilux 50 mm f/0.95, vibrant natural color photography, full color image, no black and white, rich color palette, ultra-real skin pores, subsurface scattering, analog film grain, 8K masterpiece.
<lora:Landing_Strip_V2:0.7> <lora:Detailed_imperfect_skin:0.15> <lora:aps_nipples:1.0>
```
Settings: Steps: 32, CFG: 1, Sampler: Euler simple, Size: 1024x1536

**Prompt 5 (Naples dancer):**
```
(landingstrip:0.60), skntstc, Canon R5C camera at a serene Naples beachfront promenade with the sea and distant Vesuvius at twilight. aspect-ration: 3:2.
Core motif is a full-body artistic nude of Serena the dancer, nipples and pubic area visible without open-leg positioning.
Serena in a dynamic, natural standing pose with weight shifted to one leg, arms raised in a dance extension, anatomically accurate with five fingers per hand, fluid limbs, graceful athletic form, completely nude, accessories limited to a seashell necklace and barefoot; focus on realistic neat, narrow landing strip of groomed pubic hair and body contours without exaggeration.
oval face with high cheekbones, expressive brown eyes, full lips; athletic-toned body with defined muscles, medium bust, narrow waist, wide hips; mediterranean olive skin with light freckles, pores, minor imperfections; realistic hair with strands blowing in wind, styled in wild loose curls, sparkling eyes with bright highlights.
Twilight lighting with purple hues and subtle ocean reflections, photorealistic nude style, intricate details on skin imperfections and hair movement, fabric-like wave patterns in background, 4K ultra-detailed, epic atmospheric quality.
<lora:Landing_Strip_V2:0.2> <lora:Skintastic:1.0> <lora:Nipples-under-clothes:0.6>
```
Settings: Steps: 32, CFG: 1, Sampler: Euler simple, Size: 1440x960

## Keywords

- `landingstrip` - **TRIGGER WORD**
- `landing strip`
- `nicely trimmed pubic hair`
- `groomed pubic hair`
- `narrow landing strip`

## Tested combinations

**Combination 1 (With skin detail):**
```
<lora:Landing_Strip_V2:0.75> <lora:Detailed_imperfect_skin:0.15>
```

**Combination 2 (With Skintastic):**
```
<lora:Landing_Strip_V2:0.25> <lora:Skintastic:1.0>
```

**Combination 3 (With nipples):**
```
<lora:Landing_Strip_V2:0.7> <lora:aps_nipples:0.6>
```

**Combination 4 (Full realistic):**
```
<lora:Landing_Strip_V2:0.7> <lora:KREAnsfw:0.7> <lora:Detailed_imperfect_skin:0.15> <lora:aps_nipples:1.0>
```

## Comparison with Landing Strip Pubic Hair

| Feature | Landing Strip V2 | Landing Strip Pubic Hair |
|---------|------------------|--------------------------|
| Trigger | `landingstrip` | `LandingStripPHair` |
| Type | STYLE | CONCEPT |
| Strength | 0.1-0.75 | 0.25-0.9 |
| Effect | Clean, simple | More detailed |

## Notes

- Simple trigger word `landingstrip`
- Very flexible strength (0.1 for subtle, 0.7+ for prominent)
- Works great with skin detail LoRAs (Skintastic, Detailed_imperfect_skin)
- Good for artistic/fashion nude photography

# Characters - Acorn Collection

[← Back to Main Index](../INDEX.md)

This section contains fictional character LoRAs created by author Acorn. These are repeatable, attractive female characters that are not based on real people. Each character has distinctive features and can be used to generate consistent character appearances across multiple images.

---

## Quality Statistics (Civitai)

*Updated: 2025-12-31*

| LoRA | Downloads | 👍 | Tips | Score |
|------|----------:|-------:|-----:|------:|
| [Acorn Gigi](acorn_gigi.md) | 1018 | 104 | 280 | ⭐ |
| [Acorn Annie](acorn_annie.md) | 838 | 109 | 0 | ⭐ |
| [Acorn Marina](acorn_marina.md) | 423 | 54 | 592 | - |
| [Acorn Natty](acorn_natty.md) | 304 | 30 | 10 | - |
| [Acorn Foxty](acorn_foxty.md) | 269 | 34 | 10 | - |
| [Acorn Jessica](acorn_jessica.md) | TBD | TBD | TBD | ⭐⭐⭐ |

**Legend:** ⭐⭐⭐ = Outstanding/Top tier | ⭐⭐ = High quality | ⭐ = Good

**Note:** Acorn Jessica is marked as Outstanding based on documentation review despite pending stats.

---

## Top Picks

| Category | Best LoRA | Why |
|----------|-----------|-----|
| **Overall Quality** | [Acorn Jessica](acorn_jessica.md) ⭐⭐⭐ | Outstanding photorealistic results with natural look. Best for production workflows. |
| **Most Popular** | [Acorn Gigi](acorn_gigi.md) ⭐⭐ | Highest downloads and tips. Versatile with wide strength range (0.6-1.4). |
| **Highest Rating** | [Acorn Annie](acorn_annie.md) ⭐⭐ | Best rating score. Excellent for pinup/vintage style and skin texture details. |
| **Versatility** | [Acorn Marina](acorn_marina.md) ⭐ | Works across wide strength range (0.25-1.0). Multiple hair color variations. |
| **Fantasy/Gothic** | [Acorn Foxty](acorn_foxty.md) ⭐ | Best for gothic, fantasy, and cinematic scenes with atmospheric lighting. |

---

## All Characters

### Main Collection

- [Acorn Jessica](acorn_jessica.md) ⭐⭐⭐ - **OUTSTANDING** photorealistic model with natural look. Use CFG 1, DPM2, 8-10 steps.
- [Acorn Gigi](acorn_gigi.md) ⭐⭐ - Slim build, long black hair, green eyes. Wide strength range 0.6-1.4.
- [Acorn Annie](acorn_annie.md) ⭐⭐ - Petite build, blonde wavy hair, blue eyes. Excellent skin texture capture.
- [Acorn Marina](acorn_marina.md) ⭐ - Fair skin, wavy hair (versatile colors). Very wide strength range 0.25-1.0.
- [Acorn Natty](acorn_natty.md) ⭐ - Curly blonde hair, light brown skin. Great for beach/outdoor scenes.
- [Acorn Foxty](acorn_foxty.md) ⭐ - Red/brown wavy hair. Versatile for casual, fantasy, gothic, cinematic scenes.

### Other Characters

- **Acorn Vikki** - File: `VikkiFlux.safetensors`, Trigger: `VikkiFlux`, Strength: 0.7-1.0

---

## Common Settings for Acorn Characters

Most Acorn character LoRAs work best with these settings:

| Parameter | Recommended Value |
|-----------|-------------------|
| **CFG** | **1** |
| **Steps** | **8-12** |
| **Sampler** | **DPM2** or **Euler** |
| **Scheduler** | **Beta** (alpha: 0.6, beta: 0.6) |
| **Resolution** | **1024x1280** to **1024x1496** |
| **Model** | Acorn Is Spinning Flux Hyper or AISF V1.1 DaChin Hyper 8 Step |
| **Distilled CFG** | 0 or 3.5 |

---

## Quick Reference - Triggers & Strengths

| LoRA | Trigger | Strength | Key Features |
|------|---------|----------|--------------|
| Acorn Jessica | `JessicaA` | 0.35-0.7 (0.55 typical) | Outstanding quality, LOW strength recommended |
| Acorn Gigi | `GigiFlux` | 0.6-1.4 (1.0 typical) | Wide range, long black hair, green eyes |
| Acorn Annie | `AnnieFlux` | 0.55-1.0 | Blonde wavy hair, blue eyes, petite |
| Acorn Marina | `MarinaA` | 0.25-1.0 (0.6-0.75) | Versatile strength, optional trigger |
| Acorn Natty | None | 0.7 | Curly blonde, light brown skin, beach scenes |
| Acorn Foxty | `Foxty` | 0.62-0.75 | Red/brown hair, versatile styles |

---

## Character Features Summary

| Character | Build | Hair | Eyes | Skin | Best Use |
|-----------|-------|------|------|------|----------|
| **Jessica** | Slim petite, wide hips | Variable | Variable | Real texture | Photorealistic, candid photography |
| **Gigi** | Slim petite, narrow hips | Long black | Green | Pale white | Pinup, retro, 80s style |
| **Annie** | Petite, fit | Blonde wavy | Blue | Pale | Vintage, pinup, beach |
| **Marina** | Slim physique | Wavy (brown/blonde) | Variable | Fair | Studio, dramatic lighting, retro |
| **Natty** | Slim, slender | Curly blonde | Variable | Light brown | Beach, outdoor, natural light |
| **Foxty** | Tiny build | Red/brown wavy | Dark green | Tan/neutral | Fantasy, gothic, cinematic |

---

## Recommended Combinations

### Jessica (Outstanding Quality)
```
<lora:JessicaA:0.55>
```
**Note:** Keep LoRA count minimal (1-2 additional max). Use Candid Photography style prompts.

### Gigi + Other LoRAs
```
<lora:GigiFlux_epoch_2:0.75> <lora:shaved pussy 05:1>
```

### Marina + Nipples Control
```
<lora:MarinaA:0.75> <lora:nipples:0.99>        # Enhanced nipples
<lora:MarinaA:0.6> <lora:nipples:-1>           # Hide nipples (clothed)
```

### Foxty + Film Effects
```
<lora:Acorn_Foxty_09:0.73> <lora:cinematic style film grain style film noise style v1:0.52> <lora:flux_train_replicate:0.65>
```

---

## Prompt Style Guidelines

### For Jessica (Candid Photography)
Start with `Candid Photography` and include:
- `real skin, textured skin`
- `(high saturation:1.2), (high contrast:1.1), (subsurface scattering:0.9)`
- `intricately detailed eyes`
- `slim petite body, wide hips`
- `warm rich dark color palette`

### For Gigi/Annie (Pinup/Retro)
- `cinematic dreamy erotic photo`
- `1960s style` or `1980's drive-in`
- `pinup art`, `retro`
- `real skin`, `skin textures`
- `subsurface scattering`

### For Foxty (Fantasy/Cinematic)
- Detailed atmospheric descriptions
- Gothic elements: `dark theme`, `smokey air`, `dim lighting`
- Cinematic composition details
- `High Detail, Perfect Composition, high contrast`

### For Natty (Natural/Beach)
- `natural lighting`
- `shallow depth of field`
- `beach`, `ocean`, outdoor settings
- Simple, clean prompts

---

## Notes

- **All Acorn characters work best with CFG 1** - higher CFG produces worse results
- **Use 8-12 steps with Hyper models** - more steps not needed
- **DPM2 or Euler samplers** with Beta scheduler (0.6/0.6) recommended
- **Jessica is the standout model** - marked OUTSTANDING for production use
- **Combine characters at reduced strength** when mixing multiple Acorn LoRAs
- Most characters work well with **Acorn Is Spinning Flux** checkpoint models

---

*Last updated: 2025-12-31*

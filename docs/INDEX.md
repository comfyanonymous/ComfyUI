# FLUX NSFW LoRA Documentation

Documentation for LoRA models used in generating realistic images with female anatomy focus.

---

## Table of Contents

| # | Category | File | Description |
|---|----------|------|-------------|
| 1 | [Checkpoints](01_CHECKPOINTS.md) | `01_CHECKPOINTS.md` | Base models and diffusion checkpoints |
| 2 | [Anatomy - Ass & Butt](02_ANATOMY_ASS.md) | `02_ANATOMY_ASS.md` | LoRAs for buttocks and rear anatomy |
| 3 | [Anatomy - Pussy & Vagina](03_ANATOMY_PUSSY.md) | `03_ANATOMY_PUSSY.md` | LoRAs for vaginal anatomy |
| 4 | [Anatomy - Breasts & Nipples](04_ANATOMY_BREASTS.md) | `04_ANATOMY_BREASTS.md` | LoRAs for breast and nipple generation |
| 5 | [Anatomy - Areolas](05_ANATOMY_AREOLAS.md) | `05_ANATOMY_AREOLAS.md` | LoRAs for areola variations |
| 6 | [Poses & Angles](06_POSES.md) | `06_POSES.md` | LoRAs for specific poses and camera angles |
| 7 | [Clothing & Fashion](07_CLOTHING.md) | `07_CLOTHING.md` | LoRAs for clothing effects and styles |
| 8 | [Ethnicity - Latina](08_ETHNICITY_LATINA.md) | `08_ETHNICITY_LATINA.md` | LoRAs for Latina/Hispanic characters |
| 9 | [Ethnicity - Asian](09_ETHNICITY_ASIAN.md) | `09_ETHNICITY_ASIAN.md` | LoRAs for Asian characters |
| 10 | [Ethnicity - Other](10_ETHNICITY_OTHER.md) | `10_ETHNICITY_OTHER.md` | LoRAs for Indian, Polynesian, etc. |
| 11 | [Characters - Acorn Collection](11_CHARACTERS_ACORN.md) | `11_CHARACTERS_ACORN.md` | Acorn series character LoRAs |
| 12 | [Characters - Other](12_CHARACTERS_OTHER.md) | `12_CHARACTERS_OTHER.md` | Other character LoRAs |
| 13 | [Body Types & Shape](13_BODY_TYPES.md) | `13_BODY_TYPES.md` | LoRAs for body shapes and physiques |
| 14 | [Style & Enhancement](14_STYLE_ENHANCEMENT.md) | `14_STYLE_ENHANCEMENT.md` | Style, realism, and quality enhancers |
| 15 | [Combinations & Tips](15_COMBINATIONS_TIPS.md) | `15_COMBINATIONS_TIPS.md` | Recommended combinations and guidelines |

---

## Quick Reference

### Most Used Checkpoints
- `fluxcstasyV1Fp16Fp8NF4_fp16V10.safetensors` - Best quality faces
- `flux1-dev-F16.gguf` - Standard FLUX Dev
- `fluxunchainedNF4_fluxunchainedV11NF4.safetensors` - Low VRAM option

### Essential Style LoRAs
- `MysticXXX-v6.safetensors` - NSFW enhancement
- `flux_realism_lora.safetensors` - Realism boost
- `amateur.safetensors` - Candid/amateur style

### Anatomy Essentials
- `Jib_Flux_Nipple_Fix_v2.safetensors` - Nipple fix
- `Ultimate_Realistic_Breast_GMR.safetensors` - Realistic breasts
- `Pussy_and_Ass_from_Behind.safetensors` - Rear anatomy

---

## File Locations

| Type | Path |
|------|------|
| Checkpoints/UNET | `ComfyUI\models\unet\` |
| LoRAs | `ComfyUI\models\loras\` |
| VAE | `ComfyUI\models\vae\` |
| CLIP | `ComfyUI\models\clip\` |

---

*Last updated: 2025-12-30*

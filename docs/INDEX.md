# FLUX NSFW LoRA Documentation

Documentation for LoRA models used in generating realistic images with female anatomy focus.

---

## Table of Contents

| # | Category | File | Description |
|---|----------|------|-------------|
| 1 | [Checkpoints](CHECKPOINTS/INDEX.md) | `CHECKPOINTS/` | Base models and diffusion checkpoints (17 files) |
| 2 | [Anatomy - Ass & Butt](ANATOMY_ASS/INDEX.md) | `ANATOMY_ASS/` | LoRAs for buttocks and rear anatomy (14 files) |
| 3 | [Anatomy - Pussy & Vagina](ANATOMY_PUSSY/INDEX.md) | `ANATOMY_PUSSY/` | LoRAs for vaginal anatomy (28 files) |
| 4 | [Anatomy - Breasts & Nipples](ANATOMY_BREASTS/INDEX.md) | `ANATOMY_BREASTS/` | LoRAs for breast and nipple generation (17 files) |
| 5 | [Anatomy - Areolas](ANATOMY_AREOLAS/INDEX.md) | `ANATOMY_AREOLAS/` | LoRAs for areola variations (4 files) |
| 6 | [Poses & Angles](POSES/INDEX.md) | `POSES/` | LoRAs for specific poses and camera angles (42 files) |
| 7 | [Clothing & Fashion](CLOTHING/INDEX.md) | `CLOTHING/` | LoRAs for clothing effects and styles (30 files) |
| 8 | [Ethnicity - Latina](ETHNICITY_LATINA/INDEX.md) | `ETHNICITY_LATINA/` | LoRAs for Latina/Hispanic characters (4 files) |
| 9 | [Ethnicity - Asian](ETHNICITY_ASIAN/INDEX.md) | `ETHNICITY_ASIAN/` | LoRAs for Asian characters (6 files) |
| 10 | [Ethnicity - Other](ETHNICITY_OTHER/INDEX.md) | `ETHNICITY_OTHER/` | LoRAs for Indian, Polynesian, etc. (2 files) |
| 11 | [Characters - Acorn Collection](CHARACTERS_ACORN/INDEX.md) | `CHARACTERS_ACORN/` | Acorn series character LoRAs (7 files) |
| 12 | [Characters - Other](CHARACTERS_OTHER/INDEX.md) | `CHARACTERS_OTHER/` | Other character LoRAs (14 files) |
| 13 | [Body Types & Shape](BODY_TYPES/INDEX.md) | `BODY_TYPES/` | LoRAs for body shapes and physiques (10 files) |
| 14 | [Style & Enhancement](STYLE_ENHANCEMENT/INDEX.md) | `STYLE_ENHANCEMENT/` | Style, realism, and quality enhancers (56 files) |
| 15 | [Combinations & Tips](15_COMBINATIONS_TIPS.md) | `15_COMBINATIONS_TIPS.md` | Recommended combinations and guidelines |
| 16 | [Upscalers](16_UPSCALERS.md) | `16_UPSCALERS.md` | Image upscaling models (Remacri, UltraSharp, etc.) |
| 17 | [Hands & Nails](HANDS/INDEX.md) | `HANDS/` | Hand anatomy fix, finger detail, nails (6 files) |
| 18 | [Characters - WATW](CHARACTERS_WATW/INDEX.md) | `CHARACTERS_WATW/` | Women Around The World character LoRAs |
| 19 | [Concepts](CONCEPTS/INDEX.md) | `CONCEPTS/` | Concept LoRAs (cum effects, etc.) |
| 20 | [Animals](ANIMALS/INDEX.md) | `ANIMALS/` | Animal LoRAs (cats, etc.) |
| 21 | [Video Generation](VIDEO/INDEX.md) | `VIDEO/` | Image-to-Video models (Wan 2.1) |

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
| Upscalers | `ComfyUI\models\upscale_models\` |

---

*Last updated: 2025-12-31*

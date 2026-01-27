# unStable Evolution KREA

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 14,577 |
| **👍** | 611 |
| **Tips** | 12,760 |

| Parameter | Value |
|-----------|-------|
| **File** | `unstableEvolution_Fp16_22GB.safetensors` |
| **Original filename** | `unstableEvolution_Fp1622GB.safetensors` |
| **Civitai** | https://civitai.com/models/1931032/unstable-evolution-krea |
| **Trigger word** | None |
| **Type** | Checkpoint / UNET (Flux.1 Krea based) |
| **Size** | 22GB (fp16) |

### Description

Realism of Flux Krea, improved with a lot of NSFW capabilities. Perfect in combination with Character LoRAs without changing the face. Works well for both NSFW and SFW content.

### Recommended Settings

| Parameter | Value |
|-----------|-------|
| **CFG scale** | 1 |
| **Distilled CFG Scale** | 2.5 |
| **Sampler** | Euler / Flux Realistic |
| **Schedule type** | Beta |
| **Beta schedule alpha** | 0.6 |
| **Beta schedule beta** | 0.6 |
| **Steps** | 25-30 |
| **Resolution** | 896x1152 (portrait) |

### CLIP Configuration

| Module | File |
|--------|------|
| **Module 1 (VAE)** | `ae.safetensors` |
| **Module 2 (CLIP L)** | `clip_l.safetensors` |
| **Module 3 (T5)** | `t5xxl_fp8_e4m3fn.safetensors` |

### Sample prompts

**Prompt 1 (Bathtub nude):**
```
This is a photograph of a woman lying nude in a white bathtub filled with water. Her hair is wet and spread out on the tub's surface. She is positioned on her back with her legs bent and spread apart, revealing her genitals. Her breasts and nipples are visible. The bathtub is set in a bathroom with tiled walls, featuring small, dark gray and beige tiles. The lighting is bright and even, suggesting the use of artificial lighting. The overall scene is intimate and candid.
```
Settings: Steps: 25, CFG: 1, Sampler: Euler, Scheduler: Beta

**Prompt 2 (Fashion/SFW):**
```
photography of a woman, dressed in black sports bra, black leggings, textured fabric, small white logo, sitting pose, one hand on chin, legs crossed, head slightly tilted, natural lighting, close-up shot, eye-level angle, colors, strong backlight, dramatic, decorative lighting, cinematic sensual, (from above:1.5), shot on Nikon Z50 with Nikon Z DX 16-50mm f-3.5-6.3
```
Settings: Steps: 30, CFG: 1, Sampler: Euler, Scheduler: Beta

**Prompt 3 (Explicit):**
```
The image is a high-resolution photography depicting an explicit sexual act. It shows a nude woman, performing oral sex on a man. The woman is kneeling on a carpet, gazing up at the camera with a neutral expression. Her lips are wrapped around the man's erect penis, which is prominently in the foreground. The man's lower torso and legs are visible, indicating he is standing. His skin tone is light, and he has a muscular build. The lighting is bright, suggesting the photo was taken with professional equipment in a well-lit indoor setting.
```
Settings: Steps: 25, CFG: 1, Sampler: Euler, Scheduler: Beta

### Keywords

- `high-resolution photography`
- `professional lighting`
- `nude`
- `natural lighting`
- `cinematic`
- Camera descriptions (e.g., `shot on Nikon Z50`)

### Notes

- **CFG must be 1** - this is a distilled model
- Works with `t5xxl_fp8_e4m3fn` for lower VRAM usage
- Best results with Beta scheduler (alpha: 0.6, beta: 0.6)
- Compatible with character LoRAs without face distortion
- Mostly for realism, but can handle some 3D styles
- No negative prompts tested/needed

### Workflow

See: `NSFW_Krea_Evolution.json`

---

*Added: 2026-01-23*

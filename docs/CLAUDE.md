# Claude Instructions for ComfyUI Documentation

## Overview

This file contains instructions for Claude on how to manage the ComfyUI documentation and install new models/LoRAs.

---

## Documentation Structure

All documentation is in `ComfyUI\docs\` folder with the following structure:

| File | Content |
|------|---------|
| `INDEX.md` | Main index with table of contents |
| `01_CHECKPOINTS.md` | Base models and diffusion checkpoints (unet/) |
| `02_ANATOMY_ASS.md` | LoRAs for buttocks and rear anatomy |
| `03_ANATOMY_PUSSY.md` | LoRAs for vaginal anatomy |
| `04_ANATOMY_BREASTS.md` | LoRAs for breast and nipple generation |
| `05_ANATOMY_AREOLAS.md` | LoRAs for areola variations |
| `06_POSES.md` | LoRAs for specific poses and camera angles |
| `07_CLOTHING.md` | LoRAs for clothing effects and styles |
| `08_ETHNICITY_LATINA.md` | LoRAs for Latina/Hispanic characters |
| `09_ETHNICITY_ASIAN.md` | LoRAs for Asian characters |
| `10_ETHNICITY_OTHER.md` | LoRAs for Indian, Polynesian, etc. |
| `11_CHARACTERS_ACORN.md` | Acorn series character LoRAs |
| `12_CHARACTERS_OTHER.md` | Other character LoRAs |
| `13_BODY_TYPES.md` | LoRAs for body shapes and physiques |
| `14_STYLE_ENHANCEMENT.md` | Style, realism, quality enhancers, NSFW unlock, cum effects |
| `15_COMBINATIONS_TIPS.md` | Recommended combinations and guidelines |

---

## Entry Format

Each LoRA/Model entry MUST follow this format:

```markdown
## Model Name

| Parameter | Value |
|-----------|-------|
| **File** | `filename.safetensors` |
| **Original filename** | `original_name.safetensors` (if renamed) |
| **Civitai** | https://civitai.com/models/XXXXX |
| **Trigger word** | `keyword` or None |
| **Strength** | 0.5-1.0 |
| **Type** | Enhancement / CONCEPT / etc. |

### Description
Brief description of what this LoRA does.

### Sample prompts

**Prompt 1 (description):**
```
prompt text here <lora:filename:1>
```

### Keywords
- `keyword1`
- `keyword2`

### Notes
- Additional notes if needed
```

---

## Installation Procedure

When user provides a new model to install:

### 1. Identify Type and Destination

| Type | User says | Destination folder |
|------|-----------|-------------------|
| Checkpoint/UNET | BASE MODEL | `ComfyUI\models\unet\` |
| LoRA | CONCEPT, STYLE, CHARACTER | `ComfyUI\models\loras\` |
| VAE | VAE | `ComfyUI\models\vae\` |
| Custom Node | TOOL, NODE | `ComfyUI\custom_nodes\` (git clone) |
| Workflow | WORKFLOW | `ComfyUI\user\default\workflows\` |

### 2. Copy File
```powershell
Copy-Item 'SOURCE_PATH' -Destination 'DEST_PATH' -Force -Verbose
```

### 3. Add Documentation

Choose the correct file based on LoRA type:
- **Checkpoints/UNET** → `01_CHECKPOINTS.md`
- **Anatomy (ass, butt)** → `02_ANATOMY_ASS.md`
- **Anatomy (pussy, vagina)** → `03_ANATOMY_PUSSY.md`
- **Anatomy (breasts, nipples)** → `04_ANATOMY_BREASTS.md`
- **Anatomy (areolas)** → `05_ANATOMY_AREOLAS.md`
- **Poses, angles** → `06_POSES.md`
- **Clothing, fashion** → `07_CLOTHING.md`
- **Latina ethnicity** → `08_ETHNICITY_LATINA.md`
- **Asian ethnicity** → `09_ETHNICITY_ASIAN.md`
- **Other ethnicity** → `10_ETHNICITY_OTHER.md`
- **Characters (Acorn)** → `11_CHARACTERS_ACORN.md`
- **Characters (other)** → `12_CHARACTERS_OTHER.md`
- **Body types** → `13_BODY_TYPES.md`
- **Style, enhancement, NSFW, cum effects** → `14_STYLE_ENHANCEMENT.md`
- **Combinations, tips** → `15_COMBINATIONS_TIPS.md`

### 4. Update Table of Contents

Add entry to the Table of Contents at the top of the relevant file.

### 5. Remove Source File
```powershell
Remove-Item 'SOURCE_PATH' -Force
```

---

## File Locations

| Type | Path |
|------|------|
| Checkpoints/UNET | `ComfyUI\models\unet\` |
| Diffusion Models | `ComfyUI\models\diffusion_models\` |
| LoRAs | `ComfyUI\models\loras\` |
| VAE | `ComfyUI\models\vae\` |
| CLIP | `ComfyUI\models\clip\` |
| Custom Nodes | `ComfyUI\custom_nodes\` |
| Workflows | `ComfyUI\user\default\workflows\` |
| Documentation | `ComfyUI\docs\` |

---

## Required Information

When user provides a new model, the following information is **REQUIRED**:

| Field | Required | Example |
|-------|----------|---------|
| **Civitai URL** | **YES** | `https://civitai.com/models/XXXXX` |
| **File path** | YES | `C:\Users\...\file.safetensors` |
| **Type** | YES | CONCEPT, STYLE, CHARACTER, BASE MODEL |
| **Trigger word** | If applicable | `keyword` or None |
| **Strength** | Recommended | 0.8-1.0 |

**IMPORTANT:** If user does not provide Civitai URL, **ASK FOR IT** before proceeding. Full documentation requires the source URL for reference.

---

## Quick Checklist

When adding new model:
- [ ] **Verify Civitai URL is provided** (ask if missing!)
- [ ] Identify type (CONCEPT, STYLE, BASE MODEL, TOOL)
- [ ] Copy to correct folder
- [ ] Add entry to correct documentation file (matching format)
- [ ] Update Table of Contents in that file
- [ ] Remove source file from downloads
- [ ] Confirm with user

---

*Last updated: 2025-12-30*

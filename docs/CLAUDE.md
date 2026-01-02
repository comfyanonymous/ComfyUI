# Claude Instructions for ComfyUI Documentation

## Overview

This file contains instructions for Claude on how to manage the ComfyUI documentation and install new models/LoRAs.

---

## Documentation Structure

All documentation is in `ComfyUI\docs\` folder with **folder-based structure**:

### Migrated Categories (Folder Structure)

| Folder | Content | Files |
|--------|---------|-------|
| `CHECKPOINTS/` | Base models and diffusion checkpoints | 17 |
| `ANATOMY_ASS/` | LoRAs for buttocks and rear anatomy | 14 |
| `ANATOMY_PUSSY/` | LoRAs for vaginal anatomy | 30 |
| `ANATOMY_BREASTS/` | LoRAs for breast and nipple generation | 17 |
| `ANATOMY_AREOLAS/` | LoRAs for areola variations | 4 |
| `POSES/` | LoRAs for poses and camera angles | 45 |
| `CLOTHING/` | LoRAs for clothing effects | 32 |
| `STYLE_ENHANCEMENT/` | Style, realism, NSFW unlock, cum effects | 57 |
| `ETHNICITY_LATINA/` | LoRAs for Latina/Hispanic characters | 4 |
| `ETHNICITY_ASIAN/` | LoRAs for Asian characters | 6 |
| `ETHNICITY_OTHER/` | LoRAs for Indian, Polynesian, etc. | 2 |
| `CHARACTERS_ACORN/` | Acorn series character LoRAs | 7 |
| `CHARACTERS_OTHER/` | Other character LoRAs | 14 |
| `BODY_TYPES/` | LoRAs for body shapes and physiques | 10 |
| `HANDS/` | Hand anatomy fix, finger detail, nails | 6 |
| `CHARACTERS_WATW/` | Women Around The World character LoRAs | 8+ |
| `CONCEPTS/` | Concept LoRAs (cum effects, etc.) | 4 |
| `ANIMALS/` | Animal LoRAs (cats, etc.) | 1 |

### Legacy Single Files (Small - No Migration Needed)

| File | Content |
|------|---------|
| `INDEX.md` | Main index with table of contents |
| `15_COMBINATIONS_TIPS.md` | Recommended combinations and guidelines |
| `16_UPSCALERS.md` | Image upscaling models (3 entries) |

### Folder Structure Details

Each migrated category folder contains:
```
CATEGORY_NAME/
├── INDEX.md           # Stats table, top picks, categorized lists
├── lora_name_1.md     # Individual LoRA documentation
├── lora_name_2.md
└── ...
```

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

**Filename conventions:**
- **NO SPACES** in destination filenames - use underscores `_` instead
- Replace spaces with underscores: `FLUX Female Anatomy.safetensors` → `FLUX_Female_Anatomy.safetensors`
- Avoid special characters (use only alphanumeric, underscore, hyphen)
- Use CamelCase or snake_case consistently
- If renaming, document **Original filename** in the entry

### 3. Add Documentation

Choose the correct folder/file based on LoRA type:

**Folder-based categories (preferred):**
- **Anatomy (ass, butt)** → `ANATOMY_ASS/` folder
- **Anatomy (pussy, vagina)** → `ANATOMY_PUSSY/` folder
- **Poses, angles** → `POSES/` folder
- **Clothing, fashion** → `CLOTHING/` folder
- **Style, enhancement, NSFW, cum** → `STYLE_ENHANCEMENT/` folder
- **Anatomy (breasts, nipples)** → `ANATOMY_BREASTS/` folder

**Legacy single files:**
- **Checkpoints/UNET** → `01_CHECKPOINTS.md`
- **Anatomy (areolas)** → `05_ANATOMY_AREOLAS.md`
- **Latina ethnicity** → `08_ETHNICITY_LATINA.md`
- **Asian ethnicity** → `09_ETHNICITY_ASIAN.md`
- **Other ethnicity** → `10_ETHNICITY_OTHER.md`
- **Characters (Acorn)** → `11_CHARACTERS_ACORN.md`
- **Characters (other)** → `12_CHARACTERS_OTHER.md`
- **Body types** → `13_BODY_TYPES.md`
- **Combinations, tips** → `15_COMBINATIONS_TIPS.md`

### 4. Update INDEX.md

**For folder-based categories:** Update the category's `INDEX.md` file.
**For legacy files:** Add entry to the Table of Contents at the top.

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
- [ ] Copy to correct folder (models/loras/, models/unet/, etc.)
- [ ] Fetch Civitai stats (downloads, 👍 thumbs, tips)
- [ ] Create individual .md file in appropriate category folder
- [ ] Update category's INDEX.md with new entry in stats table
- [ ] Remove source file from downloads
- [ ] Confirm with user

---

## Adding LoRA to Folder-Based Category (DETAILED)

When adding a new LoRA to a migrated folder-based category (e.g., `ANATOMY_ASS/`, `POSES/`, etc.):

### Step 1: Fetch Civitai Stats

Use WebFetch to get stats from Civitai API:
```
WebFetch: https://civitai.com/api/v1/models/XXXXX
Prompt: Extract: downloads count, thumbsUpCount (rating), tippedAmountCount (tips)
```

**Civitai Stats Fields:**

| Field | API Name | Description |
|-------|----------|-------------|
| **Downloads** | `stats.downloadCount` | Total downloads |
| **👍** | `stats.thumbsUpCount` | Unique positive reviews (thumbs up) |
| **Tips** | `stats.tippedAmountCount` | Total buzz tips received |

### Step 2: Create Individual LoRA File

Create new file: `CATEGORY/lora_name.md` using **snake_case** naming.

**Filename conventions:**
- `Flux Skirt Lift` → `flux_skirt_lift.md`
- `POV Blowjob + ASS` → `pov_blowjob_ass.md`
- `NSFW MASTER FLUX` → `nsfw_master_flux.md`

**File template:**
```markdown
# LoRA Name

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | X,XXX |
| **👍** | XXX |
| **Tips** | X,XXX |
| **Score** | ⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `filename.safetensors` |
| **Civitai** | https://civitai.com/models/XXXXX |
| **Trigger word** | `trigger` or None |
| **Strength** | 0.7-1.0 |
| **Type** | CONCEPT / STYLE / etc. |

## Description

Brief description of what this LoRA does.

## Sample prompts

**Prompt 1 (description):**
```
prompt text here
```
Settings: Steps: 30, CFG: 3.5, Sampler: Euler

## Keywords

- `keyword1`
- `keyword2`

## Notes

- Additional notes
```

### Step 3: Update Category INDEX.md

Add new entry to the **stats table** in INDEX.md:

```markdown
| [New LoRA Name](new_lora.md) | X,XXX | XXX | X,XXX | ⭐⭐ |
```

**Score ratings:**
- ⭐⭐⭐ = Top tier (>5K-10K downloads depending on category)
- ⭐⭐ = High quality (>2K downloads)
- ⭐ = Good (>500 downloads)
- `-` = New/Low stats

Also add to appropriate **category list** in INDEX.md.

### Step 4: Verify

Check that:
- [ ] Individual .md file created with all sections
- [ ] INDEX.md stats table updated (sorted by downloads desc)
- [ ] INDEX.md category list updated
- [ ] Back link works: `[← Back to Index](INDEX.md)`

---

## INDEX.md Template (for category folders)

```markdown
# Category Name

[← Back to Main Index](../INDEX.md)

Description of this category.

---

## Quality Statistics (Civitai)

*Updated: YYYY-MM-DD*

| LoRA | Downloads | 👍 | Tips | Score |
|------|----------:|-------:|-----:|------:|
| [Top LoRA](top_lora.md) | 10,000 | 500 | 1,000 | ⭐⭐⭐ |
| [Good LoRA](good_lora.md) | 5,000 | 250 | 500 | ⭐⭐ |

**Legend:** ⭐⭐⭐ = Top tier | ⭐⭐ = High quality | ⭐ = Good

---

## Top Picks

| Category | Best LoRA | Why |
|----------|-----------|-----|
| **Category A** | [LoRA Name](file.md) | Reason |

---

## All LoRAs by Category

### Subcategory 1
- [LoRA A](lora_a.md) ⭐⭐⭐ - Description
- [LoRA B](lora_b.md) ⭐⭐ - Description

---

## Quick Reference - Triggers

| LoRA | Trigger | Strength |
|------|---------|----------|
| LoRA Name | `trigger` | 0.8-1.0 |

---

*Last updated: YYYY-MM-DD*
```

---

## Creating Workflows

When creating new test/production workflows, **ALWAYS**:

### 1. Use Installed LoRAs Based on Documentation

Check INDEX.md and documentation files for:
- **RECOMMENDED** markers - priority LoRAs for workflows
- **⭐⭐⭐ CRITICAL** markers - must-use quality enhancers
- **"Planned for future workflow"** notes - tested good quality LoRAs

### 2. Standard Quality Stack

Always include these quality enhancers in workflows:

| LoRA | Strength | Purpose |
|------|----------|---------|
| `detail_enhancer_flux_v1` | 0.7 | ⭐⭐⭐ CRITICAL - details |
| `MysticXXX-v6` | 0.5-0.7 | NSFW unlock |
| `flux_realism_lora` | 0.5-0.7 | Realism boost |

### 3. Character Priority

Check for marked characters:
- **FictiveCharacter1** - RECOMMENDED sexy model (`ohwx` trigger)
- **JessicaA** (Acorn) - Outstanding natural results (`JessicaA` trigger)

### 4. Search Documentation

Before creating workflow, search docs for:
```
RECOMMENDED|CRITICAL|Planned|future workflow|Outstanding
```

### 5. Workflow Location

Save test workflows to: `ComfyUI\user\default\workflows\`

Naming: `TEST_[ConceptName]_v1.json`

---

---

## Missing Items (TODO)

### LoRAs to Install

The following LoRAs are mentioned in documentation but NOT installed:

| LoRA | Purpose | Documentation |
|------|---------|---------------|
| `Flux_Skin_Detailer` | Skin texture enhancement | Used in many combinations |
| `RealSkin_xxXL` | Realistic skin (SDXL) | 12_CHARACTERS_OTHER.md |
| `Realistic_Photos_Detailed_Skin` | Detailed skin textures | 12_CHARACTERS_OTHER.md |

### LoRAs Missing Documentation

The following LoRAs are installed but NOT documented:

| File | Folder | Needs |
|------|--------|-------|
| `pornmaster skin-IL-V1-lora.safetensors` | loras/ | Full documentation entry |

**Action:** When installing new LoRAs, always add documentation. When finding undocumented LoRAs, add entries to appropriate doc files.

---

## Updating Statistics

Use `update_stats.ps1` to refresh all Civitai stats in documentation:

```powershell
# Full update (all files)
.\update_stats.ps1 -Verbose

# Update specific category only
.\update_stats.ps1 -Category "POSES" -Verbose

# Preview changes without modifying files
.\update_stats.ps1 -DryRun

# Faster update (reduce API delay)
.\update_stats.ps1 -DelayMs 200
```

**What the script does:**
1. Scans all `.md` files for Civitai URLs
2. Fetches fresh stats from Civitai API
3. Updates Downloads, 👍, Tips, and Score in each file
4. Updates INDEX.md stats tables
5. Updates CLAUDE.md timestamp

**Run periodically** (e.g., weekly) to keep stats current for workflow decisions.

---

## GPU Compatibility (Blackwell / RTX 50 Series)

This system uses **NVIDIA RTX PRO 3000 Blackwell** (compute capability 12.0).

### Known Incompatibilities

| Package | Status | Issue | Solution |
|---------|--------|-------|----------|
| **xformers** | DO NOT INSTALL | Requires compute capability ≤9.0 | Use PyTorch native SDPA |

### xformers Error

If you see this error after installing xformers:
```
No operator found for `memory_efficient_attention_forward`
requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```

**Fix:**
```powershell
& 'C:\Users\spoko\www\ai\ComfyUI\venv\Scripts\pip.exe' uninstall xformers -y
```

Then restart ComfyUI. It will use PyTorch's native SDPA attention which supports Blackwell.

### Attention Mechanisms

| Mechanism | Blackwell Support | Performance |
|-----------|-------------------|-------------|
| PyTorch SDPA | YES | Good |
| xformers | NO | N/A |
| Flash Attention 3 | Check version | May work |

### Notes

- Blackwell GPUs (RTX 50 series, compute 12.0) are very new
- Most attention libraries need updates for compatibility
- PyTorch native attention works reliably
- Check for xformers updates periodically for Blackwell support

---

*Last updated: 2026-01-02*

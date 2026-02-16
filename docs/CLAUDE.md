# Claude Instructions for ComfyUI Documentation

## Overview

This file contains instructions for Claude on how to manage the ComfyUI documentation and install new models/LoRAs.

---

## Documentation Structure

All LoRA documentation lives in the **Starlight site** at `lora-docs-site/src/content/docs/` as `.mdx` files.

### Categories

| Path | Content |
|------|---------|
| `characters/acorn/` | Acorn series character LoRAs |
| `characters/other/` | Other character LoRAs |
| `characters/watw/` | Women Around The World character LoRAs |
| `anatomy/breasts/` | LoRAs for breast and nipple generation |
| `anatomy/pussy/` | LoRAs for vaginal anatomy |
| `anatomy/ass/` | LoRAs for buttocks and rear anatomy |
| `body-types/general/` | General body/anatomy LoRAs |
| `body-types/shape/` | Body shape LoRAs (hourglass, petite, etc.) |
| `body-types/special/` | Special body LoRAs (pregnant, belly button) |
| `clothing/panties-underwear/` | Panties, underwear effects |
| `clothing/skirts-upskirt/` | Skirts, skirt lifting, upskirt |
| `clothing/pantyhose-stockings/` | Pantyhose, stockings, garter belts |
| `clothing/lingerie-sheer/` | Lingerie, see-through, corsets |
| `clothing/shoes-footwear/` | Stiletto boots, platform heels, pumps |
| `clothing/tops-dresses-other/` | Tops, dresses, pants, swimwear |
| `poses/doggystyle-behind/` | Doggystyle, from behind, bent over |
| `poses/anal/` | Anal missionary, cowgirl, riding, fisting |
| `poses/sex-positions/` | DP, vaginal, cowgirl, dildo, machines |
| `poses/oral-tongue/` | Oral, tongue, cunnilingus, handjobs |
| `poses/sexy-poses/` | Modeling poses, shower, spreading |
| `poses/clothing-flashing/` | Clothing removal, flashing, mooning |
| `poses/facesitting-misc/` | Facesitting, car poses, camera angles |
| `style/aesthetic/` | Aesthetic style enhancers |
| `style/body/` | Body-focused style LoRAs |
| `style/detail/` | Detail enhancement LoRAs |
| `style/nsfw/` | NSFW unlock and nudity LoRAs |
| `checkpoints/` | Base models and diffusion checkpoints |

All paths are relative to `lora-docs-site/src/content/docs/`.

---

## Entry Format (.mdx)

Each LoRA entry is a `.mdx` file with YAML frontmatter and a LoraCard component:

```mdx
---
title: "LoRA Name"
description: "Category for FLUX"
downloads: 0
rating: 0
tips: 0
score: "-"
type: "Category"
trigger: "trigger_word"
strength: "1.0"
civitaiUrl: "https://civitai.com/models/XXXXX"
compatibility: "FLUX"
sidebar:
  badge:
    text: "-"
    variant: note
---

import LoraCard from '../../../components/LoraCard.astro';

<LoraCard
  title="LoRA Name"
  slug="category/lora_name"
  downloads={0}
  rating={0}
  tips={0}
  score="-"
  type="Category"
  trigger="trigger_word"
  strength="1.0"
  civitaiUrl="https://civitai.com/models/XXXXX"
  imageUrl="/images/loras/lora_name.jpg"
  compatibility="FLUX"
/>

## Description
Brief description of what this LoRA does.

## Sample prompts

**Prompt 1 (description):**
```
prompt text here <lora:filename:1>
```
Settings: Steps: 30, CFG: 3.5, Sampler: Euler

## Keywords
- `keyword1`
- `keyword2`

## Notes
- Additional notes
```

**Filename conventions:**
- Use **snake_case**: `Flux Skirt Lift` -> `flux_skirt_lift.mdx`
- Always `.mdx` extension
- Image: same base name with `.jpg` in `public/images/loras/`

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
- Replace spaces with underscores: `FLUX Female Anatomy.safetensors` -> `FLUX_Female_Anatomy.safetensors`
- Avoid special characters (use only alphanumeric, underscore, hyphen)
- If renaming, document **Original filename** in the entry

### 3. Add Documentation

Create a new `.mdx` file in the appropriate category under `lora-docs-site/src/content/docs/`:

| LoRA type | Target directory |
|-----------|-----------------|
| Anatomy (ass, butt) | `anatomy/ass/` |
| Anatomy (pussy, vagina) | `anatomy/pussy/` |
| Anatomy (breasts, nipples) | `anatomy/breasts/` |
| Poses, angles | `poses/<subcategory>/` |
| Clothing, fashion | `clothing/<subcategory>/` |
| Style, enhancement, NSFW | `style/<subcategory>/` |
| Characters | `characters/other/` (or `acorn/`, `watw/`) |
| Body types | `body-types/<subcategory>/` |
| Checkpoints | `checkpoints/` |

Use the `.mdx` entry template from the "Entry Format" section above. Adjust the `import LoraCard` path depth based on nesting level (`../../` for 2 levels, `../../../` for 3 levels).

Starlight auto-generates the sidebar from the file structure - no manual index updates needed.

### 4. Remove Source File
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
| Documentation (site) | `ComfyUI\docs\lora-docs-site\` |
| Documentation (content) | `ComfyUI\docs\lora-docs-site\src\content\docs\` |

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
- [ ] Fetch Civitai stats (downloads, thumbs, tips)
- [ ] Create `.mdx` file in appropriate category under `lora-docs-site/src/content/docs/`
- [ ] Download preview image to `lora-docs-site/public/images/loras/`
- [ ] Remove source file from downloads
- [ ] Confirm with user

---

## ⚠️ CRITICAL: No Duplicate Documentation

**RULE:** NEVER create new documentation files if documentation already exists for that LoRA. ALWAYS update existing files instead.

### Workflow Before Creating New Documentation

**STEP 1: Search for Existing Documentation**

Before creating any new `.mdx` file, ALWAYS search for existing documentation:

```bash
# Search by trigger word, filename, or CivitAI URL
grep -r "dpmp\|trigger_phrase\|civitai\.com/models/XXXXX" D:\AI\ComfyUI\docs/
```

Or use the Read tool with patterns to find files:
```bash
# Search all .mdx files for the LoRA name or keywords
find D:\AI\ComfyUI\docs -name "*.mdx" -o -name "*.md" | xargs grep -l "search_term"
```

### Common Locations to Check

Most LoRAs have documentation in these standard locations:

| Category | Path |
|----------|------|
| Poses (DP, positions) | `poses/sex-positions/` |
| Poses (doggystyle) | `poses/doggystyle-behind/` |
| Poses (anal) | `poses/anal/` |
| Poses (oral) | `poses/oral-tongue/` |
| Characters | `characters/other/` or `characters/acorn/` |
| Clothing | `clothing/<subcategory>/` |
| Anatomy | `anatomy/<subcategory>/` |
| Style/NSFW | `style/nsfw/` or `style/detail/` |

**File naming pattern:** `lora_name.mdx` in `lora-docs-site/src/content/docs/<category>/`

### If Documentation Already Exists

**DO THIS:**
1. Read the existing `.mdx` file
2. Identify which prompts are already documented
3. Filter the user's prompts - remove duplicates
4. Add ONLY the missing prompts using Edit tool
5. Update metadata (downloads, rating, tips) if needed
6. DO NOT create new files

**DON'T DO THIS:**
- ❌ Create new `.mdx` file with same LoRA (creates duplicate)
- ❌ Create `.md` file if `.mdx` already exists
- ❌ Ignore existing documentation and start from scratch

### If Documentation Does NOT Exist

**THEN:**
1. Create new `.mdx` file in appropriate category
2. Use template from "Entry Format" section
3. Add all prompts provided by user
4. Update metadata from CivitAI

### Example: Correct vs Incorrect

**INCORRECT (what I did initially):**
```
# Found dpmpFLUX documentation already exists:
poses/sex-positions/double_penetration_missionary_flux.mdx

# Wrongly created NEW files:
poses/dpmp_flux.mdx          ❌ DUPLICATE
POSES/dpmp_flux.md           ❌ DUPLICATE

# Result: Overwrote metadata, had to merge prompts manually
```

**CORRECT:**
```
# Found existing documentation:
poses/sex-positions/double_penetration_missionary_flux.mdx (has 6 prompts)

# Read existing file
# Compared user's ~20 prompts with existing 6
# Added ONLY 6 missing prompts using Edit tool
# Deleted wrongly created duplicates
# Result: Clean, no overwrites, proper merge
```

---

## Fetching Civitai Stats

Use WebFetch to get stats from Civitai API:
```
WebFetch: https://civitai.com/api/v1/models/XXXXX
Prompt: Extract: downloads count, thumbsUpCount (rating), tippedAmountCount (tips)
```

**Civitai Stats Fields:**

| Field | API Name | Description |
|-------|----------|-------------|
| **Downloads** | `stats.downloadCount` | Total downloads |
| **Rating** | `stats.thumbsUpCount` | Unique positive reviews (thumbs up) |
| **Tips** | `stats.tippedAmountCount` | Total buzz tips received |

**Score ratings:**
- `⭐⭐⭐` = Top tier (>10K downloads)
- `⭐⭐` = High quality (>2K downloads)
- `⭐` = Good (>500 downloads)
- `-` = New/Low stats

---

## Updating Statistics

Use the npm pipeline to refresh all Civitai stats:

```bash
cd D:\AI\ComfyUI\docs\lora-docs-site
npm run update-stats
```

This runs two scripts in sequence:
1. `fetch-civitai-images.js` - Fetches stats and images from Civitai API, updates the cache
2. `update-stats-from-cache.js` - Updates all `.mdx` files with cached stats (downloads, rating, tips, score)

**Run periodically** (e.g., weekly) to keep stats current for workflow decisions.

---

## Creating Workflows

When creating new test/production workflows, **ALWAYS**:

### 1. Use Installed LoRAs Based on Documentation

Search `.mdx` files in `lora-docs-site/src/content/docs/` for:
- **RECOMMENDED** markers - priority LoRAs for workflows
- High-score LoRAs (`⭐⭐⭐`)

### 2. Standard Quality Stack

Always include these quality enhancers in workflows:

| LoRA | Strength | Purpose |
|------|----------|---------|
| `detail_enhancer_flux_v1` | 0.7 | CRITICAL - details |
| `MysticXXX-v6` | 0.5-0.7 | NSFW unlock |
| `flux_realism_lora` | 0.5-0.7 | Realism boost |

### 3. Character Priority

Check for marked characters:
- **FictiveCharacter1** - RECOMMENDED sexy model (`ohwx` trigger)
- **JessicaA** (Acorn) - Outstanding natural results (`JessicaA` trigger)

### 4. Workflow Location

Save test workflows to: `ComfyUI\user\default\workflows\`

Naming: `TEST_[ConceptName]_v1.json`

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

---

## LoRA Documentation Site (lora-docs-site)

### Configuration

**Location:** `D:\AI\ComfyUI\docs\lora-docs-site\`

**Port:** Port 8080 (configured in `package.json`)

**Start Server:**
```bash
cd D:\AI\ComfyUI\docs\lora-docs-site
npm run dev
```

**Access:** http://localhost:8080/

### File Structure

```
lora-docs-site/
├── src/
│   ├── content/
│   │   └── docs/           # All .mdx LoRA documentation
│   │       ├── characters/
│   │       │   ├── acorn/
│   │       │   ├── other/
│   │       │   └── watw/
│   │       ├── anatomy/
│   │       │   ├── breasts/
│   │       │   ├── pussy/
│   │       │   └── ass/
│   │       ├── clothing/
│   │       │   ├── panties-underwear/
│   │       │   ├── skirts-upskirt/
│   │       │   ├── pantyhose-stockings/
│   │       │   ├── lingerie-sheer/
│   │       │   ├── shoes-footwear/
│   │       │   └── tops-dresses-other/
│   │       ├── poses/
│   │       │   ├── doggystyle-behind/
│   │       │   ├── anal/
│   │       │   ├── sex-positions/
│   │       │   ├── oral-tongue/
│   │       │   ├── sexy-poses/
│   │       │   ├── clothing-flashing/
│   │       │   └── facesitting-misc/
│   │       ├── style/
│   │       │   ├── aesthetic/
│   │       │   ├── body/
│   │       │   ├── detail/
│   │       │   └── nsfw/
│   │       ├── body-types/
│   │       │   ├── general/
│   │       │   ├── shape/
│   │       │   └── special/
│   │       └── checkpoints/
│   └── components/
│       ├── LoraCard.astro
│       └── LoraGrid.astro
├── scripts/
│   ├── fetch-civitai-images.js   # Fetch stats + images from Civitai API
│   └── update-stats-from-cache.js # Update .mdx files from cache
├── public/
│   └── images/
│       └── loras/           # Preview images (JPEG)
├── civitai-image-cache.json # API response cache
├── package.json
└── astro.config.mjs
```

### Common Issues & Troubleshooting

#### Issue: "Tiles not loading in categories" / Parse errors

**Symptoms:**
- Category pages show no LoRA cards
- Console errors: "Failed to parse source for import analysis"

**Solution:**
```bash
# 1. Stop dev server
# 2. Clear caches
rm -rf D:\AI\ComfyUI\docs\lora-docs-site\.astro
rm -rf D:\AI\ComfyUI\docs\lora-docs-site\node_modules\.vite

# 3. Restart dev server
cd D:\AI\ComfyUI\docs\lora-docs-site
npm run dev
```

#### Issue: MDX Parse Errors

**Common Causes:**
1. **Unescaped quotes in JSX attributes** - use `&quot;`
2. **Missing or malformed frontmatter** - downloads must be number, not string
3. **Invalid JSX syntax in LoraCard** - numeric values use `{number}` not `"number"`

#### Issue: Images not loading

**Cause:** Missing image files in `public/images/loras/`

**Image Naming Convention:**
- Same base name as `.mdx` file with `.jpg` extension
- Example: `watw_japan.mdx` -> `watw_japan.jpg`

---

## Missing Items (TODO)

### LoRAs to Install

| LoRA | Purpose |
|------|---------|
| `Flux_Skin_Detailer` | Skin texture enhancement |
| `RealSkin_xxXL` | Realistic skin (SDXL) |
| `Realistic_Photos_Detailed_Skin` | Detailed skin textures |

### LoRAs Missing Documentation

| File | Folder | Needs |
|------|--------|-------|
| `pornmaster skin-IL-V1-lora.safetensors` | loras/ | Full documentation entry |

**Action:** When installing new LoRAs, always add documentation. When finding undocumented LoRAs, add entries to appropriate doc files.

---

---

## LoRA Update Procedure (Proper Workflow)

When user provides prompts to add to an existing LoRA:

### Step 1: Verify LoRA Exists
```bash
# Check if LoRA file is installed
ls -lh /d/AI/ComfyUI/models/loras/LORA_NAME.safetensors
```

### Step 2: Search for Existing Documentation
```bash
# Search for any existing .mdx documentation
grep -r "trigger_word\|lora_name\|civitai\.com/models/ID" D:\AI\ComfyUI\docs/lora-docs-site/src/content/docs/ --include="*.mdx"
```

### Step 3: Read Existing File
If found, read the existing `.mdx` file to see:
- Which prompts already exist
- Current metadata (downloads, rating, tips)
- Category/subcategory location

### Step 4: Identify New Prompts
- Compare user's prompts with existing ones
- Filter out exact duplicates and near-duplicates
- Keep unique variations with different seeds/settings

### Step 5: Update Existing File
Use Edit tool to add missing prompts to existing file:
- Keep section structure consistent
- Add prompts with proper numbering (continuing from last)
- Mark interesting/special prompts with ⭐ if user indicates

### Step 6: Update Metadata (if changed)
Update YAML frontmatter if CivitAI stats changed:
- `downloads`, `rating`, `tips`, `score`

### Step 7: Clean Up
- Delete any duplicate `.mdx` or `.md` files you may have created
- Verify no duplicate files exist for same LoRA

---

*Last updated: 2026-02-15*

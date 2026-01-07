# Concept - Partially Removed (and Open) Clothes

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 145 |
| **👍** | 25 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Partially_Removed_Clothes.safetensors` |
| **Original filename** | `concept_prtclothes_style.safetensors` |
| **Civitai** | https://civitai.com/models/2169699/concept-partially-removed-and-open-clothes |
| **Trigger word** | `prtclothes_style` |
| **Strength** | 0.6-0.8 (with other LoRAs), 1.0 (standalone) |
| **Type** | CONCEPT |

## Description

Trained on 436 images with ~150-160 different outfits to teach FLUX how clothing looks when partially removed. Focuses on garments that are lifted, unbuttoned, draped, or unzipped to reveal the body while maintaining natural, photographic style.

**IMPORTANT:** This is NOT a breasts/nipple LoRA! It focuses solely on "how does clothing look if it's partially removed?" Use with MysticXXX or NSFW checkpoint for proper anatomy.

## Covered Clothing Types

### Outerwear
- Open robes, velvet capes
- Trench coats, fur-lined coats
- Leather jackets, denim vests
- Loose knit sweaters, cardigans

### Tops
- Off-shoulder blouses
- Lifted turtlenecks
- Unbuttoned shirts
- Plaid shirts

### Lingerie
- Lace bras, bralettes
- Bodysuits, panties
- Mesh/satin robes

### Materials
- Velvet, silk, satin, lace
- Mesh, cotton knits
- Denim, leather
- Faux fur, wool

## Training Details

- 436 images, ~150-160 outfits (max 3 pics/outfit)
- 2 steps per image, 10 epochs
- Buckets: 832x1280 px
- Cosine scheduler (0.2 warmup, 0.8 decay)
- network_dim/alpha: 32
- Trained on default flux1-dev

## Sample Prompts

**Knit Sweater Lifted:**
```
prtclothes_style, Full body shot of woman standing by an ornate column. She wears a fitted red knit sweater lifted to expose her bare breasts, black lace mini skirt, and opaque black tights. Her look is confident and direct. Accessories include a vintage black beret and a red silk scarf tied to her handbag.
```
Settings: Steps: 30, Sampler: Euler a

**Plaid Shirt Open:**
```
prtclothes_style, Upper body shot, woman standing in front of a fireplace, wearing open red plaid shirt, bare chest, black lace bra. Neutral expression. Accessories: thin black choker and small silver studs.
```
Settings: Steps: 30, Sampler: Euler a

**Velvet Cape:**
```
prtclothes_style, A full body shot of a woman standing confidently with one hand on her hip. She wears a dark green velvet cape draped over her shoulders, partially open to reveal bare breasts. Her face shows a soft, sensual smile. She is adorned with a thin gold headband and small gold hoop earrings.
```
Settings: Steps: 30, Sampler: Euler a

**Denim Shorts + Unbuttoned Shirt:**
```
prtclothes_style, A full body shot of a woman standing in front of a window. She wears frayed denim shorts and a loose, white cotton shirt, unbuttoned to show bare breasts. Her face is smiling, warm and open. She accessorizes with a vibrant patterned scarf around her neck.
```
Settings: Steps: 30, Sampler: Euler a

**Leather Jacket:**
```
prtclothes_style, An upper body shot of a woman sitting cross-legged on a patterned bedspread. She wears a fierce black faux-leather jacket partially open with nothing underneath except a visible gold choker. Her expression is sultry and intense.
```
Settings: Steps: 30, Sampler: Euler a

**Silk Robe in Forest:**
```
prtclothes_style, Full body shot of a woman on a green blanket in a forest, kneeling with hands behind head. She has an open blue silk robe with a floral pattern, bare breasts visible. Her face is sensual, eyes half-closed.
```
Settings: Steps: 30, Sampler: Euler a

## Keywords

- `prtclothes_style`
- `open shirt`
- `unbuttoned`
- `lifted sweater`
- `open robe`
- `partially open`
- `bare breasts visible`

## Recommended Combinations

Use with:
- **MysticXXX** (0.5-0.7) - For proper nipple/anatomy rendering
- **NSFW MASTER FLUX** - Base model compatibility
- Character LoRAs at 0.6-0.8 strength

## Notes

- Use trigger `prtclothes_style` at start of prompt
- Lower strength to 0.6-0.8 when using with character LoRAs
- NOT trained for nipple detail - use inpaint or anatomy LoRAs
- May cause memory issues in SwarmUI/A1111 - use ComfyUI or lower TE1 weights
- Large file size due to network_dim/alpha=32

---

*Last updated: 2026-01-02*

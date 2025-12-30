# 15. Combinations & Tips

[← Back to Index](INDEX.md)

This section provides recommended LoRA workflow combinations for different scenarios and essential tips and guidelines for effective use of LoRAs in your image generation workflows.

---

# Section 14: Recommended Workflow Combinations

Common LoRA combinations for different scenarios.

## NSFW Character Generation
```
<lora:MysticXXX-v6:0.7>
<lora:Jib_Flux_Nipple_Fix_v2:0.8>
<lora:amateur:0.5>
<lora:detail_enhancer_flux_v1:0.6>
```

## Latina/Hispanic Characters
```
<lora:Latinas-000002:0.8>
<lora:MysticXXX-v6:0.7>
<lora:flux_realism_lora:0.6>
```

## South Asian/Indian Characters
```
<lora:desiespresso_flux_v2:0.8>
<lora:detail_enhancer_flux_v1:0.7>
```

## Realistic Anatomy (GMR Series)
```
<lora:Ultimate_Realistic_Breast_GMR:0.8>
<lora:Ghost_Areolas_GMR:0.5>
<lora:Skinny_Legs_Ass_GMR:0.8>
```

## Amateur/Candid Style
```
<lora:amateur:0.8>
<lora:MysticXXX-v6:0.5>
```

---

# Tips & Guidelines

## General Tips
1. **Start with lower LoRA strengths** (0.5-0.7) and increase if needed
2. **Combine complementary LoRAs** - anatomy + style + character
3. **Use trigger words** when specified - they activate the LoRA properly
4. **Test combinations** - some LoRAs work better together

## Recommended Workflow Order
1. Base character LoRA (Acorn, MysticXXX, etc.)
2. Anatomy enhancement (nipples, areolas, etc.)
3. Style/realism LoRA (amateur, detail enhancer)
4. Ethnicity LoRA if needed (desiespresso, latina)

## Important Notes
- Always check Civitai for latest versions and updates
- Some LoRAs require specific base models (FLUX, Pony, etc.)
- Negative prompts can significantly improve results

---

# Workflows

## FLUX - Realistic Amateurs v1.4

| Parameter | Value |
|-----------|-------|
| **Civitai** | https://civitai.com/models/736681/flux-realistic-amateurs |
| **Workflow file** | `user\default\workflows\flux_realistic_amateurs_v1_4.json` |
| **Wildcards file** | `custom_nodes\ComfyUI-Impact-Pack\wildcards\realistic_amateur_v1.yaml` |
| **Version** | 1.4 |

### Description
Complete ComfyUI workflow for generating infinite realistic amateur women using FLUX, LoRAs, and wildcards. Creates candid, amateur-style photos with randomized characteristics.

### Required Models

| Model | Location |
|-------|----------|
| Acorn Is Spinning FLUX | `ComfyUI/models/unet` |
| VAE (HuggingFace) | `ComfyUI/models/vae` |
| Text Encoder | `ComfyUI/models/clip` |
| Clip_l | `ComfyUI/models/clip` |

### Recommended LoRAs

**Nudity (Choose 1):**
- FLUX - Female Anatomy
- Flux Topless
- Perfect Full Round Breasts & Slim Waist

**Style (Choose 1):**
- Vintage Neon Film
- Vi-FluxFinePortrait

**Utility:**
- SameFace Fix [Flux Lora] - Avoids same-face syndrome

### Required Custom Nodes
Install via ComfyUI Manager:
- ComfyRoll Custom Nodes
- rgthree
- ComfyUI Impact Pack

### Key Feature - File Path Prompting
V1.4 introduces a unique technique: adding fake file paths to prompts creates a very candid amateur look.

### Sample prompts

**Prompt 1 (College dorm party):**
```
<lora:flux\SameFace_Fix:-0.45>C:\Users\Charlotte White\Photos\webcam\college\dorm\party\1974\IMG-6562.JPG Amateur, No watermark, poor lighting. beautiful face. She has a fat body. and large breasts. She has no makeup. She has ginger very long hair, ponytail hair, bangs, short bangs. She is wearing lacy, sexy bra She is blushing with rosy cheeks and has a nervous, shy, happy, ecstatic smile.
```

**Prompt 2 (Office girl with wildcards):**
```
<lora:flux\SameFace_Fix:-0.45>C:\Users\__a/part/female/name__\Photos\webcam\office girl\2002\IMG-6562.JPG Amateur, No watermark, poor lighting. beautiful face. __a/part/female/body__ and large sagging massive natural breasts. She is wearing makeup. She is blushing with rosy cheeks and has a nervous, shy, happy, ecstatic smile.
```

**Prompt 3 (Nude with wildcards):**
```
<lora:flux\SameFace_Fix:-0.45>C:\Users\__a/part/female/name__\Photos\webcam\college\dorm\party\2010\IMG-6562.JPG Amateur, No watermark, poor lighting. __a/part/female/body__ and large sagging massive natural breasts. __a/part/female/hair__ She is wearing nothing She is blushing with rosy cheeks and has a nervous, shy, happy, ecstatic smile.
```

**Prompt 4 (Goth girl):**
```
<lora:flux\SameFace_Fix:-0.45>C:\Users\__a/part/female/name__\Photos\webcam\goth girl\2009\IMG-6562.JPG Amateur, No watermark, poor lighting. sexy face. __a/part/female/body__ and large sagging massive natural breasts. She is wearing makeup. She is blushing with rosy cheeks and has a nervous, shy, happy, ecstatic smile.
```

### Wildcards Usage
The workflow uses `__wildcard/path__` syntax for randomization:
- `__a/part/female/name__` - Random female names
- `__a/part/female/body__` - Body type descriptions
- `__a/part/female/hair__` - Hair styles and colors

### Path Modification Tips
Modify the fake file path for different results:
- Change folder names: `college`, `office girl`, `goth girl`
- Change years: `1974`, `2002`, `2010`
- Change usernames for character variety

### Version History
- **V1.4:** File path prompting technique, removed most default LoRAs
- **V1.3:** Image Saver nodes, speed optimization
- **V1.2:** Added SameFace Fix, simplified prompts
- **V1.1:** More backgrounds, clothes, expressions

---

[← Back to Index](INDEX.md)

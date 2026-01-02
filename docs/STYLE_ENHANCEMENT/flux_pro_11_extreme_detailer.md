# FLUX Pro 1.1 Extreme Detailer

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 51271 |
| **👍** | 3382 |
| **Tips** | 2510 |

| Parameter | Value |
|-----------|-------|
| **File** | `aidmaFLUXPro1.1-FLUX-v0.3.safetensors` |
| **Civitai** | https://civitai.com/models/832683/flux-pro-11-style-lora-extreme-detailer-for-flux-illustrious |
| **Trigger word** | `aidmafluxpro1.1` |
| **Strength** | 0.4-1.0 |
| **Type** | STYLE / Detail Enhancer |
| **Version** | v0.3 |

### Description
Extreme detailer trained on FLUX Pro 1.1 images. Very strong LoRA - use it as a spice. Can change image considerably at high strength, or just add more detail at lower strength.

### Versions
- **Strong Version:** Can change image considerably - compare outputs to find right strength
- **Light Version:** Mostly adds detail without changing composition

### Sample prompts

**Prompt 1 (Portrait):**
```
aidmafluxpro1.1, portrait of a woman, highly detailed skin, professional photography <lora:aidmaFLUXPro1.1-FLUX-v0.3:0.6>
```

**Prompt 2 (Scene):**
```
aidmafluxpro1.1, fantasy landscape, intricate details, 8k resolution <lora:aidmaFLUXPro1.1-FLUX-v0.3:0.5>
```

### Keywords
- `aidmafluxpro1.1` - **TRIGGER WORD**

### Notes
- **Very strong** - start with lower strength (0.4-0.5)
- Use as "spice" - adds extreme detail
- Higher strength (0.7-1.0) may change image significantly
- Lower strength (0.4-0.6) for detail without composition change
- Works with FLUX and Illustrious

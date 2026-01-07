# Flux Dev AI Model 2.1 (Lya)

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 127 |
| **👍** | 5 |
| **Tips** | 0 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Lya_Flux.safetensors` |
| **Original filename** | `Flux_Dev_AI_Model_2.1-000017.safetensors` |
| **Civitai** | https://civitai.com/models/1093208 |
| **Trigger word** | `Lya` |
| **Strength** | 1.0 |
| **Type** | CHARACTER |

## Character Profile

| Attribute | Value |
|-----------|-------|
| **Name** | Lya |
| **Hair** | Long, slightly wavy, dark brown |
| **Style** | Elegant, relaxed |
| **Settings** | Studio, cozy interiors |

## Description

AI-generated character LoRA featuring Lya - a woman with long, slightly wavy dark brown hair. Trained on Flux.1 Dev, Basic Model Version 1, Epoch 17. Works well in studio settings, elegant poses, and cozy atmospheres.

**Capabilities:**
- Consistent character appearance
- Studio photography style
- Elegant and relaxed poses
- Natural lighting

## Sample Prompts

### Studio Portrait
```
creating a cozy atmosphere. In the foreground, The image features Lya, enhancing the tranquil mood of the scene., her long and slightly wavy dark brown hair falls naturally over her shoulders. She is positioned sideways, emphasizing her relaxed yet elegant demeanor. Natural light illuminates her face. This adds a touch of elegance to her appearance. The background appears like a softly blurred studio setting, including delicately shaped eyebrows
<lora:Lya_Flux:1>
```
Settings: Steps 30, CFG 7.5, Sampler: DDIM, Size: 512x512

### Industrial Studio Fashion
```
Lya perched on a stool in an industrial-style studio, wearing edgy, modern fashion, dramatic contrasts in lighting. (full shot).
<lora:Lya_Flux:1>
```
Settings: Steps 30, CFG 7.5, Sampler: DDIM, Size: 512x512

### Elegant Couch Scene
```
The image features Lya. She is reclining on a plush couch, one arm resting on the armrest, wearing an elegant dress. The setting is softly lit, creating a cozy atmosphere. The camera captures her in a (medium shot). (realistic skin texture), (photography).
<lora:Lya_Flux:1>
```
Settings: Steps 30, CFG 7.5, Sampler: DDIM, Size: 512x512

## Keywords

### Trigger
- `Lya` (required)

### Appearance
- `long and slightly wavy dark brown hair`
- `hair falls naturally over her shoulders`
- `delicately shaped eyebrows`
- `elegant`
- `relaxed`

### Settings
- `studio setting`
- `cozy atmosphere`
- `industrial-style studio`
- `natural light`
- `softly lit`

### Shots
- `medium shot`
- `full shot`
- `positioned sideways`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 30 |
| **CFG** | 7.5 |
| **Sampler** | DDIM |
| **Size** | 512x512 |
| **Strength** | 1.0 |

## Recommended Combinations

### With Realism
```
<lora:Lya_Flux:1>
<lora:flux_realism_lora:0.6>
```

### With Detail Enhancement
```
<lora:Lya_Flux:1>
<lora:detail_enhancer_flux_v1:0.7>
```

### NSFW
```
<lora:Lya_Flux:1>
<lora:MysticXXX-v6:0.5>
```

## Notes

- Trigger word `Lya` is required
- Basic Model Version 1, Epoch 17
- Works best with studio/interior settings
- Higher CFG (7.5) recommended for this model
- DDIM sampler gives best results
- Add "(realistic skin texture), (photography)" for enhanced realism


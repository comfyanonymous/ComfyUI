# 8. Ethnicity - Latina

[← Back to Index](INDEX.md)

## Description

LoRAs for generating Latina/Hispanic characters with realistic features.

## Table of Contents

- [Generic Latina V2](#generic-latina-v2)
- [Latina (Milly Lua)](#latina-milly-lua)
- [Latinas](#latinas)

---

## Generic Latina V2

| Parameter | Value |
|-----------|-------|
| **File** | `Nat_Flux-000001.safetensors` |
| **Civitai** | https://civitai.com/models/748356/genericlatinav2 |
| **Trigger word** | None |
| **Strength** | 0.7-1.0 |
| **Type** | Character / Ethnicity |

### Description
LoRA for generating Latina female characters with realistic features.

### Keywords
- `latina`
- `hispanic`
- `brunette`
- `olive skin`

---

## Latina (Milly Lua)

> **Note:** We have two similar LoRAs for Latina characters - this one (Milly Lua) and [Latinas](#latinas). They can be combined or used interchangeably.

| Parameter | Value |
|-----------|-------|
| **File** | `Milly_Lua-000004.safetensors` |
| **Civitai** | https://civitai.com/models/859892/latina |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 |
| **Type** | Character / Ethnicity |
| **Compatibility** | FLUX |

### Description
LoRA for generating European-Latina looking characters. Ideal for realistic portraits and full body shots with characteristic Latina beauty features - dark hair, olive complexion, expressive facial features.

### Recommended settings
- **Steps:** 20
- **Scheduler:** res_2s, bong_tangent
- **LoRA strength:** 0.8

### Sample prompts

**Prompt 1 (Back view fashion):**
```
back view, full bodyshot photography of a brunette european latina girl (not asian) with big boobs, horsefaced laughing and seducing inviting, thick kayal, she wears a large black textile hairband, wavy long hair, he is wearing a pair of acrylic transparent clear h33l 10inch pleaser platform heels, big ass, wearing textile ash-green low-rise covering half her butt and slightly flared pants and she wears a white thight top with thin spaghetti straps and huge cleavage, pearl choker, round metallic earrings, shiny glossy lipstick, big breasts, huge cleavage, standing stiff in front of window and wearing her clear high platform shoes, in student bedroom , bright sun and shadows
```

### Recommended combinations
- `<lora:Milly_Lua-000004:0.8> <lora:MysticXXX-v6:0.7>` - NSFW latina
- `<lora:Milly_Lua-000004:0.8> <lora:seanarcher:0.6>` - Professional photography

### Keywords
- `european latina`
- `brunette`
- `wavy long hair`
- `olive skin`
- `big boobs`
- `curvy`

---

## Latinas

> **Note:** We have two similar LoRAs for Latina characters - this one and [Latina (Milly Lua)](#latina-milly-lua). They can be combined or used interchangeably.

| Parameter | Value |
|-----------|-------|
| **File** | `Latinas-000002.safetensors` |
| **Civitai** | https://civitai.com/models/864543/latinas |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 |
| **Type** | Character / Ethnicity |
| **Compatibility** | FLUX |

### Description
LoRA for generating Latina looking characters. Simple to use - just add Latina character info to your prompt. Ideal for influencer-style photos.

### Recommended settings
- **Steps:** 20
- **Scheduler:** res_2s, bong_tangent
- **LoRA strength:** 0.8

### Sample prompts

**Prompt 1 (Influencer):**
```
Latina influencer girl
```

### Recommended combinations
- `<lora:Latinas-000002:0.8> <lora:MysticXXX-v6:0.7>` - NSFW latina
- `<lora:Latinas-000002:0.5> <lora:Milly_Lua-000004:0.5>` - Combine both latina LoRAs
- `<lora:Latinas-000002:0.8> <lora:amateur:0.5>` - Amateur style

### Keywords
- `latina`
- `influencer`
- `brunette`
- `tan skin`
- `curvy`

---

[← Back to Index](INDEX.md)

# FLUX - More Female Anatomy

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,671 |
| **👍** | 124 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_More_Female_Anatomy.safetensors` |
| **Original filename** | `FLUX More Female Anatomy.safetensors` |
| **Civitai** | https://civitai.com/models/903238/flux-more-female-anatomy |
| **Trigger word** | None |
| **Strength** | 0.6-1.0 |
| **Type** | CONCEPT / Anatomy Enhancement |

## Description

Enhances female anatomy in FLUX generations. Improves detail and realism of nude female body parts including breasts, nipples, and overall anatomy. Works as a general enhancement LoRA for NSFW female content.

**Enhancement areas:**
- Breasts (various sizes)
- Nipples detail
- Overall female anatomy
- Body proportions

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 30 |
| **CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 2496x3648 (high-res) |
| **Strength** | 0.6-1.0 |

## Sample Prompts

**Nude photograph with big breasts:**
```
A high quality, detailed photograph of a nude woman who is naked. The nude woman has big tits, huge tits, and detailed nipples. The environment is indoors. The perfect lighting is dramatic. The highly detailed image is realistic, sharp focus, perfect composition, and RAW. The photo is candid with the best quality and intricate details. <lora:FLUX_More_Female_Anatomy:1.0>
```
Settings: Steps 30, CFG 3.5, Euler, 2496x3648

## Keywords

- `nude woman`
- `naked`
- `big tits`
- `huge tits`
- `detailed nipples`
- `realistic`
- `sharp focus`
- `RAW`
- `candid`

## Best Checkpoints

- FLUX Dev
- Any FLUX checkpoint

## Recommended Combinations

**With NSFW unlock:**
```
<lora:MysticXXX-v7:0.7>
<lora:FLUX_More_Female_Anatomy:0.8>
```

**With detail enhancer:**
```
<lora:Detail_Enhancer_Flux:0.7>
<lora:FLUX_More_Female_Anatomy:0.8>
```

**With Playboy style (as shown in example):**
```
<lora:FLUX_Playboy:1.0>
<lora:FLUX_More_Female_Anatomy:0.8>
```

## Notes

- No trigger word required
- Works well with dramatic lighting
- Best for high-resolution outputs
- Can combine with other NSFW and detail LoRAs
- Good for emphasizing breast size and detail
- Strength 0.6-1.0 depending on desired effect

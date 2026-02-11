# HighTails - High Side Pigtails

[← Back to Index](../INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 261 |
| **👍** | 41 |
| **Tips** | 500 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `HighTails_Flux.safetensors` |
| **Original filename** | `HighTails.safetensors` |
| **Civitai** | https://civitai.com/models/1924803 |
| **Trigger word** | `hightails` |
| **Strength** | 1.0-2.0 |
| **Type** | CONCEPT (Hairstyle) |

## Description

Simple concept LoRA for generating high side pigtails hairstyle. Works with various checkpoints including Pony, Illustrious, and FLUX models.

**Capabilities:**
- High side pigtails hairstyle
- Works with wavy/straight hair
- Compatible with multiple base models

## Sample Prompts

### Beach Scene (Simple)
```
A beautiful girl on the beach with her hair in HighTails.
<lora:HighTails_Flux:1>
```
Settings: Steps 24, CFG 3.5, Sampler: DPM++ 2M Karras, Size: 1024x1024

### Post-Apocalyptic Campfire
```
female, detailed eyes, dark eyebrow, cute face, freckles, ear piercing, petite body, short, small face, cute, slender, post-apocalyptic ruins at twilight, breathtakingly beautiful young woman sitting by a small campfire, form-fitting weathered clothing with layered fabrics, (((hightails))) wavy hair softly illuminated by firelight, gentle expression with a hint of melancholy, scattered survival gear around her, cracked pavement and overgrown weeds, distant silhouette of collapsed buildings, faint smoke rising in the background, warm orange glow from the fire contrasting with cool blue twilight, cinematic composition, high detail, soft depth of field, emotional and atmospheric
<lora:HighTails_Flux:1>
```
Settings: Steps 30, CFG 5, Sampler: DPM++ 2M Karras

### Peasant Woman in Barn
```
cinematic realism, a young peasant woman, lying on hay, golden hour lighting, soft smile, loose linen blouse (slightly unbuttoned), flushed cheeks, wavy auburn hair in hightails, freckles, warm sunlight filtering through barn, delicate hands resting on hay, rustic charm, romantic atmosphere, detailed fabric textures, soft shadows, 8k
<lora:HighTails_Flux:2>
```
Settings: Steps 30, CFG 5, Sampler: DPM++ 2M Karras

## Keywords

### Trigger
- `hightails` (required)
- `HighTails`
- `hair in hightails`
- `wavy hair in hightails`

### Hair Descriptions
- `high side pigtails`
- `wavy hair`
- `auburn hair`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 24-30 |
| **CFG** | 3.5-5 |
| **Sampler** | DPM++ 2M Karras |
| **Size** | 832x1216 / 1024x1024 |
| **Strength** | 1.0-2.0 |

## Recommended Combinations

### With Character LoRAs
```
<lora:HighTails_Flux:1>
<lora:[Character_LoRA]:0.8>
```

### With Realism
```
<lora:HighTails_Flux:1>
<lora:flux_realism_lora:0.6>
```

## Notes

- Trigger word `hightails` is required
- Simple concept - just add to prompt with hair description
- Works with Pony, Illustrious, CyberRealistic, and FLUX checkpoints
- Strength 1.0-2.0 works well depending on desired effect
- Can be combined with wavy/straight hair descriptions


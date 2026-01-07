# JJ's Modern Style House

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 516 |
| **👍** | 35 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `JJsModernHouse_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/292241 |
| **Trigger word** | `modern house` |
| **Strength** | 1.0 |
| **Type** | BUILDINGS (Modern Architecture) |

## Description

Part of JJ's architecture series - generates modern style houses with contemporary architectural design. Creates stunning modern residential buildings with clean lines, geometric shapes, and dramatic angles. Works well for both exterior shots and architectural photography compositions.

**Capabilities:**
- Modern house exteriors
- Street-level architecture views
- Clean geometric designs
- Architectural photography style
- Curvy modern designs
- Dramatic lighting and shadows

## Sample Prompts

### Simple Modern House Street View
```
<lora:JJsModernHouse_Flux:1> , ((modern house)), buildings, street, sky, photography, capturing moments, storytelling, creative composition
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 4.5

### Architecture Photography Style
```
<lora:JJsModernHouse_Flux:1> , ((modern house)), buildings, street, sky, curvy, architecture photography, striking structures, clean lines, geometric shapes, dramatic angles, play of light and shadow, capturing architectural details, showcasing design elements, evoking mood, professional lighting, precise compositions, emphasizing scale and proportion, creating depth, architectural storytelling, capturing iconic landmarks, immersive experience
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 18.6

## Keywords

### Main Triggers
- `modern house` (primary)
- `buildings`

### Architecture Style
- `curvy`
- `clean lines`
- `geometric shapes`
- `striking structures`
- `modern architecture`

### Photography Style
- `architecture photography`
- `dramatic angles`
- `play of light and shadow`
- `capturing architectural details`
- `showcasing design elements`
- `professional lighting`
- `precise compositions`

### Environment
- `street`
- `sky`
- `iconic landmarks`

### Composition
- `emphasizing scale and proportion`
- `creating depth`
- `architectural storytelling`
- `immersive experience`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20 |
| **CFG** | 1 |
| **Distilled CFG** | 4.5-18.6 |
| **Sampler** | Euler |
| **Size** | 896x1152 |
| **Strength** | 1.0 |

## Recommended Checkpoint

- **flux1-dev-bnb-nf4-v2** - tested and works

## Notes

- Part of JJ's architecture LoRA series (same creator as JJ's Interior Office)
- Use `((modern house))` with emphasis for best results
- Higher Distilled CFG (18.6) gives more dramatic architectural style
- Lower Distilled CFG (4.5) for simpler street views
- Works well for real estate photography style
- Combine with sky/weather prompts for atmosphere


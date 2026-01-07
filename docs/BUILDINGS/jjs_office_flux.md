# JJ's Interior Office

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 2,623 |
| **👍** | 187 |
| **Tips** | 0 |
| **Score** | ⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `JJs_Office_Flux.safetensors` |
| **Original filename** | `JJsOffice_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/1593773/jjs-interior-office |
| **Trigger word** | `Office` |
| **Strength** | 1.0 |
| **Type** | BUILDING / Interior / Office |

## Description

Interior office space LoRA for generating modern office environments. Creates detailed office interiors with glass walls, monitors, furniture, and professional lighting. Part of JJ's Interior Space series.

### Key Features
- Modern office interiors
- Glass walls and windows
- Monitors, keyboards, screens
- Furniture (chairs, tables, sofas)
- Dramatic lighting options
- Professional/corporate aesthetics

## Sample Prompts

**Prompt 1 (Full office with details):**
```
<lora:JJs_Office_Flux:1>, ((Office)), indoors, window, chair, table, scenery, ceiling, floor, glass wall, monitor, keyboard, book, flower, dramatic lighting, panel, sofa, screen, lamp, vent, stainless
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Dist.CFG 3.5
Hires: 1.5x, 4x-UltraSharp, Denoise 0.38

**Prompt 2 (Landscape orientation):**
```
<lora:JJs_Office_Flux:1>, ((Office)), indoors, window, chair, table, scenery, ceiling, floor, glass wall, monitor, keyboard, book, flower, dramatic lighting, panel, sofa, screen
```
Settings: Steps 20, CFG 1, Euler, 1152x896, Dist.CFG 3.5
Hires: 1.5x, 4x-UltraSharp, Denoise 0.38

**Prompt 3 (With lamp and vent):**
```
<lora:JJs_Office_Flux:1>, ((Office)), indoors, window, chair, table, scenery, ceiling, floor, glass wall, monitor, keyboard, book, flower, dramatic lighting, panel, sofa, screen, lamp, vent
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Dist.CFG 3.5

## Keywords

- `Office` - **TRIGGER** (emphasize with double parentheses)
- `indoors` - interior setting
- `window` - windows
- `glass wall` - modern glass partitions
- `chair`, `table`, `sofa` - furniture
- `monitor`, `keyboard`, `screen` - tech equipment
- `dramatic lighting` - lighting style
- `ceiling`, `floor` - room elements
- `panel`, `vent`, `lamp` - details
- `stainless` - metal accents
- `book`, `flower` - decorations

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 896x1152 (portrait) / 1152x896 (landscape) |
| **Strength** | 1.0 |
| **Hires** | 1.5x, 4x-UltraSharp, Denoise 0.38 |

## Usage Tips

- Use `((Office))` with double parentheses for stronger effect
- Add furniture keywords for specific elements
- `dramatic lighting` enhances atmosphere
- Works with both portrait and landscape orientations
- Combine with character LoRAs for office scenes
- `glass wall` creates modern corporate look
- Add tech keywords (`monitor`, `keyboard`, `screen`) for workstation areas

## Notes

- Part of JJ's Interior Space series
- Trigger: `Office` (emphasized with parentheses)
- Full strength (1.0) recommended
- Low CFG (1) with Distilled CFG 3.5
- Hires upscaling improves detail quality
- Works with flux1-dev-bnb-nf4-v2 checkpoint
- Great for professional/corporate scene backgrounds


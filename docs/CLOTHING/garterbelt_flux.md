# GarterBelt Flux Dev

[← Back to CLOTHING Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `GarterBeltFlux1.0.safetensors` |
| **Civitai** | https://civitai.com/models/992358/garterbelt-flux-dev-option-nude-girl |
| **Trigger word** | `gartbt` |
| **Strength** | 1.0 (1.2 for forced nudity) |
| **Type** | Clothing / Stockings / Garters |
| **Compatibility** | FLUX Dev only (not Schnell) |

## Description
Garter belt for stockings with garters w/o anything else. This LoRA can create nude girls (also with thigh high boots or holdup stockings w/o garters) without an NSFW Checkpoint. All sample images can be created with FLUX 1 Dev and only this LoRA.

**Other versions available:**
- GarterBelt Flux Schnell - for Schnell
- GarterBelt XL - for SDXL
- GarterBelt Pony - for Pony

## Prompt structure
Build your prompt like this:
```
'gartbt', <'front'|'side'> 'view', <'slim nude girl'> in red 'stockings' and 'garters' ...
```

## Sample prompts

**Prompt 1 (Front view street):**
```
gartbt, front view, slim nude girl in red stockings and garters standing in a crowded street,
```

**Prompt 2 (Side view mall):**
```
gartbt, side view, girl in red stockings and garters standing in a crowded mall,
```

**Prompt 3 (Thigh high boots):**
```
gartbt, front view, slim nude girl in black (thigh high boots) in a crowded park
```

**Prompt 4 (Vienna with Pleaser heels):**
```
ultra detailed cinematic film still enviroment: The scene takes place on the square in front of the stephansdom in vienna., action: Photo shows extremely pretty and sexy arab woman wearing fully fashioned nylon stockings, ffstockings, and black garters (suspenders), she is standing. detailed erect nipples extremely beautiful and sexy dressed woman has glossy skin and short black hair. in in a park wearing white gartbt front view stockings garters and high heels patent leather boots h33l highly detailed perfect composition, ultra realistic photograph <lora:ffstockings8_DEV-000100:0.9> <lora:nipplediffusion-f1:1> <lora:GarterBeltFlux1.0:1> <lora:Pleaser_Brand_Shoes-000001:1>
```

## Keywords
- `gartbt` - **REQUIRED** trigger word
- `front view` / `side view`
- `slim nude girl`
- `garters`
- `stockings`
- `garter belt`
- `thigh high boots`
- `fully fashioned nylon stockings`
- `holdup stockings`

## Notes
- Use strength 1.0 for standard shots
- Increase to 1.2 for forced/more prominent nudity
- Lower strength (0.5) works well with NSFW checkpoints
- Works with thigh high boots or holdup stockings without garters
- Compatible with watercolor and artistic styles
- Does NOT work with Flux Schnell - use Schnell version instead

## Quality Stats
- **Downloads:** 1,408
- **Rating:** 104
- **Tips:** $10

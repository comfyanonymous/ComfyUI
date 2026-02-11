# Luxury Sandals: Gemstone Decorated Sandal Heels

[← Back to CLOTHING Index](../INDEX.md)

## Info
- **File:** `Lux_Sandals_Gems_FL.safetensors`
- **Civitai:** https://civitai.com/models/1308109/luxury-sandals-gemstone-decorated-sandal-heels-attire
- **Downloads:** 199
- **Rating (👍):** 14
- **Tips:** 200
- **Trigger:** `Lux_Sandals_Gems` (preceded by a color)
- **Strength:** 0.8-1.0 (0.6-0.8 when combined with other LoRAs)
- **Type:** CLOTHING / FOOTWEAR

## Description
Luxury Sandal collection featuring gemstone-embellished sandal heels. This LoRA creates opentoe stiletto heels with gemstone embellished straps. Part of a planned collection of luxury decorated sandal heels.

Works well with other Flux LoRAs for legwears (Glossy Pantyhose, BlackRealStockings, TanRealStockings).

## Color Options
The trigger word must be preceded by a color:
- `beige Lux_Sandals_Gems`
- `white Lux_Sandals_Gems`
- `black Lux_Sandals_Gems`
- `green Lux_Sandals_Gems`
- `gold Lux_Sandals_Gems`
- `silver Lux_Sandals_Gems`
- `turquoise Lux_Sandals_Gems`
- `purple Lux_Sandals_Gems`

## Toenail Options
- `natural toenails`
- `painted toenails` (can specify color: e.g., "pink painted toenails")

## Sample Prompts

**Close-up of feet:**
```
female feet, green Lux_Sandals_Gems, natural toenails <lora:Lux_Sandals_Gems_FL:1>
```

**Full body portrait:**
```
front view, full body portrait, blonde woman, standing, white Lux_Sandals_Gems, red painted toenails, mini dress, indoor, living room <lora:Lux_Sandals_Gems_FL:1>
```

**Three quarter view with detail:**
```
<lora:Lux_Sandals_Gems_FL:1>, photography, three quarter view, woman feet, turquoise Lux_Sandals_Gems, red painted nails, white pants, indoor, wooden floor
```
Settings: Steps: 40, CFG: 1, Sampler: Euler, Distilled CFG: 3.5

**Combined with bodysuit:**
```
<lora:Lux_Sandals_Gems_FL:0.7>, <lora:RealSheerBodysuit_FL:0.7>, photography, (from below:1.1), redhair woman, long wavy hair, sitting on the bedside, black RealSheerBodysuit, crossed legs, (orange Lux_Sandals_Gems:1.2), bedroom, minimal furniture, modern furniture, (feet in foreground:1.2)
```
Settings: Steps: 40, CFG: 1, Sampler: Euler, Denoising: 0.12, Distilled CFG: 3.5

**Outdoor scene:**
```
<lora:Lux_Sandals_Gems_FL:1>, photography, female feet, (purple Lux_Sandals_Gems:1.1), purple toenails, outdoor, grassy floor, sun lighting
```
Settings: Steps: 40, CFG: 1, Sampler: Euler, Distilled CFG: 3.5

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 40 |
| CFG Scale | 1 |
| Distilled CFG | 3.5 |
| Sampler | Euler |
| Strength | 0.8-1.0 (solo), 0.6-0.8 (combined) |

## Keywords
- `Lux_Sandals_Gems`
- `female feet`
- `sandal heels`
- `gemstone embellished`
- `stiletto heels`
- `opentoe`
- Color prefixes: beige, white, black, green, gold, silver, turquoise, purple
- Toenail options: natural toenails, painted toenails

## Notes
- Works well for both close-up feet shots and full portrait images
- **WARNING:** For full portrait figures where sandals appear small, inpainting the heel area is strongly recommended to enhance details
- Optimized to work with legwear LoRAs (Glossy Pantyhose, BlackRealStockings, TanRealStockings)
- Color specification before trigger word is required for best results
- Part of planned Luxury Sandal collection - more variants coming

## Compatibility
- **Model:** FLUX.1 Dev
- **Works with:** Legwear LoRAs, bodysuit LoRAs
- **Best for:** Footwear focus, fashion photography, luxury accessories

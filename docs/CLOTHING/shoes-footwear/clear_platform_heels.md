# Clear Platform Heels | Pleaser Heels | Stripper Heels

[← Back to Index](../INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 719 |
| **👍** | 82 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Clear_Platform_Heels_V2.safetensors` |
| **Civitai** | https://civitai.com/models/1421994/clear-platform-heels-or-pleaser-heels-or-stripper-heels-flux |
| **Trigger word** | `clear platform heels` |
| **Trigger (v1.0)** | `transparent platform heels` |
| **Strength** | 1.0-1.5 |
| **Type** | CLOTHING / Footwear |

## Description

Clear platform high heels LoRA (Pleaser/stripper style). Generates clear/transparent platform heels with clear straps that make legs look long and sleek. Perfect for glamour, fashion, and sexy content.

**Training notes:**
- Model sometimes favors legs and lower body only
- Training data mostly from outdoor vibes with leg-baring style
- For Fluxmania, use weight around 1.5

**Potential biases:**
- May generate only lower body sometimes
- Legs get priority in composition

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-30 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Scheduler** | Simple / DDIM |
| **Strength** | 1.0 (or 1.5 for Fluxmania) |
| **Size** | 832x1216 |

## Sample Prompts

**Heels only (product shot):**
```
A pair of clear platform heels on a dark metallic floor. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 20, CFG 1, Dist.CFG 3.5, Euler + Simple

**Luxurious indoor:**
```
This is an image of a woman wearing a white ruffled skirt and off-shoulder top with clear platform heels, standing in an ornate room featuring marble textures and gold-accented furnishings. The reflection in a large mirror and delicate decor add a luxurious atmosphere to the scene. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

**Bikini on couch:**
```
The image is of a woman seated on a white couch, wearing a bright pink bikini top, white fishnet stockings, and clear platform heels. A multicolored ice-cream-shaped pillow is placed beside her, with a fluffy white carpet covering the floor and a decorative wall in the background. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

**City street walking:**
```
This photo is of a Caucasian woman walking, wearing a pair of clear platform heels. The photo is taken from side. Her long blonde hair flows behind her as she moves. The setting is a city street, with tall buildings and traffic in the background. She is carrying a black tote bag over her shoulder and appears to be in a hurry. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

**Red dress indoor:**
```
This photo is of a woman standing near a large window, wearing transparent platform heels with tower platforms. She is dressed in a red, form-fitting dress, and she has long blonde hair. The setting appears to be indoors with natural light streaming through the window, casting soft shadows on a hardwood floor. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

**Black mini dress modern interior:**
```
This photo is of a woman leaning over a railing in a modern, curved indoor space. She is wearing a black mini dress and clear platform heels. The setting appears to be a public or commercial area with reflective surfaces and a sleek design. The lighting is bright, suggesting daytime or well-lit interior. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

**Outdoor floral sundress:**
```
This photo is of a Caucasian blonde woman standing, wearing a floral sundress and a pair of clear platform heels with tower platforms. Her legs are crossed at the ankles, highlighting the heels. The setting is outdoors, with a green lawn and trees in the background. She is holding a straw hat in one hand and smiling. <lora:Clear_Platform_Heels_V2:1>
```
Settings: Steps 30, CFG 1, Dist.CFG 3.5, Euler + Simple

## Keywords

- `clear platform heels`
- `transparent platform heels`
- `tower platforms`
- `platform heels`
- `stripper heels`
- `pleaser heels`
- `high heels`

## Style Suggestions

**Outfit pairings:**
- Sundresses, mini skirts, cute shorts
- Bikinis, lingerie
- Form-fitting dresses
- Fishnet stockings

**Settings:**
- Beach or garden (natural light)
- Luxurious interiors (marble, gold accents)
- City streets
- Modern architecture

**Accessories:**
- Sun hat
- Delicate jewelry
- Tote bags

## Best Checkpoints

- flux_dev (fp8, fp16)

## Notes

- V2 trigger: `clear platform heels`
- V1 trigger: `transparent platform heels`
- May generate only lower body - use full body descriptions
- For Fluxmania model, increase weight to 1.5
- Tower platforms variant: add `with tower platforms`
- Works well with leg-baring outfits
- Best with outdoor/bright lighting settings

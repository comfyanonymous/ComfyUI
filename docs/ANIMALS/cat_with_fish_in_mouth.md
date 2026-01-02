# Cat With Fish In Mouth (FLUX)

[← Back to ANIMALS Index](INDEX.md)

## Info
- **File:** `Cat_With_Fish_In_Mouth_Flux.safetensors`
- **Original filename:** `Cat_With_Fish_In_Mouth_Flux.safetensors`
- **Civitai:** https://civitai.com/models/1912987/cat-with-fish-in-mouth
- **Trigger:** `A CAT holding fish in its mouth`
- **Strength:** 1.0
- **Type:** ANIMAL

## Description
LoRA to correctly position a fish in a cat's mouth. The base FLUX model understands the concept but sometimes positions the fish incorrectly. This LoRA reinforces proper fish placement. Trained on small dataset of 18 images.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 8 |
| CFG | 1 |
| Sampler | Euler |
| Size | 1024x768 |

## Prompt Format
Start prompt with trigger:
```
A CAT holding fish in its mouth, [scene description]
```

## Example Prompts

### Basic Cat
```
A CAT holding fish in its mouth,
```
Settings: Steps 8, CFG 1, Euler

### Blue Cat in City
```
A CAT holding fish in its mouth, blue cat, city background, street side,
```
Settings: Steps 8, CFG 1, Euler

### Mountain Selfie
```
A CAT holding fish in its mouth, cat selfie, mountain peak view,
```
Settings: Steps 8, CFG 1, Euler, 1024x768

### Cyberpunk Anime Style
```
A CAT holding fish in its mouth, neon light, nighttime, city, street side, modern cyberpunk style, illustration style, anime,
```
Settings: Steps 8, CFG 1, Euler, 1024x768

### Cyberpunk Realistic
```
A CAT holding fish in its mouth, neon light, nighttime, city, street side
```
Settings: Steps 8, CFG 1, Euler, 1024x768

### Yoda Cat at Lakeside
```
A CAT holding fish in its mouth, lakeside, YODA, YODA_CAT
```
Settings: Steps 8, CFG 1, Euler, 1024x768

## Keywords
- `A CAT holding fish in its mouth` (trigger)
- `cat`
- `fish`
- `street side`
- `city background`
- `neon light`
- `cyberpunk`

## Best Checkpoints
- FLUX Dev
- Any FLUX-based checkpoint

## Notes
- Very fast generation - only 8 steps needed
- Low CFG (1) works best
- Reinforces correct fish positioning in cat's mouth
- Works with various styles (realistic, anime, cyberpunk)
- Small training dataset (18 images) but effective
- Can combine with other style descriptors

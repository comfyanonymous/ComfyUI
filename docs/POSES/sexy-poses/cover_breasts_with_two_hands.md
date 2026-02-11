# CoverBreastsWithTwoHands Flux

[← Back to POSES Index](../INDEX.md)

## Info
- **File:** `Cover_Breasts_With_Two_Hands_FLUX.safetensors`
- **Original filename:** `CoverBreastsWithTwoHandsFlux.1.0.safetensors`
- **Civitai:** https://civitai.com/models/986168/coverbreastswithtwohands-flux
- **Trigger:** `cbwth` + `covering breasts with two hands` or `covering breasts with two crossed hands`
- **Strength:** 1.0
- **Type:** POSES

## Description
LoRA for posing with hands covering breasts. Two variants: regular hand placement or crossed hands. Trained on NSFW images - produces nude results without needing NSFW checkpoint. Can show nude vagina as well.

Also available for SDXL and Pony.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 20 |
| CFG | 1 |
| Distilled CFG | 3.5 |
| Sampler | Euler |
| Size | 768x1280 |

## Prompt Format
```
cbwth, (naked/nude [description] girl) covering breasts with two hands, [scene]
```

Or for crossed hands:
```
cbwth, (naked/nude [description] girl) covering breasts with two crossed hands, [scene]
```

## Example Prompts

### Tower Bridge - Crossed Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in front of tower bridge. black (holdup stockings), necklace, short brunette ponytail
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Acropolis - Regular Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in front of acropolis. black (thigh high boots), necklace, short blonde hair
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Crowded Pub
```
cbwth, (naked slim 25yo girl) covering breasts with two hands is standing in a crowded pub. black (holdup stockings), necklace, short brunette hair
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Coliseum
```
cbwth, (naked slim 25yo girl) covering breasts with two hands is standing in front of coliseum. black (holdup stockings), necklace, short blonde ponytail
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Capitol - Crossed Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in front of capitol. black (thigh high boots), necklace, long brunette hair
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Statue of Liberty
```
cbwth, (naked slim 25yo girl) covering breasts with two hands is standing in front of statue of liberty. black (thigh high boots), necklace, long brunette hair
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Eiffel Tower
```
cbwth, (naked slim 25yo girl) covering breasts with two hands is standing in front of Eiffel tower. black (thigh high boots), necklace, long blonde hair
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Brandenburg Gate - Crossed Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in front of brandenburg gate. black (thigh high boots), necklace, short blonde ponytail
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Neuschwanstein - Crossed Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in front of neuschwanstein. black (holdup stockings), necklace, long brunette ponytail
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

### Fast Food Restaurant - Crossed Hands
```
cbwth, (naked slim 25yo girl) covering breasts with two crossed hands is standing in a crowded fast food restaurant. black (holdup stockings), necklace, long brunette ponytail
```
Settings: Steps 20, CFG 1, Euler, Distilled CFG 3.5, 768x1280

## Keywords
- `cbwth` (trigger)
- `covering breasts with two hands`
- `covering breasts with two crossed hands`
- `naked` / `nude`
- `slim`

## Landmark Locations That Work Well
- Tower Bridge
- Acropolis
- Coliseum
- Capitol
- Statue of Liberty
- Eiffel Tower
- Louvre
- Brandenburg Gate
- Neuschwanstein

## Accessory Suggestions
- `black (holdup stockings)`
- `black (thigh high boots)`
- `necklace`

## Best Checkpoints
- flux1-dev-fp8
- FLUX Dev

## Notes
- Use strength 1.0
- No NSFW checkpoint needed - produces nude results
- Two hand positions: regular or crossed
- Works great with landmark/tourist locations
- ENF (Embarrassed Nude Female) scenarios
- Use parentheses for emphasis: `(naked slim 25yo girl)`
- Also available for SDXL and Pony

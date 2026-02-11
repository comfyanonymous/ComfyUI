# Ballet NSFW

[← Back to POSES Index](../INDEX.md)

## Info
- **File:** `Ballet_NSFW.safetensors`
- **Civitai:** https://civitai.com/models/688948/balletnsfw
- **Trigger:** `ballet`, `ballerina`
- **Strength:** 0.6-1.0
- **Type:** POSE / Style

## Description
LoRA for generating nude ballerinas. Better at rendering women without "balloon boobs" - creates slim, athletic dancer bodies with natural proportions. Great for artistic nude ballet scenes.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 20-25 |
| CFG | 1-7 |
| Sampler | Euler |
| Scheduler | Simple |
| Distilled CFG | 3.5-7 |
| Size | 896x1152 / 800x1200 / 832x1216 |

## Example Prompts

### Standing Ballerina
```
photograph of a ballerina. ribs, uncensored, hairy bush, earrings, standing, 1girl, small breasts, smile, ballet
```
Settings: Steps 20, CFG 7, Euler, Simple, 896x1152, Distilled CFG 7

### Mature Ballerina Spread
```
uncensored photograph of a skinny mature ballerina. ribs, hairy bush, earrings, spread legs, small breasts, smile, ballet
```
Settings: Steps 20, CFG 7, Euler, Simple, 800x1200, Distilled CFG 7

### Slim Ballerina with Pubic Hair
```
uncensored photograph of a slim mature ballerina. ribs, hairy bush, pubic hair, spread legs, small breasts, smile, ballet
```
Settings: Steps 20, CFG 7, Euler, Simple, 1200x800, Distilled CFG 7

### Stage Portrait
```
A portrait of a naked beautiful slim young ballerina on stage, in luxurious ballet scene settings. There are dark red curtains on the background. She is looking at the camera.
```
Settings: Steps 25, CFG 3.5, 832x1216

### Beach Scene (Creative Use)
```
Design a highly detailed, provocative image of Miss Monique, a stunning woman with jet-black hair streaked with green, styled in loose, wind-blown waves. She is at the beach, wearing a completely torn, ultra-revealing bikini. ballerina, ballet, slim, multicolored hair, realistic, navel, two-tone hair, green hair, black hair
```
Settings: Steps 20, CFG 1, Euler, Simple, 896x1152, Distilled CFG 3.5, Strength 0.6

### Hotel Lobby Artistic Nude
```
A tall and curvaceous woman with an elegant, slender frame and flawless, pale skin, Her small breasts are slightly pointed with light-colored, delicate areolas. Her abdomen is flat with soft, smooth curves at the hips. Her pubic area is completely shaved. She is standing with her legs widely apart. The setting is a bustling, high-end hotel lobby filled with elegantly dressed people. She is adorned with black lace stockings.
```
Settings: Steps 20, CFG 1, Euler, Simple, 1280x832, Distilled CFG 1

### Fashion Artistic Nude
```
breathtaking professional photography full body shot of, modern, contemporary, and daringly sexy artistic nude photo of a breathtakingly beautiful woman in her twenties. She should be posed seated in a bold and provocative manner. Her body must be perfectly sculpted with small, firm, and exquisitely shaped breasts. She is wearing black lace thigh-high stockings with intricate details, and Nike sneakers. Her outfit also includes a large, open shirt that reveals everything underneath.
```
Settings: Steps 22, CFG 1, Euler, Simple, 832x1280, Distilled CFG 3.5, Strength 0.8

## Keywords
- `ballet`
- `ballerina`
- `ribs`
- `small breasts`
- `slim`
- `skinny`
- `spread legs`
- `hairy bush` / `pubic hair`

## Recommended LoRA Combinations
- **Inverted_nipples_for_FLUX** (1.0) - inverted nipples
- **Pro-skin** (1.0) - realistic skin
- **Small_Nipples_-_FLUX** (0.9) - small nipples
- **two_tone_hair** (1.0) - multicolored hair
- **thigh_high_stockings** (1.0) - stockings
- **downblouse_v2** (1.0) - downblouse effect
- **35mm Photo** (1.0) - photography style

## Best Checkpoints
- fluxunchainedArtfulNSFW_fuT516xfp8E4m3fnV11
- creartHyperFluxDevBnbNf4_hyperDevFp8Unet
- flux1_devFP8Kijai11GB

## ADetailer Settings (Optional)
For enhanced details:
- Face: mediapipe_face_full (confidence 0.3)
- Pussy: pussyV2.pt (prompt: hairy)
- Nipples: nipple.pt (negative: pierced)
- Denoising: 0.4

## Notes
- Better for slim/athletic body types (avoids balloon boobs)
- Works well with hairy bush aesthetic
- Good for artistic nude and stage settings
- Strength 0.6-0.8 for subtle effect, 1.0 for strong ballet aesthetic
- Combine with stockings and lingerie for fashion looks

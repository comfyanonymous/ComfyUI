# FluxBentOverInCar

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `FluxBentOverInCar.safetensors` |
| **Original filename** | `InCar.safetensors` |
| **Civitai** | https://civitai.com/models/789366/fluxbentoverincar |
| **Trigger word** | None (use descriptive prompt) |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Pose / NSFW |

## Description

LoRA for bent over poses in car interior. Creates images of women bent over on car seats, showing ass and pussy from behind. Works with various settings (night/day) and outfits.

## Sample prompts

**Prompt 1 (Cyberpunk French Maid - Night):**
```
dyngru, xslx scribble, cineboke, 7-RetroFuturism, mad-dtstrm. A stunning young Yennefer posing in a sleek, half turned to viewer. She wears Jed-frmd, French Maid, A short, frilly black dress with white apron, black heels, her dress showing a lot of cleavage and lace shiny stockings. Her dark, sultry makeup with bold eyeshadow and deep lipstick enhances her mysterious allure. Cultural and artistic beauty to her appearance. She is offering a teasing smile. from behind, bent over, 1girl, car interior, pussy, anus, all fours on car seats, apple ass shape, clothes pull, ass focus, thighs, low angle, close up on ass, grain, noise, low light, jpeg artifacts, night, anus, top down bottom up. Her expression is a playful smirk. The camera captures her from an cowboy shot. The lighting casts neon lights with a subtle **Bokeh effect** blurring the background. The hyperrealistic car is infused with neon lights, retro vector art, glitch effects, and wireframe elements, enhancing the sleek, cyberpunk aesthetic with **sharp focus** and **cinematic lighting**. realistic anatomy. <lora:FluxBentOverInCar:1>
```

**Prompt 2 (Simple sunny day):**
```
from behind, bent over, 1girl, car interior, pussy, anus, all fours on car seats, apple ass shape, clothes pull, ass focus, thighs, low angle, sunny <lora:FluxBentOverInCar:1>
```

## Keywords

- `from behind`
- `bent over`
- `car interior`
- `all fours on car seats`
- `apple ass shape`
- `clothes pull`
- `ass focus`
- `thighs`
- `low angle`
- `close up on ass`
- `top down bottom up`

## Tested combinations

**Combination 1 (Character + detailed style):**
```
Checkpoint: FLUX Dev
<lora:FluxBentOverInCar:1> <lora:Yennefer:1> <lora:extremely_detailed:1> <lora:Frenchmaid:1>
```

**Combination 2 (Pony with ass slider):**
```
Checkpoint: CyberRealistic Pony v11.0
<lora:FluxBentOverInCar:1> <lora:Ass_Size_Slider:1>
```

## Notes

- Strength 1.0 recommended
- Works with FLUX Dev and Pony checkpoints
- CFG 1-3.5 works well
- Steps 20-25 for quality
- Portrait orientation (832x1216) typical
- Combines well with character LoRAs
- Night/low light settings add atmosphere
- Cyberpunk aesthetic works well with neon lights

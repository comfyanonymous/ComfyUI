# WATW - Japan

[← Back to CHARACTERS_WATW Index](INDEX.md)

## Info
- **File:** `Japan.safetensors`
- **Civitai:** https://civitai.com/models/2167490/watw-japan
- **Trigger:** `Japan` or `Japan woman`
- **Strength:** 1.0
- **Series:** Women Around The World (WATW) - 207 countries

## Description
Part of the "Women Around The World" (#WATW) series. All WATW LoRAs are a mixture of 2-3 bodies combined with an AI face. The goal is to create unique basic types that can be customized (hair, facial features, etc.).

## Best Checkpoints
- unStable Evolution KREA
- unStable Evolution FluXXX
- flux1-dev-fp8
- Flux.1_Krea_Dev FP8 SCALED
- Standard BFL versions

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| CFG | 1-2 |
| Sampler | Flux Realistic / Euler / dpmpp_2m_sgm_uniform |
| Schedule | Beta (alpha: 0.6, beta: 0.6) |
| Size | 896x1152 / 832x1216 |
| Steps | 25 |
| Distilled CFG | 2.5 |

## Example Prompts

### Portrait
```
Breathtaking over the shoulder shot photography of a Japan woman looking at viewer, necklace, looking over shoulders, eyelashes, fine hair detail, entire hairstyle visible, perfect eyes with iris pattern, sensual lips, nose, (perfectly sharp:1.3), realistic textures, (deep focus, focus on background:1.5), 8k uhd, dslr, ultra high quality image, film grain, Fujifilm XT3
```
Settings: Steps 25, CFG 1, Flux Realistic, 896x1152

### Bikini
```
photography of a Japan woman, dressed in floral bikini, vibrant colors, textured fabric, seated, relaxed posture, looking forward, rule of thirds, projection lighting, (straight on:1.2), shot on Olympus OM-D E-M1 Mark III with M.Zuiko 12-40mm f-2.8
```
Settings: Steps 25, CFG 1, Flux Realistic, 896x1152

### Fashion
```
A high-resolution Photo of a Japan woman. She wears a sleeveless, loose-fitting mini dress with a leopard print. The pattern is in natural tones (brown, beige, black) and is very striking. She pairs it with black, tight-fitting leggings. She wears gray, open-toe high-heeled ankle boots. A wide, light-colored hairband holds back her hair with a center parting. Her makeup is rather subtle, with a focus on the lips and eyes.
```
Settings: Steps 25, CFG 1, Flux Realistic, 896x1152

### Intimate/Bedroom
```
Japan woman lies on a bed with floral-patterned bedding in a dimly lit, vintage room with a beige wall and tufted headboard. She wears a short, cropped, light pink ribbed sweater, her legs spread wide. Her expression is neutral to slightly sultry, gazing directly at the camera. Soft, slightly yellowish lighting casts a warm, nostalgic glow, with a subtle film grain and warm color grading.
```
Settings: Steps 25, CFG 1, Flux Realistic, 896x1152

### Lifestyle / Environmental
```
Masterpiece high-resolution photo of mid-twenties Japan woman, with captivating brown eyes, realistic skin texture, slim and fit with natural curves, luxurious dark brown hair, arranged in gentle ringlets reminiscent of gothic romance portraits. standing with weight on right leg, left leg slightly bent. Professional lifestyle photography, sharp midtones, subtle background bokeh. Set in an old gas station with rounded corner architecture and peeling pastel paint. She is wearing a sleeveless knit tank with white denim shorts, a heart pendant, and aviators, realistic fabric and clothing textures. Environmental shot, subject framed in full with surrounding context. Gentle afternoon sunlight with low contrast.
```
Settings: Steps 25, CFG 2, dpmpp_2m_sgm_uniform, 832x1216, flux1-dev-fp8 + XLabs Flux Realism LoRA

### Vanlife / Cozy
```
A realistic photo of a striking 25 year old beautiful Beautiful Japanese woman, "Vanlife Velvet" Inside a VW camper van strung with fairy lights and paisley curtains, a woman lounges in high-cut underwear and a sheer tank top, sipping tea from a mismatched mug; dusky light through tinted windows creates amber-toned intimacy. (high resolution image)
```
Settings: Steps 25, CFG 1, Euler, 512x512, Flux.1_Krea_Dev FP8 SCALED

### Noir / Dramatic
```
A realistic photo of a striking 25 year old beautiful Beautiful Japanese woman, wearing (gray velvet trousers:1.25), (black lace shirt:1.25), candle-lit realism, A grimy backroom poker game with smoke-filled air, scattered playing cards, and a single bare bulb overhead. One hand raised in invocation (high resolution image)
```
Settings: Steps 25, CFG 1, Euler, 512x512, Flux.1_Krea_Dev FP8 SCALED

## Recommended LoRA Combinations
- XLabs Flux Realism LoRA (for enhanced realism)

## Notes
- Use trigger word at strength 1.0
- Works well with various poses and outfits
- Good for Japanese aesthetic and styling
- Series covers 207 countries with daily releases

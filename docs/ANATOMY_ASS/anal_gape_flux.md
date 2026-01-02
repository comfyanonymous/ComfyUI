# Anal Gape - Flux

[← Back to INDEX](INDEX.md)

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Anal_Gape_-_Flux.safetensors` |
| **Civitai** | https://civitai.com/models/722641/anal-gape-flux |
| **Trigger word** | None (use descriptive prompts) |
| **Strength** | 1.0-1.4 |
| **Type** | CONCEPT / Anatomy / NSFW |
| **Compatibility** | FLUX |

## Description

LoRA for generating anal gape/gaping anus effects. Sensitive to prompts and can easily overwrite the model at higher strengths. Works with various poses (from behind, laying back, bent over).

**WARNING:** Very explicit NSFW content. Use descriptive prompts for best results.

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Strength** | 1.0-1.4 |
| **Distilled CFG** | 3.5 |
| **Steps** | 20-28 |
| **Sampler** | Euler |
| **Schedule** | Simple |
| **Clip skip** | 1-2 |

## Trigger Prompts

**From behind pose:**
```
ass, anus, anal gape, gape, pussy, spread ass, spread anus, presenting, thighhighs, from behind, leaning forward
```

**Laying back pose:**
```
ass, anus, anal gape, gape, pussy, spread ass, spread anus, presenting, thighhighs, laying back
```

## Sample Prompts

**Prompt 1 (Basic blonde):**
```
A long-haired blonde girl, nude, girl showing her ass, ass, anus, anal gape, gape, pussy, spread ass, spread anus, presenting, thighhighs, from behind, leaning forward, <lora:Anal_Gape_-_Flux:1.4>
```
Settings: Steps: 20, CFG: 1, Sampler: Euler, Size: 896x1152, Schedule: Simple, Distilled CFG: 3.5

**Prompt 2 (Realistic skin bent over):**
```
aidmarealisticskin, Uncensored, real life, 1girl, photorealistic, solo, uncensored, nude, completely nude, back, from behind, facing away, bent over, huge hips, presenting, blonde hair, huge breasts, backboob, shiny, shiny skin, narrow waist, thighs, huge ass, ass focus, hands on own ass, fingernails, nail polish grabbing own ass, ass grab, spread ass, gaping, anus, spread anus, pussy, indoors, <lora:Anal_Gape_-_Flux:1>
```
Settings: Steps: 60, CFG: 20, Size: 1216x832, Clip skip: 2

**Prompt 3 (Complex multi-LoRA stack):**
```
european mature woman, high forehead, black-brown hair, red lips, big breast, scandalously sexy gauzy off-shoulder dress, short shoulder length hair, transparent, sultry, alluring, facing viewer, directional lighting, with an abstract dark backdrop, bare back and naked ass, long sparkling earrings, 1woman, pov from behind, prone position, anus, pussy, pink pussy, ass spread, small pussy, anus demonstrated, painted fingernails, extreme close up, HD, very detailed, high quality, ass, anus, anal gape, gape, pussy, spread ass, spread anus, presenting, thighhighs, from behind, leaning forward, asstastic, her buttocks and anus are the focus of the image <lora:Anal_Gape_-_Flux:0.6>
```
Settings: Steps: 39, CFG: 1, Sampler: Euler, Size: 832x1216, Schedule: Simple, Distilled CFG: 3.5

**Prompt 4 (Pizza delivery):**
```
This is a high-quality, suggestive photograph of a curvaceous woman with long, straight, dark brown hair, wearing a tight very skimpy, red and white "PIZZA DELIVERY" uniform, bending over the hood of a car. She is wearing a red ball cap with a pizza graphic on it. She has a dark tan complexion and is wearing white high heels. The background is outdoors in front of an old sedan car with a pizza delivery topper on it at dusk, with artificial lighting highlighting her figure. (skin texture:1.5), (skin pores:1.5), (eyelashes:1.9), (dimples:1.3), (pretty teeth:1.4), (extreme detail:1.4), (face detail:1.6) (detailed eyes:1.8). The woman is looking over her shoulder directly at the viewer with seductive eyes. The camera angle is from below, with the woman's ass in the center of the frame. She has a gaping asshole, (ass detail:1.8), (anal hair:1.7), (anal gape gaping:1.8), (asshole detail:1.9). (extreme gaping:1.9), (sweaty:1.7), ass, anus, anal gape, gape, pussy, spread ass, spread anus, presenting, from behind, leaning forward, <lora:Anal_Gape_-_Flux:1.4>
```
Settings: Steps: 30, CFG: 1, Sampler: DPM++ 2M, Size: 896x1152, Schedule: Beta (0.6/0.6), Distilled CFG: 3.5, Hires upscale: 1.5 (4xUltrasharp), Denoising: 0.4

## Keywords

- `anal gape`
- `gape`
- `gaping`
- `spread ass`
- `spread anus`
- `presenting`
- `from behind`
- `leaning forward`
- `laying back`
- `prone position`
- `ass focus`
- `extreme gaping`

## Tested Combinations

**Combination 1 (Realistic multi-stack):**
```
<lora:Anal_Gape_-_Flux:0.6> <lora:asstastic_flux_lora_v2:0.7> <lora:Flux_Pussy_Anus_HD:0.5> <lora:Anus_Vulva_Helper:0.9> <lora:Flux_Prone_Ass_Spread_HD:0.7>
```
Very realistic results with this combination.

**Combination 2 (Realism boost):**
```
<lora:Anal_Gape_-_Flux:1.4> <lora:flux_realism_lora:0.8>
```

**Combination 3 (Detailed face + gape):**
```
Checkpoint: Fluxed Up NSFW 5.1_FP16
<lora:Anal_Gape_-_Flux:1.4> <lora:closeupface-v1:0.8>
```

## Notes

- Strength 1.0-1.4 (higher = more extreme gape)
- Sensitive to prompts - can overwrite model easily
- Use 0.6 when stacking with many other LoRAs
- CFG 1 with Distilled CFG 3.5 recommended
- Steps 20-28 for quality (up to 60 for extreme detail)
- Works well with ass/anatomy LoRAs (Asstastic, Prone Ass Spread, etc.)
- Use 4x Ultrasharp upscaler for hires
- Beta scheduler (0.6/0.6) or Simple works well

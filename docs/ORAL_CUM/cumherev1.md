# CumHereV1

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 5944 |
| **👍** | 379 |
| **Tips** | 1030 |

| Parameter | Value |
|-----------|-------|
| **File** | `CumHereV1.safetensors` |
| **Civitai** | https://civitai.com/models/730199/cumherev1-flux-cum-lora-clothes-hair-non-facials |
| **Trigger word** | None (understands "cum" naturally) |
| **Strength** | 0.8-1.0 |
| **Type** | Effect / NSFW |
| **Training** | 7h on RTX 4090, 4000 steps, 34 images |

### Description
Non-facial cumshot LoRA focused on realistic cum on clothes (sweaters, panties, swimsuits), breasts, tummy, hair, legs, feet, creampie, and more. Complements facial cum LoRAs. No trigger word needed - just describe what you want specifically.

### Target areas
- Clothes (sweaters, panties, swimsuits)
- Breasts and nipples
- Tummy/belly
- Hair
- Legs and feet
- Ass and panties
- Body (full coverage)
- Creampie/cum in pussy

### Usage tips
- Weight 1.0 works well, reduce to 0.8 if too much cum
- No need for "white cum" - already understands cum is white
- Be very specific: "There is cum on her ass and panties"
- Use "a lot of cum" for more quantity
- Not trained on facials but may randomly add them

### Tested resolutions
- 832 x 1216
- 896 x 1152
- 1024 x 1024

### Sample prompts

**Prompt 1 (Creampie library):**
```
Young nude skinny 25 year old woman smiling while on a mat spreading legs, creampie, cum in her pussy, she is wearing a silver necklace, sharp focus on her pussy, cum is dripping down her ass onto the floor. sunlight is coming in trough the window in a library. she has a slim naked body. She is holding a sign that says 'Cum inside me!' with a heart on it.
```

**Prompt 2 (Cowgirl cum covered):**
```
full body shot of a slim and slender young cowgirl with large breasts. she is a blonde with freckles. sitting, spreading her legs, showing her pussy, creampie, cum drip, cum in pussy. her face and body are covered in loads of cum, cum shot. she is wearing a typical classic cowboy outfit and hat, white boots with spurs. her cowboy clothes have been ripped and torn apart, revealing her body, panties pulled aside. she is leaning back with her back on the wall, head down, looking away. she is sad, embarrassed, lost, open mouth. in an empty pub in the wild west, after hours, dimmed light, shadows.. <lora:legspread-flux:0.5> <lora:CumHereV1:0.5>
```

**Prompt 3 (Snapchat style):**
```
<lora:[FLUX]Noisify:0.8>, low light, noise, grain, jpeg artifacts, night, A low-quality 2015 Snapchat photo depicts a smiling woman in bed wearing a bra with cum on her mouth, cum drips, <lora:CumHereV1:1>
```

**Prompt 4 (Egyptian palace):**
```
Naked slender Egyptian girl with small perky breasts is lying on her back, wearing intricate golden Egyptian jewelry, exhausted, looking at viewer, scenic palace in Ancient Egypt, there is cum all over her body
```

**Prompt 5 (Amsterdam red light):**
```
half bodyshot photography from street at window, girl displayed behind behind window, view through glass, reflections in window, laughing hysterically, girl is a shy and ashamed brunette girl with big boobs, choker, wearing a neon pink skirt and black see through tube top, big breasts, huge cleavage, wearing clear pleaser 10inch plattform heels, garter, fishnet thigh high stockings, she is sitting on red bed between two old men touching her thigh and breast:1.6, in red light district street in amsterdam, she has cum on her face and chest, bukkake, she feels sad, feeling uncomfortable and shocked, she has a lot of cum on her face, facial cumshot, vertical purple neon tube lights along windows, pink neon letters "happy hour bukkake", spotlight on girl
```

### Prompting examples
- `There is cum on her ass and panties`
- `There is cum all over the front of her swimsuit`
- `There is cum on her tummy`
- `There is a lot of cum on her breasts and nipples`
- `There is cum all over her body`
- `creampie, cum in her pussy, cum drip`

### Tested combinations

**Combination 1 (Creampie focus):**
```
Checkpoint: UltraReal Fine-Tune v2.0
<lora:Creampie_Flux:0.5>
<lora:CumHereV1:1>
```

**Combination 2 (Full cum scene):**
```
Checkpoint: STOIQO Afrodite FLUX
<lora:Perfect_Full_Round_Breasts:1>
<lora:Cum_On_Face_FLUX:1>
<lora:CumHereV1:1>
<lora:NippleDiffusion-Flux:1>
```

**Combination 3 (Legspread + creampie):**
```
Checkpoint: Real Horny Pro V3
<lora:CumHereV1:1>
<lora:Legspread_Flux:1>
```

**Combination 4 (Noisy/amateur):**
```
<lora:Noisify:0.8>
<lora:CumHereV1:1>
```

**Combination 5 (Skinny body):**
```
Base: FLUX Dev
<lora:CumHereV1:1>
<lora:Flux_Skinny_Thinspo_Petite:1>
```

**Combination 6 (Amateur style):**
```
Base: FLUX Dev
<lora:Amateur_Flux:0.7>
<lora:Huge_cumshot_facial:0.85>
<lora:CumHereV1:0.9>
```

### Compatible checkpoints
- flux1-dev-fp8
- flux1-dev-fp16
- flux1-schnell (adjust weight)
- UltraReal Fine-Tune
- STOIQO Afrodite
- Real Horny Pro V3
- Fluxmania Kreamania

### Notes
- Complements facial cum LoRAs (use both for full coverage)
- Trained on realistic images, anime/cartoon results are hit or miss
- Works well with legspread, creampie, and body type LoRAs
- Lower weight if cum looks unnatural

# Mooning / Upskirt Flashing / Clothed-To-Nude for Flux

[← Back to Index](../INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 4,175 |
| **👍** | 301 |
| **Tips** | 1,180 |
| **Score** | ⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Mooning_Upskirt_Flux.safetensors` |
| **Original filename** | `mooning_upskirt_flux-000028.safetensors` |
| **Civitai** | https://civitai.com/models/725725 |
| **Trigger word** | `upskirt` / `mooning` (descriptive) |
| **Strength** | 0.9-1.0 |
| **Type** | POSE (Flashing/Exposure) |

## Description

Comprehensive LoRA for mooning, upskirt flashing, and clothed-to-nude scenarios. V3.5 is trained on a de-distilled model for much better quality with less overfitting. Works as a general nude model and handles underwear mechanics well.

**Capabilities:**
- Upskirt poses (sitting, standing, squatting)
- Mooning (pulling down pants)
- Clothed-to-nude transitions
- Proper underwear handling (pulled down)
- Public/ENF scenarios
- Works great with Flux Unchained for poses
- No pubes (trained dataset is clean)

**Training Notes:**
- V3.5 trained on true de-distilled model
- Text encoder training included
- Better variety of butt shapes (heart/round)
- Much less overfitting than previous versions

## Sample Prompts

### Upskirt Squatting Under Bridge
```
upskirt cute woman with pigtails wearing an adidas sweater squatting underneath a bridge in front of people, front shot
```
Settings: Steps 20, CFG 1, Euler, 1024x1024, Distilled CFG 3.5

### Mooning at Park Bench
```
mooning cute woman with pigtails wearing an adidas sweater and adidas tracksuit pants sitting at a park bench pulling down her pants, front shot
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5

### Upskirt Getting Off Bus
```
upskirt cute woman with pigtails wearing a purple polka dot dress getting off a bus, rear shot
```
Settings: Steps 20, CFG 1, Euler, 896x1152

### Upskirt Sitting on Bus
```
upskirt woman wearing a grey ruffle dress sitting on a bus with her legs spread out, front shot
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5

### Church Pew Upskirt
```
woman wearing a plaid skirt with cheeks slightly coming out. you can see the womans g string thong. woman is leaning over onto the church pews. packed church. wearing shoes <lora:Mooning_Upskirt_Flux:0.9>
```
Settings: Steps 40, DPM++ 2M, 832x1216

### Train Sleeping Upskirt
```
woman sitting in train. woman is wearing a dress with flower print design. legs spread open showing woman is not wearing underwear. woman is sleeping arms crossed. woman has bangs uneven. wearing a watch. black tinted glasses on. shoes on.
```
Settings: Steps 40, DPM++ 2M, 832x1216

### Gothic Girl Playground
```
gothic woman, long black hair with blue streaks, black t-shirt with skeleton graphic, black and red plaid mini-skirt, thigh-high lace-up socks, canvas high-top sneakers, sitting on playground slide, coming down slide with legs spread open, pussy on display, open field background, clear blue sky, rural setting
```
Settings: Steps 40, DPM++ 2M, 832x1216

### Park Bench Phone
```
woman sitting on park bench. woman is wearing a dress with flower print design. legs spread open showing woman is not wearing underwear. nice pussy. woman is using her phone. woman has bangs uneven. wearing bracelets. glasses on her head. cleavage exposed
```
Settings: Steps 40, DPM++ 2M, 832x1216

### Gym Sweater Lift
```
woman oversized sweater lifting up sweater showing pussy. in gym. sport shoes on. lifting up sweater.
```
Settings: Steps 40, DPM++ 2M, 832x1216

### Family Dinner Scene
```
slim skinny short young 19 year old pale young big breast, long black hair wearing pink long sleeve transparent, belt and fishnet leggings. also wearing a mesh skirt. no underwear. family having dinner at table. dad man has a mad shocked angry red face. inside a family living kitchen dinner table. Woman has red blush face. nipples puffy.
```
Settings: Steps 40, DPM++ 2M, 832x1216

## Keywords

### Main Triggers
- `upskirt` (required for upskirt scenes)
- `mooning` (required for mooning scenes)

### Poses/Actions
- `squatting`
- `sitting`
- `standing`
- `legs spread open`
- `pulling down pants`
- `lifting up sweater`
- `leaning over`

### Camera Angles
- `front shot`
- `rear shot`
- `side shot`

### Clothing
- `dress`
- `mini-skirt`
- `plaid skirt`
- `tracksuit pants`
- `thong`
- `g string`
- `no underwear`
- `and underwear` (adds pulled down underwear)

### Locations
- `bus`
- `train`
- `park bench`
- `church`
- `playground`
- `gym`

### Effects
- `cheeks slightly coming out`
- `not wearing underwear`
- `pussy on display`
- `vagina out`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-40 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler / DPM++ 2M |
| **Size** | 896x1152 / 832x1216 / 1024x1024 |
| **Strength** | 0.9-1.0 |

## Recommended Checkpoints

- **Flux Unchained** - Best synergy, improves poses and photorealism
- **Fluxed Up NSFW** - Good for explicit content
- **UltraReal Fine-Tune** - For maximum realism

## Recommended Combinations

### With Flux Unchained
```
<lora:Mooning_Upskirt_Flux:0.9>
Checkpoint: fluxunchainedNF4_fluxunchainedV11NF4
```

### With Realism
```
<lora:Mooning_Upskirt_Flux:0.9>
<lora:flux_realism_lora:0.6>
```

### With NSFW Enhancement
```
<lora:Mooning_Upskirt_Flux:0.9>
<lora:MysticXXX-v6:0.5>
```

## Notes

- Use `upskirt` or `mooning` keywords for best results
- Works with Flux Unchained for improved pose flexibility
- V3.5 uses de-distilled training (much better quality)
- Text encoder training included - fixes many quality issues
- No pubes in training data - model produces clean/shaved
- Add `and underwear` to any clothing to get pulled down underwear
- Good variety of butt shapes (trends toward heart/round)
- Works great in ComfyUI
- For Forge: set LoRA to fp16 mode


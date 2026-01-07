# FLUX Dildofun / Dildo LoRA NSFW

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 4,270 |
| **👍** | 186 |
| **Tips** | 0 |
| **Score** | ⭐⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `FLUX_Dildofun.safetensors` |
| **Original filename** | `Dildofun.safetensors` |
| **Civitai** | https://civitai.com/models/790753/flux-dildofun-dildo-lora-nsfw |
| **Trigger words** | `dild0`, `penisdild0`, `d0ubledild0`, `4nalp1ug`, `st00ldil0`, `vibrat0r`, `dild0lick1ng`, `b00tledild0` |
| **Strength** | 0.7-0.8 |
| **Type** | CONCEPT / Sex Toys |

## Description

Comprehensive dildo and sex toy LoRA for FLUX. Generates various dildo types, insertions, and toy-related scenes. Uses l33tspeak trigger words to bypass content filters.

**Toy types available:**
- `dild0` - Basic dildo
- `penisdild0` - Realistic penis-shaped dildo with veins
- `d0ubledild0` - Double-ended dildo
- `4nalp1ug` / `4naldild0` - Anal plug/dildo
- `st00ldil0` - Dildo mounted on stool
- `vibrat0r` - Vibrator
- `b00tledild0` - Bottle used as dildo
- `dild0lick1ng` - Dildo licking action

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 30-60 |
| **CFG** | 1-4 |
| **Sampler** | Euler / DDIM |
| **Scheduler** | Simple |
| **Size** | 1024x1024 / 832x1216 |
| **Strength** | 0.7-0.8 (0.3-0.4 with other LoRAs) |

## Sample Prompts

**Penis dildo in hand:**
```
penisdild0, a mature woman with visible pussy and labia is sitting on a wooden floor. she has spreaded her legs wide and open. she has a penisdild0 in her hand using it on her pussy. a penisdild0 with visible veins and testicles. she is looking at viewer. She is using dildo on her pussy. she is wearing just a tight tank top showing her deep cleavage. she performs in the middle of a large living room with large windows, plants, lights and a grey sofa. <lora:FLUX_Dildofun:0.75>
```
Settings: Steps 60, CFG 1, Euler + Simple, 1024x1024

**Bottle insertion (b00tledild0):**
```
b00tledild0, a mature woman with visible pussy and labia is squatting on a wooden floor. she is looking at viewer. She is using a bottle of clear liquid deeply in her vagina. vaginal insertion. bottle deep in vagina, riding a bottle. Her right hand is holding a tube with clear lube. she is wearing just a tight tank top showing her deep cleavage. she performs in the middle of a large living room. <lora:FLUX_Dildofun:0.75>
```
Settings: Steps 20-30, CFG 1, Euler + Simple, 1024x1024

**Stool mounted dildo:**
```
st00ldil0, r00m, a mature woman with visible pussy and labia is squatting on a stool. she is on top of low metal stool with a long penisdild0 stuck onto the stool's top riding a penisdild0 deep in her pussy. She is using a large black dildo in her vagina. dildo insertion in vagina. she is shocked with wide open eyes and wide open mouth. the dildo looks like a realistic penis with a red glans and strong veins. the stool is standing in the middle of a large living room. <lora:FLUX_Dildofun:0.75>
```
Settings: Steps 30, CFG 1, Euler + Simple, 1024x1024

**Anal dildo:**
```
4naldild0, a sideview mature woman with visible pussy and labia is squatting on a wooden floor. she is looking at viewer. She is using a large pink dildo in her anus. There is a vibrat0r on the floor next to her. she is wearing just a tight tank top showing her deep cleavage. she performs in the middle of a large living room with large windows, plants, lights and a grey sofa. <lora:FLUX_Dildofun:0.75>
```
Settings: Steps 30, CFG 1, Euler + Simple, 1024x1024

**Dildo licking:**
```
dild0lick1ng. A woman with dark skin and long, straight black hair is topless, flaunting medium-sized, natural boobs with dark nipples. She's got her tongue out, licking a big, purple dildo. Her face is relaxed and a bit cheeky, with her eyes half-closed and her lips slightly parted. she is performing in a rundown pub. <lora:FLUX_Dildofun:0.75>
```
Settings: Steps 35, CFG 4, 1024x1024

**With character LoRA (lower strength):**
```
(Based on your understanding of human posing, I would like you to create a new and unique pose for this picture). She is a naked woman with blond hair and blue eyes. She has one foot on a park bench and the second foot on the ground. She is holding a dildo with one hand. She is inserting the dildo into her vagina between her labia. <lora:dittersgurl-remix:0.8> <lora:FLUX_Dildofun:0.4>
```
Settings: Steps 20, CFG 2, Euler + Simple, 896x1152

## Trigger Words Reference

| Trigger | Type | Description |
|---------|------|-------------|
| `dild0` | Basic | Generic dildo |
| `penisdild0` | Realistic | Penis-shaped with veins/testicles |
| `d0ubledild0` | Double | Double-ended dildo |
| `4nalp1ug` | Anal | Anal plug |
| `4naldild0` | Anal | Anal dildo usage |
| `st00ldil0` | Mounted | Dildo on stool for riding |
| `vibrat0r` | Vibrator | Electric vibrator |
| `b00tledild0` | Improvised | Bottle used as dildo |
| `dild0lick1ng` | Action | Licking/sucking dildo |
| `r00m` | Setting | Living room setting |

## Best Checkpoints

- flux1-dev-fp8
- FLUX Checkpoint Dev
- getphat FLUX Reality
- acornIsSpinningFLUX (Hyper 8-step)

## Recommended Combinations

**With NSFW Master:**
```
<lora:NSFW_Master_Flux:0.8>
<lora:FLUX_Dildofun:0.8>
```

**With Character LoRA (use lower Dildofun strength):**
```
<lora:CHARACTER_LORA:0.8-1.0>
<lora:FLUX_Dildofun:0.3-0.4>
```

**For upscaling workflow:**
- SD15 Upscale model: AngrA RealFlex
- Upscaler: 4x_NMKD-Siax_200k
- flux realism lora

## Notes

- Use l33tspeak triggers (0 instead of o, etc.)
- Optimal strength 0.7-0.8 standalone
- Lower to 0.3-0.4 when combining with character LoRAs
- Higher steps (30-60) for better quality
- Low CFG (1-4) works best
- Works with various settings (living room, outdoor, pub, etc.)
- Can specify dildo colors and materials
- `r00m` trigger adds living room setting
- Trained on ~80 images at 1024x1024

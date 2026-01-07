# Asian Beauty Standard (Type B)

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 761 |
| **👍** | 70 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Asian_Beauty_Standard_Type_B.safetensors` |
| **Original filename** | `beauty_standard2.safetensors` |
| **Civitai** | https://civitai.com/models/772076 |
| **Trigger word** | None |
| **Strength** | 0.7-1.0 |
| **Type** | STYLE / Ethnicity |

## Description

Style LoRA that makes subjects more Asian. Proof of concept trained on 11 SD-generated images (1100 steps). Type B has better shadows than Type A.

- Increase strength if subject is not Asian enough
- Reduce strength if you get bad anatomy

## Sample Prompts

**Prompt 1 (Portrait - black shirt):**
```
head shot portrait photo of a beautiful 20yo woman, lacey black shirt, smiling
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.85

**Prompt 2 (Portrait - white shirt):**
```
head shot portrait photo of a beautiful 20yo woman, lacey white shirt, smiling
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.85

**Prompt 3 (Futuristic pantyhose):**
```
a beautiful woman in pantyhose, neon, futuristic, 30cm wide thighs, 20cm wide waist, sitting down, spreading legs, crotch, her knees are 1 meter apart, her hands are behind her supporting her body which is 30 degree tilted backwards
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.7

**Prompt 4 (School uniform amusement park):**
```
a beautiful 30yo woman in lacey shirt, school uniform jacket, plaid skirt, in an amusement park, smiling, v hand sign, 30cm wide thighs, 20cm wide waist
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.85

**Prompt 5 (Cyberpunk goth):**
```
masterpiece, neon splash art of a goth cyberpunk girl, straight-cut bob, bangs, black hair with red streak, earrings, necklaces, red thin lips, dark eyeshadow, choker, black nail varnish, one hand near her mouth, sensual, sexy, vibrant surreal colours
```
Settings: Steps 25, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.8

**Prompt 6 (Bunny girl):**
```
a young woman seated on an ornate chair. She is wearing white thighhighs with lace detailing, white gloves, and white panties. She has on white bunny ears, which are realistic in appearance. Her black hair is styled in a bob, and she is looking directly at the viewer. The background is blurry, but it seems to be an indoor setting with a hint of a yellow object, possibly a vase or decorative item.
```
Settings: Steps 20, CFG 1, Euler, 896x1152, Distilled CFG 3.5, Strength 0.8

**Prompt 7 (Interracial NSFW - combined with POV Missionary):**
```
score_9, score_8_up, score_7_up, 1girl, cute, young woman, sitting up, propped up by pillows, folded, penis inside vagina, legs up, perfect large breasts, skinny waist, trembling, enduring pain, painful, fucked rough, shiny skin, wet pussy, garter belt, thigh high stockings, topless, bottomless, skindentation, long hair, 1boy, dark-skinned male, hyper realistic, high quality photo 8k, beautiful asian, wet skin, (leggings ripped open to reveal vagina), intercourse, sex, vaginal discharge on penis, ((downward camera angle)), pov missionary, looking down at woman, view from above, face to face, downward camera angle, gorgeous Korean woman, Korean woman, viewer above woman, (torn clothes) to expose vagina, torn clothes, perfect face, extreme size difference, BBC, huge penis, excellent face generation, mind_break, enduring_face, wince, cum_in_pussy, wide open eyes, shiny skin, legs behind head, torn hole in clothes to expose vagina, cum covered penis shaft, extremely wet and shiny penis, torn, torn clothes, torn hole in clothes around vagina, missionarypose, 1boy, penis, folded, hands on own thighs, knees to chest, spread legs, vaginal, missionary, as1an, oiledskin, distress, explicit, pov, BBC_on_Fine_Females, 1woman, interracial porn, hetero, realistic
```
Combined with: POV Missionary 0.85, BBC on Fine Females 0.45
Settings: Steps 40, CFG 3.5, 832x1216, Strength 0.5

**Prompt 8 (Thai beach):**
```
a photograph, showcasing a sexy (naked:4) 24 year old (Thai Asian:2) woman, with long black hair in long twin braid, and cute happy hazel eyes, warm cute smile., kneeling on the beach, horny, mouth open, ready for cum, tongue out. Koh Samui beach, professional photography, cinematic, golden sunset light
```
Combined with: SRPO Asian Female Pubic Hair 1.0, Asian Nipples 0.7
Settings: Steps 20, CFG 12, 832x1216, Strength 1.0

**Prompt 9 (Twitch streamer blowjob):**
```
chinese, asian, korean, cum on face, fresh, young, young, age 18, sex, sucking dick, woman, realistic, 8k, 4k, photorealistic, high pigtails, large boobs, boob and face focus, black hair, large breasts, wet penis, sloppy saliva blowjob, twintails, cumshot on her face, twintails, blonde and pink highlights hair, light pink lingerie exposing boobs, ring light reflecting in her eyes,
BREAK
Twitch streamer computer room, neon lighting, gaming chair, high end gaming computer and monitor, light pink neon sign reads "give me your cum"
```
Combined with: Blowjob PoV 0.85, Fictional Model Emma 0.4
Settings: Steps 33, CFG 7, 832x1216, Strength 0.8

## Keywords

- No trigger word required
- `asian`, `korean`, `chinese`, `thai`
- `beautiful woman`, `20yo`, `30yo`
- Works with any prompt - just adds Asian features

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 20-40 |
| **CFG** | 1-7 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 896x1152 / 832x1216 |
| **Strength** | 0.5-1.0 (0.7-0.85 recommended) |

## Combinations

Works well with:
- POV Missionary - for interracial NSFW
- Blowjob PoV - for oral content
- BBC on Fine Females - for interracial
- Asian Nipples - for NSFW detail
- SRPO Asian Female Pubic Hair - for nude content

## Notes

- Type B has better shadows than Type A
- Minimal training (11 images, 1100 steps) - proof of concept
- Increase strength if subject not Asian enough
- Reduce strength if anatomy issues appear
- Lower strength (0.5) when combining with other LoRAs


# Blacked / Bleached / Custom Text Underwear

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,754 |
| **👍** | 205 |
| **Tips** | 2,010 |
| **Score** | ⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Blacked_Text_Underwear_V2.safetensors` |
| **Original filename** | `Blacked v2_000004650.safetensors` |
| **Civitai** | https://civitai.com/models/772191/blacked-bleached-whatever-underwear-read-description |
| **Trigger word** | None (use prompt structure) |
| **Strength** | 0.9-1.0 |
| **Type** | CLOTHING / Underwear |

## Description

Sports bra and thong underwear LoRA with **customizable text** on the clothing. Can generate text like "Blacked", "Bleached", "CivitAI", "Calvin Klein", or any custom text on the underwear band.

**Key feature:** Accidentally discovered ability to specify custom text on the underwear using the phrase structure below.

**Training notes:**
- V2 with increased dataset, re-cropped images
- Higher steps help with text quality
- Text won't be perfect every time
- Good compatibility with character LoRAs
- May need anti-bleed phrases for clean results

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 26-45 |
| **CFG** | 1-3.5 |
| **Distilled CFG** | 2-3.5 |
| **Sampler** | Euler |
| **Scheduler** | Simple / Normal / Beta |
| **Size** | 832x1216 / 896x1152 / 1024x1448 |
| **Strength** | 0.9-1.0 |

## How to Use - Custom Text

**Magic phrase structure:**
```
Wearing a [colour] sports bra and thong, with text reading "[YOUR TEXT]." on it
```

**Anti-bleed phrases (helps prevent unwanted elements):**
```
she isn't wearing jewelry, she is only wearing a sports bra and thong
```

## Sample Prompts

**Basic Blacked text:**
```
a super cute, thin 20 year old she is standing in a bedroom, Wearing a sports bra and thong, with text reading "Blacked." on it. looking at viewer, <lora:Blacked_Text_Underwear_V2:0.9>, she isn't wearing jewelry, she is only wearing a sports bra and thong. her arms and hands are behind her back, and her hair is in Ponytails. shy pose. eye level camera shot looking straight at subject.
```
Settings: Steps 28, CFG 1, Dist.CFG 3.5, Euler + Simple, 1024x1448

**Bleached text (white underwear):**
```
a athletic woman. kneeling on a black silk bed, Wearing a white sports bra and thong, with text reading "Bleached." on it. looking at viewer, <lora:Blacked_Text_Underwear_V2:1>, she isn't wearing jewelry, she is only wearing a sports bra and thong. her arms and hands are behind her back, and her hair is in Ponytails. shy pose. eye level camera shot looking straight at subject. her legs are parted.
```
Settings: Steps 28, CFG 1, Dist.CFG 3.5, Euler + Simple, 896x1152

**Custom text (CivitAI):**
```
a very thin woman. very skinny woman, thin legs, small waist, tiny waist. standing in themepark Wearing a orange sports bra and thong, with text reading "CivitAI." on it. looking at viewer, very skinny figure, very skinny body. <lora:Blacked_Text_Underwear_V2:1>, she isn't wearing jewelry, she is only wearing a sports bra and thong. her arms and hands are behind her back, and her hair is in a messy bun. standing straight, shy pose.
```
Settings: Steps 28, CFG 1, Dist.CFG 3.5, Euler + Simple, 896x1152

**With character LoRA (Panam Palmer):**
```
panam_palmer Wearing a sports bra and thong, with text reading "Blacked" on it, BLACKED text on the white band of her sports bra and thong, BLACKED, posing in the doorway of a cyberpunk motel room at night with neon lighting, her back to the viewer looking over her shoulder showing off her rear <lora:Blacked_Text_Underwear_V2:1>
```
Settings: Steps 25, CFG 3.5, 832x1216

**Calvin Klein style:**
```
A highly detailed Korean body profile of a 25-year-old fit and toned Korean woman with brown, wavy hair. Wearing a black sports bra and thong, with text reading "Calvin Klein" on it. The angle is from behind with her slightly looking back, muscular-female, blinds. <lora:Blacked_Text_Underwear_V2:1>
```
Settings: Steps 25, CFG 3.5, 1024x1024

**From behind pose:**
```
a mature curvy woman is standing in a street, Wearing a sports bra and thong, with text reading "Blacked." on it. looking at viewer, <lora:Blacked_Text_Underwear_V2:1>, she isn't wearing jewelry, she is only wearing a sports bra and thong. her hands are on her head, and her hair is in a Ponytail. shy pose. eye level camera shot looking straight at subject. from behind, looking back at viewer.
```
Settings: Steps 28, CFG 1, Dist.CFG 3.5, Euler + Normal, 1024x1448

**With FictionalVicky character:**
```
FictionalVicky short teal pixie cut hairstyle, slender frame, pale white skin, blue eyes. Wearing a sports bra and thong, with text reading "Blacked." on it <lora:Blacked_Text_Underwear_V2:1.2>
```
Settings: Steps 26, CFG 3.5, 832x1216 (with FictionalVicky LoRA 0.8)

**With Jessica Swan character:**
```
A realistic photo of Jessica Swan. She is standing and facing the viewer with a sexy pose. She is wearing a sports bra and thong, with text reading "Gunholio" on it. Her hair is nicely done up in a single high hairbun. She has womanly curves, fit and toned thighs. <lora:Blacked_Text_Underwear_V2:1>
```
Settings: Steps 26, CFG 2.5, 832x1216

**Redhead with SlutWear text:**
```
Portrait of a beautiful woman, looking_at_viewer, red hair, detailed_hair_style, curly hair, pale skin, freckles, blue_eyes, Perfect illumination, warm light, outdoors, Masterpiece, large breasts, thick thighs, small waist, round ass. standing in a crowded street Wearing a black lace sports bra and lace thong with cutouts, with text reading "SlutWear" on it. looking at viewer, hourglass figure, toned belly. <lora:Blacked_Text_Underwear_V2:1>, she isn't wearing jewelry, garter belt, underboob, sideboob, boobwindow, deep Cleavage. her arms and hands are behind her head, Arms lifted, loose Pony tail. standing slightly bent forward, seductive pose, lips parted.
```
Settings: Steps 28, CFG 3.5, 832x1216

**Bleached with black woman:**
```
a athletic black woman. kneeling on a black silk bed, Wearing a white sports bra and thong, with text reading "Bleached." on it. looking at viewer, <lora:Blacked_Text_Underwear_V2:1>, she isn't wearing jewelry, she is only wearing a sports bra and thong. her arms and hands are behind her back, and her hair is in Ponytails. shy pose. eye level camera shot looking straight at subject. her legs are parted.
```
Settings: Steps 28, CFG 3.5, 832x1216

**Bleached with Panam Palmer (dark skin):**
```
panam_palmer, dark skin girl, large breasts, Wearing a sports bra and thong, with text reading "Bleached" on it, "Bleached" text on the white band of her sports bra and thong, Bleached, posing in the doorway of a cyberpunk motel room at night with neon lighting, her back to the viewer looking over her shoulder showing off her big round ass <lora:Blacked_Text_Underwear_V2:1>
```
Settings: Steps 25, CFG 3.5, 832x1216

**Pakistani woman with group scene:**
```
photo of cute brown-skinned pakistani woman wearing a sports bra and thong, with text reading "Blacked." on it, in a changing cubicle, She has a toned petite body with wide hips, toned round bottom, narrow waist. She has thick lips and almond-shaped eyes. she is sitting in the middle of a large couch. her demeanor is innocent and playful, she is blushing profusely. <lora:Blacked_Text_Underwear_V2:1>. she has a seductive and playful smile. Behind the couch is a group of tall masculine bulky muscular beefy topless black men, standing in a line, assertively and with a dominant demeanor.
```
Settings: Steps 45, CFG 1, DPM++ 2M, Dist.CFG 2, 1664x1280, Beta scheduler

**Kenzie Tan character:**
```
A realistic photo of Kenzie Tan. She is standing and facing the viewer with a sexy pose, one hand on her hip. She is wearing a sports bra and thong, with text reading "KENZIE TAN" on it. Her hair is nicely done up in a single high hairbun. She has feminine curves, large breasts, fit and toned thighs. <lora:Blacked_Text_Underwear_V2:1.05>
```
Settings: Steps 26, CFG 2.5, 832x1216

**Curvy Brnda character:**
```
wavy pattern. Her glasses are thin and round, straight black hair and light brown skin. She is smiling warmly at the camera, full body photograph of a young woman with a bright, Br3n4. curvy woman is standing in a white room, Wearing a sports bra and thong, with text reading "Blacked." on it. looking at viewer, black pantyhose, black heels, big tits, chubby, big ass, big thighs, erect nipples, she is sitting in the middle of a large couch. her demeanor is innocent and playful, she is blushing profusely. <lora:Blacked_Text_Underwear_V2:1>. Behind the couch is a group of tall masculine bulky muscular beefy topless black men.
```
Settings: Steps 45, CFG 8.5, 1024x1024

## Keywords

- `sports bra and thong`
- `with text reading "..." on it`
- `Blacked`
- `Bleached`
- Custom text in quotes

## Color Variations

| Color | Usage |
|-------|-------|
| **White** | Best for "Bleached" text |
| **Black** | Classic look |
| **Orange** | Bright contrast |
| **Black lace** | Elegant variant |

## Text Examples That Work

- "Blacked."
- "Bleached."
- "CivitAI."
- "Calvin Klein"
- "SlutWear"
- Custom names
- Brand names

## Best Checkpoints

- flux1-dev (fp8, fp16)
- FLUX Checkpoint Dev

## Character LoRA Compatibility

Works well with:
- FictionalVicky
- Panam Palmer (Cyberpunk 2077)
- Jessica Swan
- Korean Body Profile
- Baby face FLUX
- Custom character LoRAs

## Notes

- Use the exact phrase structure for text to appear
- Higher steps (28-45) improve text quality
- Text won't be perfect every time - may need regeneration
- Add anti-bleed phrases to prevent unwanted elements
- Color of underwear can be specified
- Works with various body types and ethnicities
- Can combine with character LoRAs at 0.8-1.0 strength
- For complex scenes, may need weighted prompts

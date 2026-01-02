# Rich Girl Instagram Influencer

[← Back to INDEX](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `Flux Rich Girl Instagram Influencer_epoch_5.safetensors` |
| **Original filename** | `Flux "Rich Girl Instagram Influencer_epoch_5.safetensors` |
| **Civitai** | https://civitai.com/models/1420233/flux-rich-girl-instagram-influencer |
| **Trigger word** | None |
| **Strength** | 0.5-1.0 (typically 0.7) |
| **Type** | STYLE / Instagram Aesthetic |
| **Base Model** | Flux.1 Dev |
| **Training** | 300 steps, 5 epochs |

### Description

Instagram influencer aesthetic LoRA for FLUX. Similar to Thinspo LoRAs but produces a "healthier" figure by default. Generates typical Instagram scenery like luxury cars, private jets. Most training focused on creating an ideal face and figure, making it flexible for most settings.

**From author:** *"This Lora is similar to my Thinspo Loras but it produces a 'healthier' figure by default and also is able to produce some common Instagram type settings like a luxury car or private jet. Most of the training was done to create an ideal face and figure so it should work well for most settings and be pretty flexible."*

**IMPORTANT:** This LoRA was trained on Flux.1 Dev. Using it with NSFW merged models (like nsfwMASTER) causes blur and artifacts. Use with `flux1-dev.safetensors` for best results.

### Pose recommendations

- **Sitting poses** work best (author recommendation)
- Standing/walking poses - mix in XLabs Flux Realism LoRA
- V2 will include more standing positions

### Sample prompts

**Prompt 1 (Private jet - sitting):**
```
A stunning 24 year old rich girl instagram influencer, perfect face and figure, flawless tanned skin, toned fit body, long flowing blonde hair, perfect makeup, sitting pose in luxury private jet interior, wearing designer bikini, golden hour light, confident seductive expression, high end lifestyle
```
Settings: Steps: 30, CFG: 3.5, Sampler: euler, Size: 832x1216

**Prompt 2 (Luxury car):**
```
professional photo, high resolution, sharp focus, 8k, instagram aesthetic, A gorgeous 23 year old instagram influencer, perfect symmetrical face, slim toned body, long brunette hair, sitting in luxury sports car interior, wearing designer dress, afternoon sunlight, confident pose
```
Settings: Steps: 30, CFG: 3.5, Size: 832x1216

**Prompt 3 (Yacht):**
```
professional photo, ultra detailed, instagram aesthetic, Beautiful 25 year old rich girl influencer, flawless skin, athletic fit body, blonde hair in beach waves, sitting on luxury yacht deck, wearing white bikini, Mediterranean sea background, golden hour, seductive smile
```
Settings: Steps: 30, CFG: 3.5, Size: 832x1216

**Prompt 4 (Bathroom mirror selfie):**
```
professional photo, high resolution, sharp focus, instagram aesthetic, A stunning 24 year old instagram influencer, perfect makeup, toned fit body, long flowing hair, mirror selfie in luxury bathroom, wearing designer lingerie, golden hour light through window, seductive confident expression
```
Settings: Steps: 30, CFG: 3.5, Size: 832x1216

### Keywords

- `instagram influencer`
- `rich girl`
- `luxury lifestyle`
- `perfect face and figure`
- `high end`
- `designer`
- `sitting pose`
- `private jet`
- `luxury car`
- `yacht`

### CivitAI Example Prompts

**Prompt 1 (Car selfie - nude):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography)
This shot is of an 18 year old light skin woman of caucasian descent, she has petite delicate facial features and . Her hair is short and black. She is exceptionally pretty with dark brown eyes, long eye lashes and no makeup. She is slim and perfectly proportioned with perfect arms, perfect hands, perfect stomach, perfect face, symetrical face, perfect legs, perfect pout, perfect ass and a small pert chest. petite. She is also wearing glasses with purple frame.
She is wearing a pink opened shirt and pantyhose without panties.
She is sitting inside a car taking a selfie, breasts out, medium breasts, pussy,
She is looking directly down at the camera.
```
Settings: Steps: 25, CFG: 3.5, Size: 832x1216, Clip skip: 2

**Prompt 2 (Nurse changing room):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography)
This shot is of an 18 year old nurse light skin woman of caucasian descent, she has petite delicate facial features and . Her hair is short and black. She is exceptionally pretty with dark brown eyes, long eye lashes and no makeup. She is slim and perfectly proportioned with perfect arms, perfect hands, perfect stomach, perfect face, symetrical face, perfect legs, perfect pout, thich thighs, petite. She is also wearing glasses with purple frame.
She is wearing a sky blue opened shirt and stockings, without panties. sky blue scrubs top.
She is taking a selfie inside a changing room, breasts out, medium breasts, pussy,
She is looking directly down at the camera. tututu black thigh-high stockings with lace.
```
Settings: Steps: 25, CFG: 3.5, Size: 832x1216, Clip skip: 2
LoRAs: `Breasts PV @ 0.85`, `HiSilk FLUX stockings @ 0.9`, `Rich Girl Instagram Influencer @ 0.5`

**Prompt 3 (Car selfie - clothed):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography)
This shot is of an 18 year old light skin woman, she has petite delicate facial features and . Her hair is short, black. She is exceptionally pretty with dark brown eyes, long eye lashes and no makeup. She is slim and perfectly proportioned with perfect arms, perfect hands, perfect stomach, perfect face, symetrical face, perfect legs, perfect pout, perfect ass and a small pert chest. petite. She is also wearing glasses with purple frame.
She is wearing a pink shirt and black shorts with pantyhose.
She is sitting inside a car taking a selfie.
She is looking directly down at the camera.
```
Settings: Steps: 25, CFG: 7, Size: 832x1216, Clip skip: 2

**Prompt 4 (Car selfie - pantyhose only):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography)
This shot is of an 18 year old light skin woman, she has petite delicate facial features and . Her hair is short, black. She is exceptionally pretty with dark brown eyes, long eye lashes and no makeup. She is slim and perfectly proportioned with perfect arms, perfect hands, perfect stomach, perfect face, symetrical face, perfect legs, perfect pout, perfect ass and a small pert chest. petite. She is also wearing glasses with purple frame.
She is wearing only black pantyhose, she's completely nude.
She is sitting inside a car taking a selfie.
She is looking directly down at the camera.
```
Settings: Steps: 25, CFG: 7, Size: 832x1216, Clip skip: 2

**Prompt 5 (EMO girl dystopian):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography)
This picture is of a skinny, slutty EMO girl,
18 year old, pale skin woman, petite, delicate facial features, european, russian, long poker straight blue streaked highlighted black hair, high pony-tail, pretty slutty face, long eye lashes, heavy dark makeup, slim, perfect arms, perfect hands, perfect face, symetrical face, perfect pout, smirking, small chest, 1girl, perfect proportions, perfect stomach, standing, perfect legs, blue eyes, perfect ass, underbutt, skinny, pale skin, front view,
The low angle image, small feet in the foreground in an open stance, extreme perspective from the floor,
backdrop of futuristic dystopian city scene with skyscrapers, black matt ground
```
Settings: Steps: 25, CFG: 3.5, Size: 832x1216, Clip skip: 2

**Prompt 6 (Bedroom low angle):**
```
masterpiece, newest, absurdres, (hdr, hyper realistic, ultra realistic, soft light, realism, lifelike, award winning photography),
young 18 year old, light olive skin, skin woman, petite, delicate facial features, european, british, long straight mid-brown hair in a side parting, long lashes, no makeup, slim, perfect arms, perfect hands, perfect face, symetrical face, perfect pout, small B cup breasts, 1girl, perfect proportions, perfect stomach, perfect legs, blue eyes, dusky pink pyjama shorts and spaghetti strap top, perfect feet, The low angle image,
sitting in a velvet arm chair, legs spread, crotch showing, extreme perspective from the floor, dark carpet floor,
dusky pink background, dusky pink decorated girly bed room, mood lighting
```
Settings: Steps: 25, CFG: 3.5, Size: 832x1216, Clip skip: 2
LoRAs: `Low angle photography @ 0.9`, `Rich Girl Instagram Influencer @ 0.7`

**Prompt 7 (Manhattan apartment - from description):**
```
Sexy woman poses seductively in an expansive Manhattan apartment
```
LoRAs: `XLabs Flux Realism`, `Dreamgirl enhance detailer LOW`, `Flux Skinny Thinspo Petite`, `Rich Girl Instagram Influencer @ 0.7`

**Prompt 8 (Nude with feet focus):**
```
Realistic photo of a nude 18 year old woman lying on her back with spread legs and her feet up, dyed blonde hair, showing her pussy and anus, she has perfect tiny breasts and nipples, realistic, long hair, full lips, dreamy, detailed, running mascara, eyes open, scared, shocked, sweaty, anal, anal sex, impossible fit, rough assfuck, extreme size difference, rough anal sex, cum in ass, high class hotel room, perfect little feet in the foreground
```
Settings: Steps: 38, CFG: 3.5, Size: 832x1216, Clip skip: 2
LoRAs: `feet fetish for FLUX @ 0.8`, `Rich Girl Instagram Influencer @ 0.7`

### Tested combinations (from CivitAI examples)

**Combination 1 (Realism + Detailer):**
```
FLUX Dev + XLabs Flux Realism + Dreamgirl enhance detailer LOW + Rich Girl Instagram Influencer @ 0.7
```

**Combination 2 (Realism + Bolt-Ons):**
```
FLUX Dev + XLabs Flux Realism + Flux Bolt-Ons (Breast Implants) + Rich Girl Instagram Influencer @ 0.7
```

**Combination 3 (Skinny/Petite):**
```
FLUX Dev + XLabs Flux Realism + Dreamgirl enhance detailer LOW + Flux Skinny Thinspo Petite + Rich Girl Instagram Influencer @ 0.7
```

**Combination 4 (Hourglass body):**
```
FLUX Dev + Rich Girl Instagram Influencer @ 0.7 + Hourglass Body Shape FLUX
```

**Combination 5 (Blowjob + Stockings):**
```
FLUX Dev + Blowjobside AmateurAllure + Breasts PV @ 0.85 + HiSilk FLUX stockings @ 0.9 + Rich Girl Instagram Influencer @ 0.5
```

**Combination 6 (Feet fetish):**
```
FLUX Dev + feet fetish for FLUX @ 0.8 + Rich Girl Instagram Influencer @ 0.7
```

### Notes

- **MUST use with Flux.1 Dev** - not compatible with NSFW merged models (causes blur)
- Strength 0.5-1.0 (0.7 typical, 0.5 when mixing many LoRAs)
- NSFW content possible with Flux Dev + appropriate prompts
- Sitting poses most reliable
- Good for luxury/lifestyle Instagram aesthetic
- Format 4:5 (832x1216 → 1080x1350) matches Instagram post format
- XLabs Flux Realism helps with standing/walking poses

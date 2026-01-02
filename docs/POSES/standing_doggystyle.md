# Standing Doggystyle

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `Standing_Doggystyle.safetensors` |
| **Original filename** | `standingdoggylora.safetensors` |
| **Civitai** | https://civitai.com/models/947404/standing-doggystyle-non-face-altering |
| **Trigger word** | `St4nd1ngd0ggy` |
| **Strength** | 0.6-1.0 |
| **Type** | Pose / Sex Position |

## Description

LoRA for standing doggystyle sex position. Non-face altering - allows using favorite characters without changing their face. Trained in 1024 resolution using dedistilled, with close-ups for better penetration details. Mainly side and side/rear views, with some front view capability.

### Key features

- Standing doggystyle sex position
- Non-face altering (preserves character faces)
- Side view, rear view, and front view options
- Various hand positions (wall, table, furniture)
- Multiple leg positions (bent, raised, spread)
- Works with character LoRAs

### Known limitations

- ~40% success rate with distilled models (better with dedistilled)
- Anatomy issues possible
- Girl sometimes faces the guy (use additional prompting)
- Can rarely output two male heads
- May require multiple generations

## Recommended settings

- **Steps:** 20-60
- **CFG:** 1-5
- **Sampler:** Euler / DPM++ 2M
- **Scheduler:** Simple / Beta (0.6/0.6)
- **Distilled CFG:** 2.5-3.5
- **Size:** 832x1216, 768x1024

## Sample prompts

**Prompt 1 (Emo girl, rear view):**
```
<lora:Standing_Doggystyle:1>St4nd1ngd0ggy, Rear view of cute 18 years-old emo woman having sex with a man behind her, looking back at viewer, man inserting his penis in her pussy, she is bending over and looking viewer, she has her hands on a table. She has a slim petite body with small tits. She has emo hairstyle with colored hair. The background is an urbex ruined building room with graffitis on the walls and used condoms on the floor
```

**Prompt 2 (College student, front view):**
```
(cute topless European preppy college student with thick thighs having sex with a man behind her wearing only ribbed bright blue crotchless tights), A man behind her is inserting his penis into her pussy through a hole in her tights, St4nd1ngd0ggy, skintight high-waist ribbed bright blue crotchless tights, decorated cabin, Side view, The man is grabbing her hips and having sex with her through a hole in her tights, penis, she is bending forward, medium breasts, <lora:Standing_Doggystyle:1>
```

**Prompt 3 (Shower orgasm):**
```
St4nd1ngd0ggy, A girl having an orgasm in the shower while man thrusts his cock inside woman and grabs her ass. He thrusts hips back and forth into the woman. Her body twitches in orgasm convulsions, girl cums, her legs are buckling and girl squirt, squirting orgasm, her pants are getting wet, transparent liquid dripping from vaginal, A man behind her is inserting his penis into her pussy, Side view, Rear view, She is bending forwards, She is looking back, The man is grabbing her ass, The man is grabbing her hips, She has her hands against a wall
```

**Prompt 4 (Latina camgirl):**
```
(sexy topless Latina camgirl with thick thighs having sex with a man behind her wearing only glossy grey seamless pantyhose), A man behind her is inserting his penis into her pussy through a hole in her grey pantyhose, St4nd1ngd0ggy, colorful decorated dressing room, Front view, The man is grabbing her hips and having sex with her through a hole in her grey pantyhose, penis, she is bending forward, seamless, black hair in ponytail, <lora:Standing_Doggystyle:1>
```

**Prompt 5 (Vietnamese girl, hotel):**
```
close-up low key three-quarter angled rear view of Trang (Vietnam woman, age 19, petite, nutmeg skintone, shoulder-length wet straight black hair, small breasts, standing, bending over, head inverted, hands reaching down to touch mattress, petite round buttocks exposed, very happy surprised expression, clear viscous liquid dripped from between stretched labia), vaginal sex (st4ndingd0ggy) with Man (standing, legs apart, pressing his crotch against her buttocks, 40 years old, english (pale white skin, light skin contrasting with Trang's darker skintone), angled rear view, naked, massive large very thick very long wet penis extending from his groin inserted between Trang's small labia, his large hands pressing Trang's lower back). Cheap mattress on hard dirt floor. masterpiece photograph, underexposed, dramatic, deep shadows, very low key. <lora:Standing_Doggystyle:0.6>
```

## Keywords

- `St4nd1ngd0ggy` - **TRIGGER WORD** (required)
- `A man behind her is inserting his penis into her pussy`
- `Side view` / `Rear view` / `Front view`
- `She is bending forwards`
- `Side view of woman having sex with a man behind her`
- `She is looking back`
- `Rear view of a woman having sex from behind`
- `The man is grabbing her ass`
- `The man is grabbing her hips`
- `She has her hands against a wall`
- `She is against a wall`
- `She has one hand against a wall`
- `She has one leg raised with her knee bent`
- `She has one hand on furniture`
- `She has hands on a table`
- `Front view of a woman having sex from behind`
- `She has her hands on the man's legs`
- `She has her hands on a bar attached to the wall`

## Tested combinations

**Combination 1 (With big cock LoRA):**
```
<lora:Standing_Doggystyle:1> <lora:bick_cock_flux:0.7>
```

**Combination 2 (With MysticXXX):**
```
<lora:Standing_Doggystyle:0.6> <lora:MysticXXX-v7:0.5>
```

**Combination 3 (With seamless pantyhose):**
```
<lora:Standing_Doggystyle:1> <lora:seamless:0.2>
```

**Combination 4 (Amateur style):**
```
<lora:Standing_Doggystyle:1> <lora:MeltingPot05:0.7>
```

## Notes

- **Use dedistilled model for best results** (~40% success with distilled)
- Trigger word `St4nd1ngd0ggy` is required (l33t style)
- Non-face altering - great for character LoRAs
- Side/rear views work best, front view also possible
- Include detailed position descriptions for best results
- Use Beta scheduler (0.6/0.6) for quality
- Combine with big cock LoRAs for male anatomy
- Combine with MysticXXX for NSFW unlock
- Multiple generations may be needed for perfect anatomy

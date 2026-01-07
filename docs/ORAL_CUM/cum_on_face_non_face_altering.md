# Cum on face - Non-Face Altering

[← Back to INDEX](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `cumonfacelorav2.safetensors` |
| **Civitai** | https://civitai.com/models/858262/cum-on-face-flux-non-face-altering |
| **Trigger word** | `COF` |
| **Strength** | 0.8-1.0 |
| **Type** | CONCEPT / Cum effect |
| **Version** | v2.0 |

### Description
Cum on face LoRA that does NOT alter character faces - allows using with favorite character LoRAs. V2 retrained using De-distilled Flux with recaptioned dataset for better efficiency.

### Important: Distilled vs Dedistilled
- **Dedistilled models**: Full cum effect, works extremely well
- **Distilled models**: Less cum, need to max CFG and use all reinforcing words
- Recommended: Use Dedistilled models for best results

### Key features
- Non-face altering - safe for character LoRAs
- Works on face, body, mouth with Dedistilled
- V2 improved over V1

### Reinforcing words (important for Distilled)
```
she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue
```

### Alternative reinforcing words
```
white sticky cum on face, white sticky semen on face, white sticky sperm on face, face covered with white sticky cum
```

### Sample prompts

**Prompt 1 (Emo punk girl):**
```
<lora:cumonfacelorav2:1>COF, Above view of 18 years-old smiling emo punk girl with pink colored hairstyle, tongue out, above view, she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue. The scene happens in a rave during night
```

**Prompt 2 (Christmas market):**
```
Close-up Portrait of a very beautiful smiling woman called Elle wearing a super nice knitted white hat, knitted white gloves and a winter outfit, is standing on a christmas market and holding a Norwegian mulled wine mug in her hands. She has clear sticky cum with white reflections over her face. Her face is covered in cum after an enormous ejaculation. Christmas mood, love, fantasy, dreaming, cinematic <lora:cumonfacelorav2:1>
```

**Prompt 3 (Motorhead fan):**
```
A closeup photo of a gorgeous dark haired fan girl of the heavy metal band Motörhead is kneeling, mouth open, eyes closed. The point of view is from above her head, her face is towards the viewer. Her face is completely covered by cum, She has clear sticky cum with white reflections over her face, cum on nose, cum on lips, cum on chin, dripping cum, cum on eyes, cum on cheeks, cum on teeth, cum on forehead, cum dripping from cheeks, cum dripping from chin, cum on hair, front view, view from above, cum dripping from lips. Her face is almost completely covered by semen. Huge amount of sperm. She has a black "Motörhead Overkill" fan t-shirt on <lora:cumonfacelorav2:1>
```

**Prompt 4 (Satin maid):**
```
satin maid uniform, short-sleeve satin dress, white peterpan collar, white satin apron, skirtlift, a woman is lifting her dress to show her crotch area and vagina, COF, She has clear sticky cum with white reflections over her face, cum on nose <lora:cumonfacelorav2:1>
```

**Prompt 5 (Cumzilla style - works well):**
```
cumface woman with lots of white, thick, gooey cum all over and covering her face, cheeks, hair and forehead. The cum coats her face in a thick layer <lora:cumonfacelorav2:1>
```

### Keywords
- `COF` - **TRIGGER WORD**
- `clear sticky cum with white reflections`
- `dripping cum`
- `cum on forehead/cheeks/chin/eyes/lips/tongue`
- `cum dripping from chin`
- `face covered with cum`

### Prompting tip
Link reinforcing words with "with":
- GOOD: "Young woman face with sticky cum on face"
- BAD: "Young woman, sticky cum"

### Tested combinations
- MysticXXX LoRA
- Missionary POV LoRA
- Character LoRAs (Elle from Rick & Morty, etc.)
- Satin Maid LoRA
- Skirt Lift Concept LoRA
- Desi Espresso LoRA

### Compatible checkpoints
- FLUX Dev
- Fluxmania
- De-distilled Flux (RECOMMENDED)

### Notes
- Use trigger word `COF`
- Does NOT alter character face - safe for character combos
- Much better results with Dedistilled models
- For Distilled: max CFG, use ALL reinforcing words
- Can generate cum on other body parts with Dedistilled

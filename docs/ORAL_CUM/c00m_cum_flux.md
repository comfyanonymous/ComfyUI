# C00M - Cum [Flux]

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,502 |
| **👍** | 118 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `C00M_Cum_Flux.safetensors` |
| **Original filename** | `Sploosh.safetensors` |
| **Civitai** | https://civitai.com/models/1252954/c00m-cum-flux |
| **Trigger word** | `C00M` |
| **Strength** | 1.0-1.2 |
| **Type** | CUM EFFECTS (Facial/Body) |

## Description

High-quality cum effect LoRA trained on 84 manually selected and cropped photos. The trigger word `C00M` was introduced as a single token in training captions like "Photo of a woman with C00M on her face and in her mouth." This makes the model very aware of what C00M is and allows it to generalize across different settings.

## Key Features

- Single token trigger word `C00M`
- Generalizes well to new settings
- Face-masked training (works with character LoRAs)
- Pearlescent, stringy, gloopy texture
- Works on face, lips, eyelids, tongue, body

## Sample Prompts

**Basic Facial:**
```
Photo of a woman with her mouth open and tongue sticking out. She has C00M on her face and in her mouth. Stringy blobs of pearlescent C00M covers her face.
```
Settings: Steps: 30, CFG: 7.5, Sampler: DDIM, Strength: 1.0

**Pakistani Woman Hotel:**
```
high-angle highly-detailed candid uncensored mirror selfie photo of a cute endearing brown-skinned pakistani woman with gloopy C00M on her face lips and eyelids. she is shy and embarrassed and blushing deeply, holding an outburst of laughter. her expression and posture are restrained and playful. her skin is sweaty and she is wearing an embroidered cotton kameez. She is kneeling on the floor and has a playful smirk. her lips are pressed tightly shut to avoid the C00M slipping into her mouth.
```
Settings: Steps: 45, CFG: 1, Sampler: DPM++ 2M, Strength: 1.2

**Beach Scene:**
```
A professional realistic closeup photo of a woman, relaxed open mouth, naked at tropical beach, The woman is happy and smiling. Photo of a woman with her mouth open and tongue sticking out. She has C00M on her face and in her mouth. Stringy blobs of pearlescent C00M covers her face, she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue, cum all over her breasts
```
Settings: Steps: 16, CFG: 1, Sampler: Euler a, Strength: 1.0

**With Missionary LoRA:**
```
a high quality sharp photographic image of a sexy skinny young pretty smiling Filipina Asian woman with tan skin lying down on her back and looking at the viewer with tons of white, gooey, liquid cum covering her face. Full body visible, spreading legs open to reveal pussy covered in cum. She takes a long and hard cock into her vagina. Full body pov with her legs spread wide open for intensely provocative missionary sex @V@G, C00M
```
Settings: Steps: 20, CFG: 1, Strength: 0.25 (with other LoRAs)

**Nurse Scene:**
```
A well-lit medium shot captures a striking female nurse, standing confidently in a clinical corridor. She wears teal open medical scrubs—a V-neck top revealing her full breasts. Photo of a woman with her mouth open and tongue sticking out. She has C00M on her face and in her mouth. Stringy blobs of pearlescent C00M covers her face, she has clear sticky cum with white reflections over her face, dripping cum, cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, cum dripping from chin, cum on tongue, cum all over her breasts
```
Settings: Steps: 8, CFG: 1, Sampler: Euler, Strength: 1.0

## Keywords

- `C00M` (trigger - required)
- `opalescent stringy blobs of C00M`
- `a mix of chunky blobs and thick viscous liquid C00M`
- `gloopy C00M`
- `pearlescent C00M`
- `C00M on her face and in her mouth`
- `dripping cum`
- `cum on forehead/cheeks/chin/eyes/lips/tongue`

## Placement Keywords

| Location | Keywords |
|----------|----------|
| Face | `C00M on her face`, `cum on forehead`, `cum on cheeks` |
| Mouth | `C00M in her mouth`, `cum on tongue`, `cum on lips` |
| Eyes | `C00M on her eyelids`, `cum on eyes` |
| Body | `cum all over her breasts`, `pussy covered in cum` |

## Recommended Combinations

Works well with:
- **MysticXXX** - NSFW unlock
- **POV Missionary Vaginal** - Sex scenes with cum
- **Character LoRAs** - Face-masked training allows compatibility
- **TWbabeFlux** - Character enhancement

## Strength Guidelines

| Use Case | Strength |
|----------|----------|
| Solo use | 1.0-1.2 |
| With multiple LoRAs | 0.25-0.5 |
| Heavy cum effect | 1.2 |

## Notes

- Lower CFG (1-2) + more tokens = better realism
- Higher CFG = better prompt adherence but AI look
- Does NOT fill mouth, only on tongue and dripping
- 28 steps recommended
- Guidance 2-3.5 works well
- Training had no full faces (30-45% max) - good for character LoRAs
- Can add descriptors: `opalescent`, `stringy`, `gloopy`, `pearlescent`

## Training Info

- 84 manually selected photos (1024x1024)
- 8400 steps
- LR: 0.00010
- Batch size: 2
- Network dimensions: 32

---

*Last updated: 2026-01-02*

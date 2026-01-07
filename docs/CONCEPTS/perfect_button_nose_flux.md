# Perfect Button Nose for FLUX

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 394 |
| **👍** | 48 |
| **Tips** | 10 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Perfect_Button_Nose_Flux.safetensors` |
| **Original filename** | `Perfect_button_nose_small_upturned_nose_for_FLUX-000033.safetensors` |
| **Civitai** | https://civitai.com/models/1144665/perfect-button-nose-small-upturned-nose-for-flux |
| **Trigger word** | None |
| **Strength** | 0.6-1.0 |
| **Type** | CONCEPT (Facial Feature) |

## Description

LoRA for generating reliable small, upturned "button" nose shape. Trained on various images including close-ups, wide shots with censored features, and multiple views. Solves the common problem of getting this specific nose type consistently.

## Nose Type

- **Button nose** - small, rounded tip
- **Upturned** - tip points slightly upward
- **Small** - proportionally petite
- **Cute/youthful** appearance

## Sample Prompts

**Portrait with Metallic Makeup:**
```
23 years old, 1girl, solo, portrait, metallic makeup, makeup, eyelashes
```
Settings: Steps: 30, CFG: 3-4, Sampler: DPM++ 2M

**Columbian Girl:**
```
girl Columbian in thin black leggings and spandex sports bra stands with her front to the camera. She has a youthful figure. The softly lit room highlights her captivating beauty.
```
Settings: Steps: 30, CFG: 3.5

**Maid Uniform:**
```
girl Columbian in black satin maid uniform stands with her front to the camera, pastel-themed bedroom. Her youthful figure, adding a touch of elegance. The softly lit room highlights her captivating beauty.
```
Settings: Steps: 30, CFG: 3.5

**Close-up Face:**
```
The image is a high-resolution photograph focusing closely on a person's eye and part of their nose. The subject's eye is a striking blue, slightly glossy texture, revealing her face.
```
Settings: Steps: 30, CFG: 7.5, Sampler: DDIM

## Keywords

- `button nose`
- `small nose`
- `upturned nose`
- `cute nose`
- `petite nose`

## Recommended Combinations

Works well with:
- **Perfect Eyes** LyCORIS - Eye enhancement
- **Full lips and upturned ski sloped nose** - Additional nose refinement
- Character LoRAs
- Body shape LoRAs (Perfect Round Breasts, Perfect Round Ass)
- Expression Helper

## Strength Guidelines

| Use Case | Strength |
|----------|----------|
| Subtle enhancement | 0.6 |
| Standard use | 0.8-1.0 |
| Strong effect | 1.0+ |
| With other face LoRAs | 0.6-0.8 |

## Notes

- No trigger word needed - works automatically
- Trained on FLUX, works best with FLUX models
- Compatible with various checkpoints (CyberRealistic, etc.)
- Good for close-ups and portraits
- Combines well with other facial feature LoRAs
- Use with ADetailer for best face results

## Recommended Settings

- **Steps:** 20-34
- **CFG:** 3-4 (FLUX) or 7.5 (DDIM)
- **Sampler:** DPM++ 2M, DDIM, Euler
- **Use with:** ADetailer (face_yolov8n.pt)

---

*Last updated: 2026-01-02*

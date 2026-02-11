# Sitting on a Dildo

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 1,005 |
| **👍** | 55 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `sitting_on_dildo.safetensors` |
| **Original filename** | `dildo1.safetensors` |
| **Civitai** | https://civitai.com/models/821403/sitting-on-a-dildo |
| **Trigger word** | None (descriptive) |
| **Strength** | 0.8-1.2 |
| **Type** | CONCEPT / Pose / NSFW |

## Description

LoRA specifically for the "sitting on a dildo" pose. Trains FLUX to understand a woman lowering/sitting onto a dildo, typically on furniture. Works up to strength 1.2. No trigger words needed - describe the scene directly.

Trained from ~30 photos. Complementary to "Dildo Riding" LoRA but focused specifically on the sitting/lowering motion rather than active riding.

## Recommended Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 25 |
| **CFG** | 1 |
| **Sampler** | Euler / [Forge] Flux Realistic |
| **Scheduler** | Simple |
| **Distilled CFG** | 3 |
| **Upscaler** | 4x_NMKD-Siax_200k |
| **Hires Upscale** | 2x |
| **Hires Denoise** | 0.14 |

## Sample Prompts

**Prompt 1 (Fireplace - leaning over):**
```
female, dark brown hair, (wearing a black lace bra:1.5), she's sitting on a dildo in her living room, she leans over as she pleasures herself, her mouth is open, near the fire place, the room is dimly lit, it's late in the evening, as the light of the fireplace lights the room, her expression is one of pleasure, <lora:sitting_on_dildo:1.1>
```
Settings: Steps 25, CFG 1, Sampler: Flux Realistic/Euler, Distilled CFG 3, Size: 512x768 → Hires 2x

**Prompt 2 (Fireplace - eyes closed):**
```
female, dark brown hair, (wearing a black lace bra:1.5), she's sitting on a dildo in her living room, near the fire place, the room is dimly lit, it's late in the evening, as the light of the fireplace lights the room, her expression is one of pleasure, her eyes are closed as she looks at the view and smiles <lora:sitting_on_dildo:1>
```
Settings: Steps 25, CFG 1, Sampler: Flux Realistic/Euler, Distilled CFG 3, Size: 512x768 → Hires 2x

## Keywords

- `sitting on a dildo`
- `lowering herself onto dildo`
- `dildo on chair`
- `pleasuring herself`
- `dildo insertion`
- `riding`

## Recommended Combinations

**With GigiFlux character:**
```
<lora:GigiFlux-000002:1.4> <lora:sitting_on_dildo:1.0>
```

**With Dildo Riding for stronger effect:**
```
<lora:Flux_Dildo_Riding:0.7> <lora:sitting_on_dildo:0.8>
```

**With Dildofun for specific toy type:**
```
<lora:FLUX_Dildofun:0.4> <lora:sitting_on_dildo:1.0>
```

## Notes

- Strength up to 1.2 works well
- No trigger words needed - use descriptive language
- Works best with furniture (chair, stool, floor)
- Add character LoRAs for face quality
- Low CFG (1) with Distilled CFG 3 recommended
- Complements Dildo Riding LoRA (use both for stronger effect)
- Small file (38.8 MB)
- Best for static sitting poses, not active riding motion
- Creator: Niko3DX

---

*Last updated: 2026-02-10*

# Amateur Nudes - FLUX, Wan 2.2, ZIT

[← Back to CONCEPTS Index](INDEX.md)

## Info
- **File:** `Amateur_Nudes_FLUX.safetensors`
- **Original filename:** `nudes-3k-prodigy_rank24_bf16.safetensors`
- **Civitai:** https://civitai.com/models/764517/amateur-nudes-flux-wan-22-zit
- **Trigger:** None (optional: `Low quality cellphone photo of`)
- **Strength:** 0.75-1.0
- **Type:** CONCEPT

## Description
LoRA trained on ~50 amateur nude images, mostly selfies and mirror selfies. Moves away from perfect "model body" look toward more authentic amateur aesthetic. Knows "nude selfie" concept and responds to breast sizes/body shapes to some degree. Reduces Flux's tendency toward Asian women and cinematic color grading.

V1.5 is more consistent with less quality degradation. Art nude dataset added and nude selfie dataset expanded considerably.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 20-30 |
| CFG | 3.5 |
| Sampler | Euler / Undefined |
| Size | 896x1152 / 832x1216 / 1024x1024 |

## Prompt Format
Works without trigger. Optionally use:
```
Low quality cellphone photo of [description]
```

Or simply describe the scene:
```
nude selfie in messy bathroom
```

## Example Prompts

### Simple Nude Selfie
```
nude selfie in messy bathroom
```
Settings: Steps 20, CFG 3.5, Euler, 896x1152

### 1960s Retrofuturism - Asian
```
1960s retrofuturism, skinny asian woman nude. holding a sign with text: "Just nudes, I guess", straight bangs, suggestive smile, panavision, indoors
```
Settings: Steps 20, CFG 3.5, Euler, 896x1152

### 1960s Retrofuturism - Ginger
```
1960s retrofuturism, skinny ginger woman topless with red bikini bottom, straight bangs, suggestive smile, panavision, jungle with dappled light
```
Settings: Steps 20, CFG 3.5, Euler, 896x1152

### Cheerleader Flashing
```
18yo, blonde hair, long hair, large round breasts, youthful face, soft face, cute, pouty lips, almond eyes, button nose, full cheeks, petite, cheerleader outfit, cleavage, half body shot, hanging out the window of a yellow school bus, blue eyes, skinny, cool color tone, depth of field, 35mm, soft pink nipples, ((lifting shirt to flash breasts)), flashing breasts, embarassed, excited, amateurish photo, topless, low lighting
```
Settings: Steps 25, CFG 3.5, 1024x1024

### Dancing in Rain
```
A whimsical, cinematic realistic photo of a woman dancing joyfully in a rain-soaked street, wearing a transparent raincoat that flows with her movement and yellow knee-high rain boots splashing playfully in the puddles around her. She holds a brightly colored umbrella above her head, adding a pop of color to the scene. Her raincoat is sheer, revealing hints of her naked body with medium sized breasts underneath, and it glistens with raindrops that catch the soft, diffused light from overcast skies. The background reveals a blurred cityscape with misty rain falling, adding an ethereal feel to the setting.
```
Settings: Steps 25, CFG 3.5, 832x1216

### Korean Girl - Bathroom
```
nipples, nude selfie in a messy bathroom, korean girl with large sagging breasts
```
Settings: Steps 25, CFG 3.5, 832x1216

### 90s Nostalgia - Mature Woman
```
A candid, unfiltered indoor portrait of an attractive naked woman in her early 40s, standing in a warmly lit 1990s-style home. She has a healthy, curvy figure—neither overweight nor overly thin—gracefully aged like fine wine. Her shoulder-length blonde hair, styled in a casual, slightly layered cut reminiscent of the era, frames her naturally expressive face with a warm motherly smile. Her real, lived-in skin features freckles, faint moles, tan lines, sunspots, subtle acne scars, and visible pores, emphasizing a beauty untouched by excessive makeup or digital smoothing. The background is softly blurred but suggests a cozy 90s living space—perhaps a kitchen with wooden cabinets, a CRT television in the living room, or floral wallpaper typical of the decade. The lighting is warm and slightly diffused, mimicking the feel of amateur 90s film photography or a snapshot taken with a disposable camera. The image quality is slightly grainy, adding to the authentic, nostalgic feel.
```
Settings: Steps 30, CFG 3.5, 832x1216, Strength 0.75

## Keywords
- `nude selfie`
- `amateur`
- `mirror selfie`
- `messy bathroom`
- `amateurish photo`
- `Low quality cellphone photo of` (optional trigger)

## Recommended LoRA Combinations
- **16mm Film Emulator style** (1.0) - retro film look
- **UltraRealistic Lora Project** (0.85) - realism boost

## Best Checkpoints
- FLUX Dev

## Notes
- Works without trigger phrase
- Optional trigger: "Low quality cellphone photo of" for more authentic amateur look
- V1.5 version - more consistent, less quality degradation
- Moves away from "model body" toward authentic amateur look
- Reduces tendency toward Asian women
- Anatomy is not perfect - for accurate genitals use other specialized LoRAs
- Also available for WAN 2.2 T2V models and ZIT
- ZIT version has subdued effect, reduces cinematic color

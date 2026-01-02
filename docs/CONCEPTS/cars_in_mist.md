# Cars in Mist

[← Back to CONCEPTS Index](INDEX.md)

## Info
- **File:** `Cars_In_Mist.safetensors`
- **Original filename:** `Cars_in_mist.safetensors`
- **Civitai:** https://civitai.com/models/974428/cars-in-mist
- **Trigger:** None (descriptive prompts)
- **Strength:** 1.0-1.25
- **Type:** CONCEPT

## Description
LoRA for generating supersport cars in misty, atmospheric scenes with neon glow effects. Creates dramatic, cinematic car photography with fog, mist, and striking lighting.

## Recommended Settings
| Parameter | Value |
|-----------|-------|
| Steps | 30 |
| CFG | 7.5 |
| Sampler | DDIM / DPM++ 2M |
| Size | 512x512 (base) |

## Prompt Style
Use descriptive photography language:
```
high-resolution photograph, sleek [car brand/model], misty environment, dramatic lighting, neon glow, [color] accents
```

## Example Prompts

### Lamborghini with Neon Red
```
The image is a high-resolution, showcasing a prominent, and a prominent front splitter., golden light of either sunrise or sunset., with a sleek, capturing a sleek, The image is a high-resolution photograph capturing a dramatic, with a striking neon red outline around the car's edges and a red stripe along the lower body, and wheels. The rear view features the car's distinctive LED taillights and the Lamborghini logo prominently displayed on the trunk.
```
Settings: Steps 30, CFG 7.5, DDIM, 512x512

### Yellow Lamborghini Huracán
```
is painted in a striking yellow with black accents, aerodynamic design and striking black paint, a Lamborghini Huracán, angular rear., The image is a high-resolution photograph of a sleek, a hybrid hypercar, emitting a bright, misty environment. The car is depicted in a dark, golden light of either sunrise or sunset.
```
Settings: Steps 30, CFG 7.5, DDIM, 512x512

### Car with Red LED in Snow
```
casting a dramatic, This is a photograph taken in a dramatic, driving on a snowy, ethereal atmosphere., cinematic scene of a sleek, parked on a misty, and a prominent front splitter., The image is a high-resolution photograph capturing a dramatic, but the car has red LED lights installed
```
Settings: Steps 30, CFG 7.5, DDIM, 512x512

### Ferrari Hypercar Concept
```
Ferrari hypercar concept
```
Settings: Steps 30, CFG 7.5, DPM++ 2M, Strength 1.25

## Keywords
- `misty environment`
- `mist`
- `fog`
- `neon glow`
- `dramatic lighting`
- `high-resolution photograph`
- `sleek`
- `supersport car`
- `hypercar`
- `LED lights`
- `sunrise`
- `sunset`
- `cinematic`
- `ethereal atmosphere`

## Car Brands That Work Well
- Lamborghini (Huracán, etc.)
- Ferrari
- Hypercar concepts

## Recommended Upscaler
- Remacri

## Best Checkpoints
- FLUX Dev (691639)

## Notes
- No specific trigger word - use descriptive prompts
- Strength 1.0-1.25 depending on desired effect
- Creates atmospheric, dramatic car scenes
- Works well with neon/LED accent descriptions
- Higher CFG (7.5) for more defined results
- Good for both realistic and concept car designs
- Combine with weather descriptors (mist, fog, snow)

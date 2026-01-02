# GetOutOfCar Flux

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `GetOutOfCarFlux.0.9.2.safetensors` |
| **Civitai** | https://civitai.com/models/837811/getoutofcar-flux |
| **Trigger word** | `getoocar` |
| **Strength** | 1.0 |
| **Type** | Pose / Automotive |

## Description

Better poses for person getting out of car. Creates natural "exiting vehicle" poses with subject sitting in car with door open or legs out.

**Other versions available:**
- GetOutOfCar XL - for SDXL
- GetOutOfCar Pony - for Pony

### Prompt structure
```
<person description> sitting in a <color> car. getoocar. <clothing> <footwear>, <hair description>
```

## Sample prompts

**Prompt 1 (Simple):**
```
Slim girl sitting in a white car. getoocar.
```

**Prompt 2 (Wildcard randomizer):**
```
(slim 25yo girl) sitting in {black|red|blue|white|silver} car, getoocar, {green|red|blue|yellow|purple} {short skirt, {green|red|blue|yellow|purple} blouse|short dress|top and denim shorts} {boots|high heels|sneakers}, {long|short} {blonde|brunette} {hair|ponytail}
```

**Prompt 3 (Red car with dress):**
```
slim 25yo girl sitting in red car, getoocar, blue short dress, high heels, long blonde ponytail
```

**Prompt 4 (Sports car):**
```
slim 25yo girl sitting in silver sports car, getoocar, black top and denim shorts, sneakers, short brunette hair
```

## Keywords

- `getoocar` - **REQUIRED** trigger word
- `sitting in a <color> car`
- `car door open`
- `legs out of car`
- `exiting vehicle`

### Car colors
- black
- red
- blue
- white
- silver

### Clothing suggestions
- short skirt + blouse
- short dress
- top and denim shorts

### Footwear
- boots
- high heels
- sneakers

## Notes

- ForgeUI users: Set 'Diffusion in Low Bits' to 'Automatic (fp16 LoRA)'
- Always include `sitting in a ... car` and `getoocar` in prompt
- Works with various car colors
- Good for automotive/glamour photography
- Combine with clothing and character LoRAs

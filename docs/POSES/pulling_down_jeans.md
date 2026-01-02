# PullingDownJeans Flux (Configurator)

[← Back to Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `PullingDownJeansFlux.1.0.safetensors` |
| **Civitai** | https://civitai.com/models/812374/pullingdownjeans-flux-configurator |
| **Trigger word** | `pudoje` + optional switches |
| **Strength** | 1.0 |
| **Type** | Pose / Action / Configurator |

## Description

Better poses for pulling down jeans/denim shorts using hands. Configurable LoRA with multiple switches for different variations.

**Other versions available:**
- PullingDownJeans XL - for SDXL
- PullingDownJeans Pony - for Pony

**Note:** This is for PULLING DOWN jeans (using hands). LoRAs for already pulled down jeans (with free hands) coming soon for Flux.

### Trigger words / Switches

| Switch | Trigger | Effect |
|--------|---------|--------|
| **Main trigger** | `pudoje` | **REQUIRED** - activates the LoRA |
| **Far down** | `pufado` | Pulls jeans far down (shows vagina/butt) |
| **Denim shorts** | `densho` | Shows short denim shorts (Hot-pants) instead of long jeans |
| **No panties** | `nopaun` | No panties underneath the pulled down pants |

### Prompt structure
```
pudoje, (optional: pufado, densho, nopaun), <person description> <pulling action> (, no panties underneath)
```

**Pulling action options:**
- `pulling down jeans trousers a little` - slight pull (default)
- `pulling jeans trousers far down` - full exposure (requires `pufado`)
- `pulling down denim shorts a little` - slight pull, shorts (requires `densho`)
- `pulling denim shorts far down` - full exposure, shorts (requires `pufado` + `densho`)

## Sample prompts

**Prompt 1 (Denim shorts, far down, no panties - pub):**
```
pudoje, pufado, densho, nopaun, slim 25yo girl pulling denim shorts far down, no panties underneath, in a pub. sneakers, long blonde ponytail.
```

**Prompt 2 (Jeans trousers, a little - park):**
```
pudoje, slim 25yo girl pulling down jeans trousers a little, in a park, boots, short brunette hair.
```

**Prompt 3 (Jeans trousers, far down - mall):**
```
pudoje, pufado, slim 25yo girl pulling jeans trousers far down, in a mall. sneakers, long blonde hair.
```

**Prompt 4 (Denim shorts, a little, no panties - street):**
```
pudoje, densho, nopaun, slim 25yo girl pulling down denim shorts a little, no panties underneath, in a street, boots, short brunette ponytail.
```

**Prompt 5 (Medical selfie):**
```
A photo of a young woman taking a selfie in a medical setting. She is wearing light blue scrubs that accentuate her curves. Her hair is pulled back into a high ponytail. She is wearing a colorful medical face mask and latex gloves. A stethoscope is hanging around her neck. She is posing from the side, showcasing her fit figure and her nice cleavage. She is holding a smartphone in one hand, capturing the image in the reflection of a bathroom mirror. Paper towels are visible in the background. pulling jeans trousers far down, no panties underneath,
```

## Keywords

- `pudoje` - **REQUIRED** main trigger
- `pufado` - far down / full exposure
- `densho` - denim shorts / Hot-pants
- `nopaun` - no panties underneath
- `pulling down jeans trousers a little`
- `pulling jeans trousers far down`
- `pulling down denim shorts a little`
- `pulling denim shorts far down`
- `no panties underneath`

## Tested combinations

**Combination 1 (Style + Detail):**
```
<lora:PullingDownJeansFlux:1>
<lora:SinfullyStylish:1> (dramatic lighting)
<lora:DetailedPerfection:1>
```

## Notes

- ForgeUI users: Set 'Diffusion in Low Bits' to 'Automatic (fp16 LoRA)'
- **IMPORTANT:** Follow the prompt schema exactly for useful results
- Use strength 1.0
- Combine switches as needed for different variations
- Works well with location descriptions (pub, park, mall, street)
- Can combine with character LoRAs

# NippleDiffusion FLUX

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 47276 |
| **👍** | 1,607 |
| **Tips** | 1040 |
| **Score** | ⭐⭐⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `nipplediffusion-f1.safetensors` |
| **Civitai** | https://civitai.com/models/935673/nipplediffusion-flux |
| **Trigger words** | Optional - see list below |
| **Strength** | 0.7-1.5 |
| **Type** | Anatomy / Breasts / Nipples |
| **Version** | General LoRA v1.0 |

## Description

High-quality LoRA for detailed nipples and areolas with proper textures and depth. Curated from the best breasts and areolas dataset. Unlike other nipple LoRAs, this one provides realistic areola textures instead of plain, soulless circles.

**Note:** Negative weight puts clothes on (LOL).

## Trigger words (optional)

**Included in General LoRA:**

| Trigger | Effect |
|---------|--------|
| `big areolas` | Rounded medium to big areola size |
| `dark nipples` | Dark nipples with more contrast to skin |
| `ghost nipples` | Pale nipples that appear slightly |
| `long nipples` | Nipples that you can bite easily |
| `puffy nipples` | Small breasts with dense nipple tip |
| `saggy breasts` | Breasts hanging due to gravity |
| `small areolas` | Areolas with small radius |
| `veiny areolas` | Veins appearing under skin |
| `ginger nipples` | Nipples from redheads |
| `regular nipples` | Normal regular nipples/areolas - start here if unsure |

**Stand-alone LoRAs only (not in General):**

| Trigger | Effect |
|---------|--------|
| `banana tits` | Puffy-like, focused on breast shape |
| `silicone tits` | 90s spheres, focused on shape |
| `empty tits` | Similar to saggy but empty |
| `flat tits` | Flat chested woman |
| `oval areolas` | Cold-weather oval shape, good texture |
| `bumpy areolas` | Cold-weather bumpy texture |
| `wrinkled areolas` | Cold-weather wrinkled big areolas |

**Bonus triggers:**
- `hairy pussy` - hairy pussy of standing woman
- `shaved pussy` - shaved pussy of standing woman

## Sample prompts

**Prompt 1 (Vietnam motorcycle girl):**
```
Vietnam woman Trang (age 19, nutmeg skintone, shoulder-length straight black hair, thin, small breasts), standing in doorway of rustic ramshackle shack made of rough timbers and corrugated scrap metal sheets, in deep tropical jungle, at the end of a muddy trial, her small vintage metallic green 1972 Honda motorcycle leaned against the building exterior. Trang is wearing crimson pleated very short miniskirt hiked up high on her hips, black comfortable sneakers, topless (dark brown nipples), dark blue glitter 1970 vintage Bell helmet. <lora:Vietnam:.7> <lora:MysticXXX-v7:.7> <lora:nipplediffusion-f1:.4>
```

**Prompt 2 (Influencer selfie):**
```
Ultra realistic cinematic portrait of a young influencer lying nude on her pastel pink princess-style bed, taking a selfie with her smartphone, surrounded by plush toys, soft lighting, and an over-decorated girly bedroom, her look is heavily stylized with exaggerated makeup, glossy lips, and precisely contoured features, expression confident yet slightly artificial, long perfectly styled hair, smooth skin with signs of cosmetic enhancement subtly visible <lora:MysticXXX-v7:0.8> <lora:nipplediffusion:0.8>
```

**Prompt 3 (Fashion magazine sheer):**
```
This is a HiRes Top Fashion Magazine Photograph featuring a stark naked young woman with medium brown skin and long, straight, light brown hair, sitting on stone steps outdoors. She is draped in a white, one-shoulder covering, flowing, sheer white fabric with a sheer, embroidered overlay and gold thread floral patterns. The fabric is clinging to her nude body, revealing her face, nipples, breasts and pussy underneath. <lora:nipplediffusion-f1:0.55> <lora:pussydiffusion-f1:0.55>
```

**Prompt 4 (Beach Nikon):**
```
RAW photo, sharp focus, shot with Nikon Z7 II and Fujifilm XF 100-400mm f/4.5-5.6 R LM OIS WR at ISO 100, A young european naked woman with long brown hair, is sitting on a beach, posing for the camera. She is wearing black sunglasses. The scene is a tropical beach with palm trees, blue water, and a clear sky. pussy hair, pubic hair, large soft breast, nipples <lora:nipplediffusion-f1:0.6> <lora:pussydiffusion-f1:1>
```

## Keywords

- `dark nipples` / `ghost nipples` / `long nipples`
- `puffy nipples`
- `big areolas` / `small areolas`
- `veiny areolas`
- `saggy breasts`
- `regular nipples`
- `ginger nipples`

## Tested combinations

**Combination 1 (MysticXXX):**
```
<lora:nipplediffusion-f1:0.4-0.8>
<lora:MysticXXX-v7:0.7-0.8>
```

**Combination 2 (PussyDiffusion - Full anatomy):**
```
<lora:nipplediffusion-f1:0.5-0.6>
<lora:pussydiffusion-f1:0.5-0.6>
```

**Combination 3 (Extreme Detailer):**
```
<lora:nipplediffusion-f1:1>
<lora:FLUX_Pro_1.1_Extreme_Detailer:0.7>
```

**Combination 4 (Style mix):**
```
<lora:nipplediffusion-f1:0.5>
<lora:Cinematic_Glamour_F1D:1>
<lora:Midjourney_V7_FLUX:0.1>
```

**Combination 5 (Analog film):**
```
<lora:nipplediffusion-f1:0.6>
<lora:pussydiffusion-f1:1>
<lora:2000s_Analog_Core:0.1>
<lora:GrainScape_UltraReal:0.85>
```

## Tips

- Combine multiple trigger words for specific looks
- Use inpainting to improve specific areas
- Strength 0.7-1.5 depending on scenario
- Negative weight adds clothes (useful for partially dressed)
- May sometimes generate translucent horizontal pattern (rare)
- Flux does what it wants - use inpainting for precise control

## Notes

- Most detailed nipple/areola LoRA available
- Works well with PussyDiffusion for complete anatomy
- Trigger words are optional but help guide results
- Best combined with MysticXXX or style LoRAs
- Available as individual LoRAs for specific nipple types

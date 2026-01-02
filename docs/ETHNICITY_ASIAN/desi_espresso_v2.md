# Desi Espresso v2

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | N/A (404) |
| **👍** | N/A |
| **Tips** | N/A |
| **Score** | - |

**Note:** Model URL returned 404 - may have been deleted from Civitai.

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `desiespresso_flux_v2.safetensors` |
| **Civitai** | https://civitai.com/models/990802/desi-espresso-lora-for-indian-south-asian-faces-flux-1d |
| **Trigger word** | `desiespresso` (v2), `d351 d4rk` (v3) |
| **Strength** | 0.6-1.0 |
| **Type** | Character / Ethnicity |

## Description

LoRA for generating Indian/South Asian faces with various skin tones - from fair to very dark. Ideal for realistic portraits and photos of models with Asian heritage.

## Recommended Settings

- **Sampler:** Euler
- **Scheduler:** Beta
- **Steps:** 12-32
- **CFG:** 1
- **LoRA strength:** 0.6-1.0

## Sample Prompts

**Prompt 1 (Dark skin beauty):**
```
desiespresso, a dark-skinned Indian woman stands under soft, natural light, wearing a yellow chiffon saree draped gracefully over one shoulder, her hair in loose waves, large gold hoop earrings catching the light, in a traditional haveli courtyard with intricate marble carvings and arched doorways, warm sunset light casting golden reflections on her glowing skin
```

**Prompt 2 (Traditional saree):**
```
desiespresso, extremely high resolution photo, a stunning 25-year-old woman wearing red slik saree, natural makeup, soft lighting, wearing a simple blouse underneath, dark skin, professional photoshoot, studio lighting
```

**Prompt 3 (Bridal look):**
```
desiespresso, a gorgeous indian bride wearing pink lehnga choli at her wedding, gold jewellery, dark skin, intricate henna on hands, professional wedding photography, bokeh background
```

**Prompt 4 (Casual modern):**
```
desiespresso, closeup portrait, indian woman with fair skin, wearing modern western clothing, jeans and crop top, outdoor cafe setting, natural daylight
```

**Prompt 5 (Diverse skin tones):**
```
desiespresso, portrait of indian woman with dusky complexion, traditional nose ring, bindhi on forehead, minimal makeup, soft studio lighting
```

**Prompt 6 (NSFW - nude portrait):**
```
desiespresso, artistic nude portrait of indian woman, dark skin, soft natural lighting, tasteful pose, studio setting, professional photography
```

## Keywords

- `desiespresso` (trigger word)
- `dark-skinned Indian woman`
- `fair skin` / `dusky complexion` / `dark skin`
- `saree`, `lehnga`, `salwar kameez`
- `gold jewellery`, `henna`
- `haveli`, `traditional setting`

## Combinations with Other LoRAs

```
<lora:desiespresso_flux_v2:0.8>
<lora:detail_enhancer_flux_v1:0.7>
<lora:flux_realism_lora:0.6>
```

## Notes

- Supports various skin tones from fair to very dark
- Works well for both traditional and modern clothing
- Good for SFW and NSFW content
- Model may have been removed from Civitai (404 error on URL)

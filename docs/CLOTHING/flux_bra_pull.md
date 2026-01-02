# Flux Bra Pull

[← Back to CLOTHING Index](INDEX.md)

| Parameter | Value |
|-----------|-------|
| **File** | `flux_bra_pull_lora_v1.safetensors` |
| **Civitai** | https://civitai.com/models/903896/flux-bra-pull |
| **Trigger word** | `starting to remove her bra` |
| **Strength** | 0.8-1.0 |
| **Type** | Clothing / Undressing |
| **Training** | 36 hires images, natural language prompts |

## Description
LoRA for bra removal / undressing poses. Trained on girls undressing their bras but generalizes well to other clothing. Captures realism while remaining flexible. Quick to activate concept - bras pulled down to reveal breasts.

## Prompting
- Always include: `starting to remove her bra`
- For other clothing: mention wearing it, e.g., `starting to remove her dress, wearing a red strapless dress`

## Sample prompts

**Prompt 1 (Pakistani hijabi penthouse):**
```
highly detailed candid amateur photo of a cute brown-skinned pakistani hijabi woman. she is unbuttoning and starting to remove her bra and she is wearing low-waist thong, revealing bare shoulders, small saggy breasts and dark nipples. She has thick lips and almond-shaped eyes. her mood is playful and seductive, enjoying surprising the viewer with her naughty behaviour. She has a slim petite small body, delicate and dainty and she is barefoot. A highly detailed view of a modern penthouse apartment in London, bathed in natural light streaming through floor-to-ceiling glass windows. <lora:flux_bra_pull_lora_v1:1>
```
Settings: Steps: 45, CFG: 1, Sampler: DPM++ 2M, Size: 768x1024

**Prompt 2 (Black woman pool):**
```
A topless black biracial woman is standing in front of a swimming pool starting to remove her bra to flash her saggy breasts and show her dark nipples. You can see pubic hair sticking out from the bottom of her swim suit. She is directly facing the camera so we can see her whole body which is shapely and curvy realistic body shape. she has thick thighs and a small gut. Her hair is short, curly, and black. She is about 45 years old with strong arms. There are naked women in the pool facing the viewer. It looks like a formal portrait with a film camera, very realistic and photogenic. <lora:flux_bra_pull_lora_v1:1>
```
Settings: Steps: 24, CFG: 7, Size: 832x1216

**Prompt 3 (Pakistani living room):**
```
highly detailed candid amateur photo of a cute brown-skinned pakistani hijabi woman. she is unbuttoning and starting to remove her bra and she is wearing high-waist panties, revealing bare shoulders, small saggy breasts and dark nipples. She has thick lips and almond-shaped eyes. her mood is playful and seductive. She has a slim petite small body. A cozy Pakistani family living room in England, blending traditional and modern styles. The theme of the photo is playful and carefree with a hint of seduction and sensuality. <lora:flux_bra_pull_lora_v1:1>
```
Settings: Steps: 45, CFG: 1, Sampler: DPM++ 2M

## Keywords
- `starting to remove her bra` - **KEY PHRASE**
- `unbuttoning`
- `revealing bare shoulders`
- `small saggy breasts`
- `dark nipples`
- `playful and seductive`
- `candid amateur photo`

## Tested combinations
- desiespresso-v2-flux (0.4)
- Character/ethnicity LoRAs
- Flux Dev fp8

## Recommended settings
- **Steps:** 24-45
- **CFG:** 1-7
- **Sampler:** DPM++ 2M
- **Schedule:** Beta (alpha: 0.6, beta: 0.6) or Simple
- **Size:** 768x1024, 832x1216

## Notes
- Generalizes beyond bras to other clothing removal
- For other clothing: specify what she's wearing
- Trained with natural language - use descriptive prompts
- Works well with candid/amateur photo style
- Flexible with body types and ethnicities

## Quality Stats
- **Downloads:** 1,110
- **Rating:** 78
- **Tips:** $828

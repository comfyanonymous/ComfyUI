# FLUX FaeTastic Details

[← Back to INDEX](INDEX.md)

## Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 30110 |
| **👍** | 2745 |
| **Tips** | 3338 |

| Parameter | Value |
|-----------|-------|
| **File** | `FluxDFaeTasticDetails.safetensors` |
| **Civitai** | https://civitai.com/models/643886/flux-faetastic-details |
| **Trigger word** | None |
| **Strength** | 1.0 |
| **Type** | CONCEPT / Detail Enhancer |
| **Version** | v1.0 |

### Description
Detail concept LoRA for FLUX. Test training to see if detail concept could work on FLUX - and it does! Trained on Flux Dev. Works on Schnell but effect is weaker.

### Tested combinations
```
<lora:FluxDFaeTasticDetails:1>
<lora:aidmaFLUXPro1.1-FLUX-v0.3:1>
<lora:aidmaMJ6.1-FLUX-v0.5:0.4>
```

```
<lora:FluxDFaeTasticDetails:1>
<lora:Hand_F1D_v2:1>
<lora:aidmaFluxProUltra-FLUX-v0.1:1>
```

### Notes
- Works at full strength (1.0)
- Trained on Flux Dev
- Weaker effect on Schnell
- Ensure your workflow with LoRAs works properly
- Combines well with other detail enhancer LoRAs

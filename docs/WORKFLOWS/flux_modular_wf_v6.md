# FLUX Modular WF v6.0 (GGUF Edition)

[← Back to Index](INDEX.md)

## Info

| Parametr | Wartość |
|----------|---------|
| **Plik** | `FLUX Modular WF v6.0_gguf_final.json` |
| **Lokalizacja** | `ComfyUI\user\default\workflows\` |
| **Civitai** | https://civitai.com/models/1129063 |
| **Wersja** | 6.0 (Lipiec 2025) |
| **Typ** | All-in-one "Swiss Army Knife" |

## Opis

Kompleksowy workflow dla FLUX Dev.1 - "scyzoryk szwajcarski" łączący wszystkie podstawowe funkcje w jednym miejscu. Wersja GGUF pozwala na używanie kwantyzowanych modeli z mniejszym zużyciem VRAM.

## Moduły / Funkcje

### Generacja Obrazów

| Moduł | Opis | Użycie |
|-------|------|--------|
| **txt2img** | Tekst → Obraz | Podstawowa generacja |
| **img2img** | Obraz → Obraz | Modyfikacja istniejących obrazów |
| **Inpaint** | Malowanie fragmentów | Poprawki, usuwanie elementów |
| **Outpaint** | Rozszerzanie obrazu | Dodawanie tła, powiększanie kadru |

### Flux Kontext

| Moduł | Opis |
|-------|------|
| **Single Image** | Generacja z 1 obrazem referencyjnym |
| **Multi Image (3)** | Do 3 obrazów wejściowych |
| **Multi Output (4)** | 4 spójne warianty z 1 obrazu |

### ControlNet

| Typ | Zastosowanie |
|-----|--------------|
| **Depth** | Zachowanie głębi/perspektywy |
| **Canny** | Zachowanie krawędzi/konturów |
| **Union v2** | Uniwersalny model ControlNet |

### Post-Processing

| Moduł | Opis | Przydatne dla |
|-------|------|---------------|
| **HiRes Fix** | Upscaling z detalizacją | Wszystkie obrazy |
| **FaceDetailer** | Poprawa twarzy | Portrety kobiet |
| **Ultimate SD Upscaler** | Zaawansowany upscale + skin detail | Zdjęcia realistyczne |
| **Flux Redux** | Multi-image style transfer | Spójność stylu |

### LayerStyle (Color Grading)

| Node | Funkcja | Przykład użycia |
|------|---------|-----------------|
| **LayerColor: Exposure** | Jasność/Kontrast | Korekta świateł |
| **LayerColor: BrightnessContrastV2** | Balans tonów | Film look |
| **LayerColor: LUT Apply** | Gotowe presety kolorów | Cinematic grading |
| **LayerFilter: AddGrain** | Szum filmowy | Realizm, vintage |

## Wymagane Modele

### Podstawowe

| Typ | Folder | Pliki |
|-----|--------|-------|
| UNET (GGUF) | `models/unet/` | `flux1-dev-*.gguf` |
| VAE | `models/vae/` | `ae.safetensors` |
| DualCLIP | `models/clip/` | `t5xxl_*.safetensors`, `clip_l.safetensors` |

### Opcjonalne

| Typ | Folder | Do czego |
|-----|--------|----------|
| Style Model | `models/style_models/` | Flux Redux |
| CLIP Vision | `models/clip_vision/` | Kontext/Redux |
| Upscale | `models/upscale_models/` | Ultimate SD Upscaler |
| SAM | `models/sams/` | FaceDetailer |
| Ultralytics | `models/ultralytics/` | FaceDetailer (detekcja) |
| LUT | `models/luts/` | LayerColor: LUT Apply |

## Zalecane Ustawienia

### Dla Realistycznych Zdjęć

```
Model: flux1-dev-Q8_0.gguf (lub Q6_K dla mniej VRAM)
Steps: 20-30
CFG: 1.0
Distilled CFG: 3.5
Sampler: euler
Scheduler: simple
```

### Dla Portretów Kobiet

```
FaceDetailer: ON
Ultimate SD Upscaler: ON (skin detail)
LayerColor: Exposure +0.1-0.2
LayerFilter: AddGrain 0.02-0.05
```

### Dla Samochodów

```
ControlNet Depth: ON (zachowanie proporcji)
HiRes Fix: ON
LayerColor: LUT Apply (automotive preset)
```

### Dla Krajobrazów

```
Outpaint: rozszerzenie kadru
LayerColor: LUT Apply (landscape/cinematic)
LayerFilter: AddGrain 0.01-0.03
```

## Tips & Tricks

1. **Fast Groups Muter (rgthree)** - Szybkie włączanie/wyłączanie modułów jednym kliknięciem
2. **Power Lora Loader** - Łatwiejsze zarządzanie wieloma LoRA
3. **GGUF** - Użyj Q8_0 dla jakości, Q4_K_M dla oszczędności VRAM
4. **Bypass grupy** - Workflow ma zorganizowane grupy, które można bypass'ować

## Porównanie z innymi Workflows

| Feature | Modular WF v6 | Kontext Simple | Basic txt2img |
|---------|---------------|----------------|---------------|
| txt2img | ✅ | ✅ | ✅ |
| img2img | ✅ | ✅ | ❌ |
| Inpaint/Outpaint | ✅ | ❌ | ❌ |
| Multi-Kontext | ✅ | ❌ | ❌ |
| FaceDetailer | ✅ | ❌ | ❌ |
| Color Grading | ✅ | ❌ | ❌ |
| GGUF Support | ✅ | ❌ | ❌ |
| Complexity | Wysoka | Niska | Minimalna |

## Changelog

- **v6.0** (Lipiec 2025): Redukcja z 35 do 14 wymaganych node packs, dodanie Flux Kontext, ControlNet Union v2
- **v5.1**: Bugfix
- **v5.0**: Pierwsza pełna wersja

---

*Last updated: 2026-01-03*

# FLUX Kontext Txt2Img Workflows

[← Back to Index](INDEX.md)

## Pliki

| Workflow | Plik |
|----------|------|
| Ultimate (ControlNet Union) | `flux_text2img_4.5_Ultimate (CN Union).json` |
| Ultimate + LLM | `flux_text2img_4.5_Ultimate_LLM_20-06-2025.json` |

**Lokalizacja:** `ComfyUI\user\default\workflows\`

**Źródło:** [FLUX.1-DEV & Kontext Workflows Megapack](https://civitai.com/models/1129063)

---

## flux_text2img_4.5_Ultimate (CN Union).json

### Opis

Zaawansowany workflow txt2img z obsługą ControlNet Union v2 - uniwersalnego modelu ControlNet obsługującego wiele typów kontroli w jednym.

### Funkcje

| Moduł | Opis |
|-------|------|
| **ControlNet Union v2** | Depth, Canny, Pose, Tile w jednym modelu |
| **Adetailer** | Automatyczna poprawa twarzy |
| **Ultimate SD Upscaler** | Upscaling z detalizacją |
| **Metadata Overlay** | Wyświetlanie parametrów na obrazie |

### ControlNet Union - Tryby

| Tryb | Użycie | Dla |
|------|--------|-----|
| Depth | Zachowanie głębi | Samochody, architektura |
| Canny | Zachowanie krawędzi | Logo, grafiki |
| Pose | Pozycja ciała | Portrety, pozy |
| Tile | Detale tekstur | Upscaling |

### Wymagany Model

```
models/controlnet/
└── controlnet_union_v2.safetensors
```

---

## flux_text2img_4.5_Ultimate_LLM_20-06-2025.json

### Opis

Wersja Ultimate z dodatkowym modułem LLM (Large Language Model) do automatycznego ulepszania promptów.

### Jak działa LLM Enhancement

```
Twój prompt → LLM → Rozszerzony, szczegółowy prompt → FLUX → Obraz
```

### Przykład

| Input | LLM Output |
|-------|------------|
| "kobieta przy samochodzie" | "A beautiful woman in her 30s standing next to a sleek black sports car, golden hour lighting, professional photography, shallow depth of field, urban background..." |

### Kiedy używać

| Sytuacja | Rekomendacja |
|----------|--------------|
| Krótkie, proste prompty | ✅ LLM pomoże |
| Długie, szczegółowe prompty | ❌ Może "przesolić" |
| Eksperymentowanie | ✅ Szybkie iteracje |
| Precyzyjna kontrola | ❌ Lepszy manual |

### Wymagania

- Zainstalowany node LLM (np. ComfyUI-LLM)
- Model LLM (lokalny lub API)

---

## Porównanie Workflows

| Feature | Ultimate | Ultimate+LLM |
|---------|----------|--------------|
| ControlNet Union | ✅ | ✅ |
| Adetailer | ✅ | ✅ |
| Upscaler | ✅ | ✅ |
| Prompt Enhancement | ❌ | ✅ (LLM) |
| Complexity | Średnia | Wysoka |
| VRAM | Średnie | Wysokie |

---

## Zalecane Ustawienia dla Twoich Projektów

### Kobiety (Portrety)

```
ControlNet: OFF lub Pose
Adetailer: ON
Steps: 25-30
CFG: 1.0
```

### Samochody

```
ControlNet: Depth (z obrazem referencyjnym)
Adetailer: OFF
Steps: 20-25
CFG: 1.0
```

### Krajobrazy

```
ControlNet: OFF lub Depth
Adetailer: OFF
Steps: 20
CFG: 1.0
```

### Okładki Blogowe (SEO/Programowanie)

```
ControlNet: Canny (dla logo/grafik)
LLM Enhancement: ON (dla szybkich iteracji)
Steps: 20
```

---

*Last updated: 2026-01-03*

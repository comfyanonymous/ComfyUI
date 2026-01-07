# FLUX Kontext Img2Img Workflows

[← Back to Index](INDEX.md)

## Pliki

| Workflow | Plik |
|----------|------|
| Basic Img2Img | `flux_img2img_1_2.json` |
| Img2Img + HighresFix | `flux_img2img_1_3_HighresFix.json` |

**Lokalizacja:** `ComfyUI\user\default\workflows\`

**Źródło:** [FLUX.1-DEV & Kontext Workflows Megapack](https://civitai.com/models/1129063)

---

## flux_img2img_1_2.json

### Opis

Podstawowy workflow img2img dla FLUX Dev. Pozwala na modyfikację istniejących obrazów z zachowaniem kompozycji.

### Zastosowania

- Zmiana stylu istniejącego zdjęcia
- Poprawki kolorystyczne
- Dodawanie/usuwanie elementów
- Zmiana ubrań/tła

### Parametry

| Parametr | Zalecane |
|----------|----------|
| Denoise | 0.3-0.7 (im wyższy, tym więcej zmian) |
| Steps | 20-30 |
| CFG | 1.0 |
| Distilled CFG | 3.5 |

---

## flux_img2img_1_3_HighresFix.json

### Opis

Img2Img z wbudowanym upscalingiem. Generuje w niższej rozdzielczości, następnie upscaluje z detalizacją.

### Zastosowania

- Tworzenie wysokiej rozdzielczości z img2img
- Poprawa detali przy transformacji
- Lepsze twarze i detale skóry

### Workflow

```
Input Image → Downscale → Img2Img → Upscale → Detail Pass → Output
```

### Zalety vs Podstawowy

| Cecha | Basic | HighresFix |
|-------|-------|------------|
| Szybkość | Szybszy | Wolniejszy |
| Detale | Standardowe | Lepsze |
| VRAM | Mniej | Więcej |
| Rozdzielczość wyjściowa | = Input | Wyższa |

---

## Tips

1. **Denoise strength** - Kluczowy parametr:
   - 0.2-0.4: Subtelne zmiany, zachowanie kompozycji
   - 0.5-0.7: Znaczące zmiany, nowy styl
   - 0.8+: Prawie jak txt2img

2. **Dla portretów kobiet** - używaj niższego denoise (0.3-0.5) aby zachować rysy twarzy

3. **Dla samochodów** - średni denoise (0.4-0.6) pozwala zmienić kolor/tło zachowując kształt

---

*Last updated: 2026-01-03*

# Responsive Mockup Generator - Poradnik

## Lokalizacja Workflow
`ComfyUI/user/default/workflows/Responsive_Mockup_Generator.json`

---

## Metoda 1: FLUX + Zewnętrzna Kompozycja (Zalecana)

### Krok 1: Wygeneruj bazową scenę

Użyj promptu:
```
Professional product photography of a modern minimalist desk setup with three devices: a large computer monitor in the center, a tablet on the left side, and a smartphone on the right side. All devices have blank white screens. Clean white/gray background, soft studio lighting, high-end commercial photography style, 8K, ultra detailed
```

**Ustawienia:**
- Steps: 25-30
- CFG: 3.5
- Sampler: euler
- Size: 1216x832 (landscape) lub 832x1216 (portrait)

### Krok 2: Kompozycja w edytorze graficznym

1. **Otwórz** wygenerowany obraz w Photoshop/GIMP/Figma
2. **Importuj** screenshoty (desktop, tablet, mobile)
3. **Perspective Transform** każdy screenshot aby pasował do ekranu
4. **Blend mode:** Normal lub Multiply dla lepszego wtopienia
5. **Dodaj efekty:**
   - Lekki glow na krawędziach ekranu
   - Subtelne odbicie światła
   - Cień pod urządzeniami

---

## Metoda 2: ComfyUI z Inpainting

### Krok 1: Wygeneruj scenę z maskami

1. Wygeneruj bazowy obraz z urządzeniami
2. Zapisz obraz
3. W edytorze graficznym stwórz **maski** (białe obszary = ekrany)
4. Załaduj maskę do ComfyUI

### Krok 2: Użyj ImageCompositeMasked

```
Node: ImageCompositeMasked (ComfyUI_essentials)
- destination: bazowy mockup
- source: twój screenshot
- mask: maska ekranu
- x, y: pozycja (dostosuj)
```

---

## Alternatywne Prompty

### Floating Devices (popularne)
```
Three floating devices against soft gradient background: iPhone 15 Pro, iPad Pro, and MacBook Pro, all displaying blank white screens, minimalist style, soft drop shadows, commercial product mockup, ultra realistic, 8K
```

### Isometric View
```
Isometric 3D view of responsive web design mockup, desktop monitor, tablet and smartphone arranged diagonally, blank screens, clean white background, soft ambient occlusion, modern tech aesthetic
```

### Cozy Desk Setup
```
Modern wooden desk with 27-inch monitor, iPad, and iPhone arranged to show responsive website, blank screens, cozy home office background, plants, natural window lighting, lifestyle photography
```

### Dark Mode Setup
```
Sleek dark desk setup with gaming monitor, tablet and phone, blank screens with dark gray color, RGB ambient lighting, modern tech aesthetic, professional product photography
```

### On Marble Surface
```
Luxury marble surface with rose gold iPhone, iPad Pro and MacBook Air, blank white screens, soft shadows, minimalist high-end product photography, fashion brand style
```

---

## Wskazówki

### Dla lepszych ekranów:
- Używaj `blank white screens` lub `blank gray screens`
- Unikaj `website`, `app` w promptcie - to doda random content
- Dodaj `no text on screens, clean displays`

### Rozmiary screenshotów:
| Urządzenie | Proporcje | Przykład |
|------------|-----------|----------|
| Desktop | 16:9 / 16:10 | 1920x1080, 1920x1200 |
| Tablet | 4:3 / 3:4 | 1024x768, 768x1024 |
| Mobile | 9:19.5 | 390x844, 375x812 |

### Perspective Transform tips:
- Ekrany rzadko są idealnie płaskie w mockupach
- Dodaj lekkie zaokrąglenie na krawędziach
- Użyj Warp tool dla naturalnego wyglądu

---

## Przykładowy Pipeline

```
1. FLUX Dev → Generuj bazę (25 steps)
2. Zapisz PNG
3. Photoshop:
   - Import screenshotów
   - Free Transform → Perspective
   - Warstwa → Screen blend mode (opcjonalnie)
   - Dodaj reflection overlay
4. Export finalny mockup
```

---

## Narzędzia Online (alternatywa)

Jeśli potrzebujesz szybko:
- **Mockup World** - darmowe PSD mockupy
- **Smartmockups** - online generator
- **Placeit** - web-based mockups

Ale z FLUX masz pełną kontrolę nad sceną!

---

*Utworzono: 2026-01-02*

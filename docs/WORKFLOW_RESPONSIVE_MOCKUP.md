# Responsive Mockup Generator - Workflow

## Lokalizacja
`ComfyUI/user/default/workflows/Responsive_Mockup_Complete.json`

---

## Cel
Generowanie profesjonalnych mockupów pokazujących responsywną stronę internetową na 3 urządzeniach:
- Monitor (desktop)
- Tablet (iPad)
- Telefon (iPhone)

---

## Wymagane Custom Nodes

| Node Pack | Funkcja | Status |
|-----------|---------|--------|
| ComfyUI_essentials | ImageComposite, ImageScale | ✅ Zainstalowany |
| comfy_mtb | MTB_TransformImage (shear, rotate) | ✅ Zainstalowany |
| was-node-suite-comfyui | Dodatkowe nody kompozycji | ✅ Zainstalowany |

---

## Instrukcja Krok po Kroku

### KROK 1: Wygeneruj Bazową Scenę

1. Otwórz workflow w ComfyUI
2. Edytuj prompt w node **"PROMPT - Edit this!"**
3. Kliknij **Queue Prompt**
4. Poczekaj na wygenerowanie obrazu z urządzeniami
5. **Zapisz** wygenerowany obraz (automatycznie do `output/mockup/`)

#### Przykładowe Prompty

**Floating Devices (domyślny):**
```
Professional product photography, three Apple devices floating against soft white gradient background: 27-inch iMac monitor in center, iPad Pro on left angled slightly, iPhone 15 Pro on right angled slightly. All devices show completely blank white screens, no content. Soft drop shadows beneath devices, minimalist commercial mockup style, studio lighting, ultra realistic, 8K quality
```

**Isometric View:**
```
Isometric 3D mockup of three devices: MacBook Pro laptop, iPad Pro tablet, iPhone 15 smartphone, arranged diagonally from top-left to bottom-right, all with blank white screens, soft shadows, minimal pure white background, commercial product photography
```

**On Desk (Lifestyle):**
```
Modern wooden desk setup with 27-inch iMac monitor, iPad Pro on stand, and iPhone laying flat, all devices showing blank white screens, cozy home office background with plants, natural window lighting, lifestyle product photography, warm tones
```

**Dark Mode:**
```
Three devices on dark matte surface: ultrawide monitor, tablet, smartphone, all with blank dark gray screens, subtle ambient RGB lighting in background, gaming/tech aesthetic, professional product photography, moody lighting
```

### KROK 2: Przygotuj Screenshoty

Przygotuj 3 screenshoty swojej strony:

| Urządzenie | Rozdzielczość | Proporcje |
|------------|---------------|-----------|
| Desktop | 1920x1080 | 16:9 |
| Tablet | 1024x768 lub 768x1024 | 4:3 |
| Mobile | 390x844 lub 375x812 | 9:19.5 |

**Wskazówki:**
- Screenshoty powinny być czyste, bez paska przeglądarki
- Dla mobile użyj widoku pionowego
- Dla tabletu możesz użyć poziomego lub pionowego

### KROK 3: Załaduj Obrazy

1. Załaduj **Desktop Screenshot** do node #20
2. Załaduj **Tablet Screenshot** do node #21
3. Załaduj **Mobile Screenshot** do node #22
4. Załaduj **Zapisany Base Mockup** do node #23

### KROK 4: Dostosuj Skalowanie

Edytuj nodes **ImageScale** (30, 31, 32):

| Node | Urządzenie | Domyślny rozmiar | Dostosuj do |
|------|------------|------------------|-------------|
| #30 | Desktop | 500x310 | Rozmiar ekranu monitora |
| #31 | Tablet | 180x240 | Rozmiar ekranu iPada |
| #32 | Mobile | 80x170 | Rozmiar ekranu iPhone'a |

### KROK 5: Dostosuj Pozycje

Edytuj nodes **ImageComposite** (40, 41, 42):

| Node | Urządzenie | Domyślne X,Y | Opis |
|------|------------|--------------|------|
| #40 | Desktop | 580, 150 | Środek obrazu (monitor) |
| #41 | Tablet | 150, 200 | Lewa strona (iPad) |
| #42 | Mobile | 1150, 180 | Prawa strona (iPhone) |

**Wskazówka:** Wartości X,Y zależą od wygenerowanego obrazu. Eksperymentuj!

### KROK 6: Generuj Finał

1. Kliknij **Queue Prompt**
2. Sprawdź podgląd w node **"6. FINAL MOCKUP"**
3. Jeśli pozycje nie pasują - wróć do kroku 5
4. Zapisz finalny mockup

---

## Zaawansowane: Dodanie Perspektywy

Jeśli ekrany w base mockup są pod kątem, użyj **MTB_TransformImage** przed ImageComposite:

```
MTB_TransformImage
├── shear: -5 do 5 (pochylenie)
├── angle: -10 do 10 (obrót)
├── zoom: 0.9 do 1.1 (skalowanie)
└── x, y: przesunięcie
```

### Przykładowy Setup dla Pochylonego iPada:
1. Dodaj **MTB_TransformImage** po ImageScale #31
2. Ustaw `shear: 3` i `angle: -5`
3. Podłącz output do ImageComposite #41

---

## Rozwiązywanie Problemów

### Screenshot nie pasuje do ekranu
- Dostosuj rozmiar w ImageScale
- Zmień pozycję X,Y w ImageComposite

### Ekrany są krzywe w base mockup
- Wygeneruj nowy base z innym seedem
- Lub użyj MTB_TransformImage dla perspektywy

### Jakość screenshotów jest niska
- Użyj screenshotów w wyższej rozdzielczości
- W ImageScale użyj metody `lanczos`

### Krawędzie screenshotów są widoczne
- Dodaj lekki blur na krawędziach (w edytorze graficznym)
- Lub użyj inpainting w ComfyUI

---

## Alternatywne Podejście: Zewnętrzna Kompozycja

Dla najlepszych rezultatów z perspektywą:

1. Wygeneruj base mockup w ComfyUI
2. Otwórz w **Photoshop/GIMP/Figma**
3. Importuj screenshoty
4. **Edit → Free Transform → Perspective**
5. Dopasuj do każdego ekranu
6. Dodaj efekty:
   - Lekki glow na krawędziach
   - Odbicie światła na ekranie
   - Cień pod urządzeniami

---

## Struktura Plików

```
ComfyUI/
├── user/default/workflows/
│   ├── Responsive_Mockup_Generator.json   (podstawowy)
│   └── Responsive_Mockup_Complete.json    (pełny)
├── output/mockup/
│   ├── base_devices_00001.png
│   └── final_responsive_00001.png
└── docs/
    └── WORKFLOW_RESPONSIVE_MOCKUP.md      (ten plik)
```

---

*Utworzono: 2026-01-02*

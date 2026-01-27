# SEO Blog Cover Generator

[← Back to Index](INDEX.md)

## Opis

Workflow do generowania profesjonalnych okładek dla artykułów blogowych o tematyce SEO, marketingu i technologii. Obsługuje dwa style:

1. **Flat Design** - minimalistyczne ilustracje wektorowe
2. **Realistyczne zdjęcia biurowe** - z użyciem Office LoRA

---

## Dostępne warianty

| Workflow | Plik | Styl |
|----------|------|------|
| SEO Blog Cover | `SEO_Blog_Cover.json` | Flat design / ilustracje |
| SEO Blog Cover Office LoRA | `SEO_Blog_Cover_Office_LoRA.json` | Realistyczne zdjęcia biur |

---

## Parametry

| Parametr | Wartość |
|----------|---------|
| **Rozmiar** | 1200x900 (4:3) - 3x większe niż 400x310 |
| **Steps** | 30 |
| **CFG** | 3.5 |
| **Sampler** | euler / simple |
| **Model** | flux1-dev lub visionRealistic |
| **LoRA** | JJs_Office_Flux (opcjonalnie) |

---

## Przykładowe prompty

### Flat Design - WordPress Security
```
Flat design illustration for blog cover, WordPress security concept, shield icon with padlock, server rack with protection symbols, minimalist style, blue and green color palette, clean geometric shapes, professional tech illustration, vector art style, no text
```

### Flat Design - Backup
```
Flat design illustration, cloud backup concept, server connected to cloud storage, data transfer arrows, backup icons, blue and orange color scheme, minimalist geometric style, tech illustration, isometric perspective, clean lines
```

### Flat Design - AI/LLM
```
Flat design illustration, artificial intelligence concept, robot head with neural network connections, data streams, machine learning icons, purple and blue gradient background, modern tech illustration, geometric shapes, futuristic style
```

### Flat Design - Reddit SEO
```
Flat design illustration, Reddit social media marketing concept, upvote arrows, community icons, discussion bubbles, orange and white color scheme, minimalist style, social engagement graphics
```

### Realistyczne biuro - SEO Analytics
```
((Office)), modern digital marketing agency, large monitors displaying Google Analytics and SEO tools, keyword ranking charts, traffic graphs on screens, glass wall, professional workspace, blue accent lighting, natural daylight, minimalist design, monitors, keyboard, laptop, notebook, coffee cup, indoors, window, chair, table, dramatic lighting
```

### Realistyczne biuro - Whiteboard Strategy
```
((Office)), marketing strategy meeting room, large whiteboard with SEO keywords and flowcharts, conference table, monitors showing search rankings, modern glass office, professional atmosphere, ceiling lights, clean minimal design, monitor, screen, chair, table, dramatic lighting
```

### Realistyczne biuro - WordPress Security
```
((Office)), IT security workspace, multiple monitors displaying WordPress admin panels and security dashboards, server status screens, dark mode interfaces, professional tech environment, dramatic blue lighting, keyboard, monitor arrays, clean desk, glass wall, modern interior
```

### Realistyczne biuro - Data Visualization
```
((Office)), data analytics workspace, large curved monitor showing colorful data visualization charts, pie charts and bar graphs, modern minimalist desk, plants, natural light from windows, clean Scandinavian design, laptop, notebook, coffee, professional atmosphere
```

---

## Tematyka artykułów - mapowanie promptów

| Temat artykułu | Sugerowany styl | Kluczowe elementy |
|----------------|-----------------|-------------------|
| Kopie zapasowe WordPress | Flat design | cloud, backup icons, server, data transfer |
| Zabezpieczenie WordPress | Flat design / Office | shield, padlock, security dashboard |
| Ukrycie wersji WP | Flat design | code brackets, hide icon, security |
| AI i LLM | Flat design | robot, neural network, data streams |
| Reddit SEO | Flat design | upvote arrows, community, orange |
| Optymalizacja wizualna | Office | monitors with images, analytics |
| Strategia SEO | Office | whiteboard, charts, meeting room |

---

## Wskazówki

### Dla Flat Design:
- Używaj słów kluczowych: `flat design`, `minimalist`, `vector art style`, `geometric shapes`
- Określ paletę kolorów: `blue and green`, `orange and white`, `purple gradient`
- Dodaj `no text` aby uniknąć wygenerowanego tekstu
- Perspektywa `isometric` dodaje głębi

### Dla Realistycznych zdjęć biur:
- ZAWSZE używaj triggera `((Office))` na początku promptu
- Dodawaj elementy: `monitor`, `keyboard`, `glass wall`, `dramatic lighting`
- Dla scen bez ludzi dodaj do negative: `people, person, human`
- Kolory: `blue accent`, `natural daylight`, `minimalist design`

### Negative prompt:
```
text, watermark, logo, ugly, blurry, low quality, distorted, deformed, amateur
```
Dla wersji bez ludzi dodaj: `people, person, human`

---

## Użycie

1. Otwórz workflow w ComfyUI
2. Edytuj prompt w nodzie "POSITIVE PROMPT"
3. Opcjonalnie zmień rozmiar w nodzie "Blog Cover"
4. Kliknij "Queue Prompt"
5. Obrazy zapisują się w `output/seo_blog_cover/`

---

*Last updated: 2026-01-07*

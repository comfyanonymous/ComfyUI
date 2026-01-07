# Generowanie wielu postaci (Multiple Characters)

[← Back to INDEX](INDEX.md)

## Metody generowania wielu postaci

### 1. Przez prompt (najprostsze)

Opisanie wielu osób bezpośrednio w prompcie:

```
two women standing together, blonde woman on the left wearing red dress,
brunette woman on the right wearing blue dress, both smiling at camera
```

**Zalety:**
- Proste, nie wymaga dodatkowych narzędzi
- Działa od razu

**Wady:**
- Flux może mieszać cechy obu postaci (np. obie blondynki)
- Mała kontrola nad poszczególnymi postaciami
- Trudne z więcej niż 2 postaciami

---

### 2. Kilka Character LoRA naraz

Użycie wielu LoRA postaci w jednym workflow:

```
<lora:CharacterA:0.7> <lora:CharacterB:0.7>
Prompt: two women, CharacterA on the left, CharacterB on the right...
```

**Zalety:**
- Możliwość użycia znanych postaci
- Lepsza kontrola niż sam prompt

**Wady:**
- LoRA się "mieszają" - obie postacie mogą wyglądać podobnie
- Wymaga balansowania wag (zwykle zmniejszyć do 0.5-0.7 każda)
- Trigger words mogą się nakładać

**Tips:**
- Zmniejsz wagi LoRA gdy używasz wielu (np. 0.5-0.6 zamiast 1.0)
- Użyj wyraźnych opisów pozycji: "on the left", "on the right"
- Dodaj różne cechy dla każdej postaci w prompcie

---

### 3. Regional Prompting (najlepsze dla kontroli)

Różne prompty dla różnych regionów obrazu - każda postać ma swój dedykowany obszar.

**Wymagane narzędzia:**
- ComfyUI-Impact-Pack (zainstalowany w `custom_nodes/`)
- Regional Prompt nodes

**Jak działa:**
1. Definiujesz regiony (np. lewa/prawa połowa obrazu)
2. Każdy region ma własny prompt
3. Każdy region może mieć własne LoRA

**Zalety:**
- Najwyższa kontrola nad każdą postacią
- LoRA nie mieszają się między regionami
- Możliwość różnych stylów dla każdej postaci

**Wady:**
- Wymaga bardziej skomplikowanego workflow
- Więcej czasu na konfigurację

**Przykładowy workflow:** `custom_nodes/ComfyUI-Impact-Pack/tests/workflows/regional_prompt.json`

---

### 4. Inpainting (najbardziej kontrolowane)

Generowanie postaci jedna po drugiej:

1. Wygeneruj obraz z jedną postacią
2. Zamaluj region gdzie ma być druga postać
3. Użyj inpainting z nowym promptem dla drugiej postaci
4. Powtórz dla kolejnych postaci

**Zalety:**
- Pełna kontrola nad każdą postacią
- Brak mieszania się cech
- Można używać pełnej mocy LoRA dla każdej postaci

**Wady:**
- Czasochłonne (wiele przejść)
- Wymaga umiejętności maskowania
- Może być widoczna granica między regionami

---

## Podsumowanie

| Metoda | Kontrola | Trudność | Czas | Jakość |
|--------|----------|----------|------|--------|
| Prompt | ★☆☆☆☆ | Łatwa | Szybki | Zmienna |
| Multi-LoRA | ★★☆☆☆ | Średnia | Szybki | Średnia |
| **Regional Prompting** | ★★★★☆ | Średnia | Średni | Dobra |
| Inpainting | ★★★★★ | Trudna | Wolny | Najlepsza |

---

## Rekomendacje

- **Szybkie testy:** Użyj promptu lub multi-LoRA
- **Produkcja:** Regional Prompting lub Inpainting
- **Specyficzne postacie (character LoRA):** Regional Prompting

---

## TODO

- [ ] Stworzyć przykładowy workflow Regional Prompting dla 2 postaci
- [ ] Stworzyć przykładowy workflow Inpainting dla dodawania postaci
- [ ] Przetestować kombinacje character LoRA z różnymi wagami

---

*Last updated: 2026-01-02*

# Car Crash

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 354 |
| **👍** | 46 |
| **Tips** | 100 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Car_Crash_Flux.safetensors` |
| **Original filename** | `Car crash.safetensors` |
| **Civitai** | https://civitai.com/models/730253 |
| **Trigger word** | None (descriptive prompts) |
| **Strength** | 0.7-1.3 |
| **Type** | CONCEPT / Vehicle / Destruction |

## Description

LoRA for generating destroyed, crashed, and burning vehicles. Works with various car brands.

## Sample Prompts

**Prompt 1 (Burning Toyota):**
```
a photo of a burning destroyed Toyota Camry after a car crash, At an old gas station, people standing around, firefighters, police officers, <lora:Car_Crash_Flux:1>
```

**Prompt 2 (Tree crash):**
```
a photo of a totally destroyed Toyota Yaris after a tree crash, Near sand dunes, Austria, people standing around, firefighters, police officers, <lora:Car_Crash_Flux:1.3>
```

**Prompt 3 (Miniature rally):**
```
<lora:microworlds_flux:1> microworldlora, a photo of a crashing miniature (totally destroyed:1.3) wrc rally car <lora:Car_Crash_Flux:0.7>
```

**Prompt 4 (Abandoned wreck):**
```
a photo of a abandoned forgotten car wreck in the bushes, <lora:Car_Crash_Flux:1.1>
```

**Prompt 5 (Jeep Mexico):**
```
a photo of a (totally destroyed:1.3) Jeep Wrangler, Underpass or overpass, Mexico, people standing around, smoke, firefighters, police officers, <lora:Car_Crash_Flux:1.3>
```

**Prompt 6 (Epic Everest):**
```
A heavily battered Range Rover L322, bearing the deep scars of a catastrophic crash, claws its way up the treacherous slopes near Everest's base at nearly 8000 meters above sea level. Torn metal panels rattle in the howling wind, and shattered windows offer little protection against the swirling, ice-laden snow.
```

## Keywords

- `destroyed`, `totally destroyed`, `burning`, `wrecked`, `crashed`
- `car crash`, `firefighters`, `police officers`
- Car brands: Toyota, Jeep, Mercedes, Aston Martin, Range Rover, etc.

## Settings

Steps 20, CFG 1, Euler, 1280x720, Distilled CFG 3.5, Strength 0.7-1.3


# Futuristic Cars Flux

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 355 |
| **👍** | 34 |
| **Tips** | 2,425 |
| **Score** | - |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `Futuristic_Cars_Flux.safetensors` |
| **Original filename** | `Futuristic_Cars_v2.5_flux.safetensors` |
| **Civitai** | https://civitai.com/models/726437 |
| **Trigger word** | `FC3, Futuristic_Cars` |
| **Strength** | 0.8-1.0 |
| **Type** | CONCEPT (Vehicles) |

## Description

Collection of futuristic cars trained for FLUX. Creates cyberpunk and sci-fi style vehicles with wide bodies, intricate designs, and post-apocalyptic aesthetics. Works great for vehicle focus shots, scene compositions, and combining with other vehicle LoRAs.

**Capabilities:**
- Cyberpunk style cars
- Post-apocalyptic vehicles
- Wide body race cars
- Alien-themed vehicles
- Rusted/weathered cars
- Offroad crawlers
- Glowing night city scenes

## Sample Prompts

### Basic Cyberpunk Car
```
FC3, Futuristic_Cars cinematic lighting, cyberpunk car, futuristic car, ground vehicle, wide body car, intrincated style car
```
Settings: Steps 35, CFG 4, Euler

### Post-Apocalyptic Crawler with Weapons
```
FC3, Futuristic_Cars cinematic lighting, cyberpunk car, futuristic car, ground vehicle, wide body car, intrincated style car, wide wheels, offroad, weapons on roof, crawler, post-apocalyptic scene
```
Settings: Steps 35, CFG 4, Euler

### Photorealistic with Railgun
```
FC3, Futuristic_Cars cinematic lighting, (photorealistic:2), cyberpunk car, futuristic car, ground vehicle, wide body car, intrincated style car, wide wheels, offroad, railgun roof, crawler, post-apocalyptic scene
```
Settings: Steps 35, CFG 4, Euler

### Alien-Themed Asteroid Pattern
```
alien-themed asteroid pattern FC3, Futuristic_Cars cinematic lighting, (hyperrealistic:2), (cyberpunk car:1.5), futuristic car, rusted car, wide body car, intrincated style car, alien body paint, wide wheels, offroad, toolboxes on roof, post-apocalyptic scene, alien ancient city, a sleek, futuristic car with glowing red eyes cruises down the street, its engine purring softly. Its body is made of metallic material and it emits an otherworldly sound as it speeds through a chaotic cityscape. extraterrestrial, cosmic, otherworldly, mysterious, sci-fi, highly detailed
```
Settings: Steps 35, CFG 4, Euler

### Rusted Car in Ancient Alien City
```
FC3, Futuristic_Cars cinematic lighting, (hyperrealistic:2), cyberpunk car, futuristic car, rusted car, wide body car, intrincated style car, wide wheels, offroad, toolbox on root, crawler, post-apocalyptic scene, alien ancient city
```
Settings: Steps 35, CFG 4, Euler

### Desert Crawler
```
FC3, Futuristic_Cars cinematic lighting, (hyperrealistic:2), cyberpunk car, futuristic car, rusted car, wide body car, intrincated style car, wide wheels, offroad, toolbox on root, crawler, post-apocalyptic scene, dessert
```
Settings: Steps 35, CFG 4, Euler

### Night City Glowing
```
FC3, Futuristic_Cars cinematic lighting, cyberpunk car, futuristic car, ground vehicle, wide body car, intrincated style car, wide wheels, offroad, weapons on roof, crawler, post-apocalyptic scene, glowing, night city, drugstore
```
Settings: Steps 35, CFG 4, Euler, 1216x832

### VW Bulli T1 Futuristic
```
realistic photo of a Bulli ti van, wide body, fender flares, star five spoke racing wheels, racing tires, ultra high quality, sharp focus, highly detailed, <lora:Futuristic_Cars_Flux:0.8>, futuristic_cars, vehicle focus, motor vehicle
```
Settings: Steps 25, CFG 4.5, 1024x1024
Combinations: Futuristic Cars (0.8), VW Bulli T1 (1.0)

## Keywords

### Main Triggers
- `FC3, Futuristic_Cars` (required)
- `futuristic_cars` (alternate)

### Vehicle Types
- `cyberpunk car`
- `futuristic car`
- `ground vehicle`
- `wide body car`
- `rusted car`
- `crawler`
- `motor vehicle`
- `vehicle focus`

### Style
- `intrincated style car`
- `wide wheels`
- `fender flares`
- `racing wheels`
- `racing tires`

### Accessories
- `weapons on roof`
- `railgun roof`
- `toolbox on roof`
- `toolboxes on roof`

### Environment
- `offroad`
- `post-apocalyptic scene`
- `alien ancient city`
- `night city`
- `dessert` (desert)
- `drugstore`

### Mood/Theme
- `alien-themed`
- `asteroid pattern`
- `alien body paint`
- `extraterrestrial`
- `cosmic`
- `otherworldly`
- `mysterious`
- `sci-fi`
- `glowing`

### Quality
- `cinematic lighting`
- `photorealistic`
- `hyperrealistic`
- `highly detailed`
- `ultra high quality`
- `sharp focus`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 25-35 |
| **CFG** | 4-4.5 |
| **Sampler** | Euler |
| **Size** | 1024x1024 / 1216x832 |
| **Strength** | 0.8-1.0 |

## Recommended Combinations

### With Glowing Effects
```
<lora:Futuristic_Cars_Flux:1>
<lora:Glowing_Light_Particles:0.9>
```

### With Detail Enhancement
```
<lora:Futuristic_Cars_Flux:0.8>
<lora:FaeTastic_Details:0.35>
```

### With Classic Car LoRAs
```
<lora:Futuristic_Cars_Flux:0.8>
<lora:vw_bulli_t1:1>
```

## Notes

- Main trigger `FC3, Futuristic_Cars` is required
- Use `(photorealistic:2)` or `(hyperrealistic:2)` for more realistic results
- Works great with post-apocalyptic and cyberpunk themes
- Can combine with other vehicle LoRAs for futuristic versions
- Rusted/weathered cars work well for dystopian scenes
- Also trained as SDXL version with more exaggerated cyberpunk look
- Good for night city scenes with glowing elements


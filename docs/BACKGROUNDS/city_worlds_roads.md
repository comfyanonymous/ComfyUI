# City Worlds 01: Roads

[← Back to Index](INDEX.md)

## Civitai Stats

| Metric | Value |
|--------|-------|
| **Downloads** | 832 |
| **👍** | 93 |
| **Tips** | 0 |
| **Score** | ⭐ |

## Parameters

| Parameter | Value |
|-----------|-------|
| **File** | `City_Worlds_Roads_Flux.safetensors` |
| **Original filename** | `City_Worlds_01_v1_Roads_flux.safetensors` |
| **Civitai** | https://civitai.com/models/1227344 |
| **Trigger word** | `citybkg` |
| **Strength** | 1.0 |
| **Type** | BACKGROUND (Roads, Cars, Scenery) |

## Description

Part of the City Worlds series - creates city backgrounds focused on roads and cars. Not just city backgrounds - also country roads, beaches, mountains, forests, and more. Can generate various road types, vehicles, weather conditions, and scenic environments.

**Capabilities:**
- City roads with parked cars
- Highway scenes with traffic
- Mountain and hill roads
- Sea side roads with waves
- Forest roads (muddy, creepy)
- Desert and gravel roads
- Car interiors (driving POV)
- Racing scenes with motion blur
- Various weather (rain, snow, mist)

## Sample Prompts

### Rally Racing in Muddy Forest
```
citybkg, clouds, ambiance, detailed, futuristic, vegetation, ambiance, evening, clouds, very detailed, illuminated, mist, from tail, vegetation, big rocks, sinuous road, very narrow, in the forest, wet dirty muddy road, lots of wet mud with grass, creepy, depth of field, very old trees, roots, moss, pound, dirty rally car racing in the mud, wet, motion blur, solo, lots of splashing mud, smoke, jump
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Cyberpunk Seaside Mountain Road
```
citybkg, cyberpunk, day, clouds, sunny, ambiance, detailed, futuristic, vegetation, ambiance, day, clouds, very detailed, illuminated, reflections, mist, motion blur, speed, wet, rain, sunset, sun shades, mountain road, sea side road, from tail, palm trees, sea waves, rocky shore, sinuous road
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Summer Town with Parked Cars
```
citybkg, vegetation, summer, city road, low buildings, parked cars on each road sides, palm tree, town
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Country Hill Road
```
citybkg, day, clouds, sunny, ambiance, detailed, futuristic, vegetation, ambiance, day, clouds, very detailed, illuminated, mist, day, sun shades, hill road, from tail, vegetation, big rocks, sinuous road, fields, forest, dirty muddy broken road, cracked road with grass
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Sea Side Gravel Road
```
citybkg, clouds, sea side road, waves, hills, gravel road
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Bridge Over Stormy Sea
```
citybkg, scenery, sea side road, on a bridge, both sides sea, storm, high waves, old car
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Abandoned Forest Road with Snow
```
citybkg, clouds, ambiance, detailed, futuristic, vegetation, ambiance, evening, clouds, very detailed, illuminated, mist, from tail, vegetation, big rocks, sinuous road, very narrow, in the forest, wet dirty muddy road, lots of wet mud with grass, creepy, depth of field, very old trees, roots, broken abandonned graveyard, moss, pound, abandoned rusty old car full of moss, winter, snow
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Mountain Town in Snow
```
citybkg, signs, car, mountain, town, snow
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Night Highway Racing
```
citybkg, cars, racing, night, highway
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Creepy Abandoned House Road
```
citybkg, day, clouds, sunny, ambiance, detailed, futuristic, vegetation, ambiance, day, clouds, very detailed, illuminated, mist, day, sun shades, hill road, from tail, vegetation, big rocks, sinuous road, fields, forest, dirty muddy broken road, cracked road with grass, old abandonned house, creepy
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Car Interior Driving POV
```
citybkg, car interior, incar driving, vegetation, hand on the wheel, sea side road
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5

### Girl with Car (Lightning Storm)
```
citybkg, 1girl, car, in front of car, dancing, standing, lightning, tree
```
Settings: Steps 30, CFG 1, Euler, 800x1024, Distilled CFG 3.5 + ADetailer

## Keywords

### Main Trigger
- `citybkg` (required)

### Vehicles
- `car`, `cars`, `many cars`
- `vehicule focus`
- `parked cars (on each road sides)`
- `two tone paintjob`
- `traffic jam`
- `truck`, `trailer`
- `bus`, `buses`
- `old car`, `autonomous car`

### Road Types
- `road`, `road markings`
- `highway`, `tunnel`
- `mountain road`, `hill road`
- `sea side road`, `desert road`
- `gravel road`, `sinuous road`
- `broken road`, `cracked road`
- `dirty muddy road`, `wet road`

### Environment
- `forest`, `vegetation`, `fields`
- `mountain`, `mountains`, `hills`
- `palm trees`, `very old trees`
- `sea`, `waves`, `both sides sea`
- `town`, `low buildings`

### Weather/Time
- `day`, `night`, `sunset`, `evening`
- `mist`, `clouds`, `rain`
- `winter`, `snow`, `cold`
- `summer`, `sunny`

### Camera/Style
- `from tail`
- `motion blur`, `depth of field`
- `incar driving`, `car interior`
- `hand on the wheel`
- `racing`

### Urban Elements
- `signs`, `billboards`
- `lampost`, `bridge`
- `road metal barriers`
- `car park`, `parking`

## Settings

| Parameter | Value |
|-----------|-------|
| **Steps** | 30 |
| **CFG** | 1 |
| **Distilled CFG** | 3.5 |
| **Sampler** | Euler |
| **Size** | 800x1024 / 1024x1024 |
| **Strength** | 1.0 |

## Recommended Checkpoint

- **PixelWave FLUX.1-dev 03** - tested and works great

## Notes

- Main trigger `citybkg` is required
- Part of City Worlds series (cyberpunk/city themed)
- Works for both city and country roads
- Use Adetailer for faces when adding characters
- Combine with character LoRAs for scenes with people
- Great for car photography backgrounds
- Supports various weather and lighting conditions


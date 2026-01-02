# LoRA Training Guide: Volkswagen Polo 6R

Guide for creating a custom Flux LoRA to accurately render VW Polo 6R.

---

## Training Images Requirements

### Quantity
- **Minimum:** 15 images
- **Recommended:** 25-40 images
- **Maximum useful:** ~50 images (more may cause overfitting)

### Image Quality
- **Resolution:** minimum 1024x1024 (higher is better, will be resized)
- **Format:** JPG or PNG
- **Quality:** Sharp, well-lit, no blur or motion artifacts
- **File size:** At least 500KB per image (indicates good quality)

---

## Photo Categories (aim for variety)

### 1. Exterior Angles (10-15 photos)
| Angle | Description | Priority |
|-------|-------------|----------|
| Front 3/4 | Classic car photo angle, front-left or front-right | HIGH |
| Rear 3/4 | Back angle showing taillights and side | HIGH |
| Side profile | Full side view, left and right | HIGH |
| Direct front | Head-on front view | MEDIUM |
| Direct rear | Head-on back view | MEDIUM |
| High angle | Looking down at car | LOW |
| Low angle | Looking up at car | LOW |

### 2. Detail Shots (5-10 photos)
| Detail | Description |
|--------|-------------|
| Front grille/badge | VW logo, grille pattern |
| Headlights | Front light clusters |
| Taillights | Rear light design |
| Wheels/rims | Tire and rim detail |
| Side mirrors | Shape and design |
| Door handles | Distinctive features |

### 3. Environment Variety (spread across all photos)
- Parking lot
- Street/road
- Driveway
- Nature/outdoor
- Urban setting
- Different times of day (daylight preferred)

### 4. Color Variations (if possible)
Your car color will be primary, but if you can include:
- Your Polo in different lighting conditions
- Optional: photos of other Polo 6R colors from web (mark separately)

---

## Photo Tips

### DO:
- Use natural daylight when possible
- Clean the car before photos (or not, for realistic dirt/dust)
- Include full car in frame with some background
- Shoot from eye level and various heights
- Include close-ups of distinctive Polo 6R features
- Take photos with consistent focus on the car

### DON'T:
- Use heavily filtered/edited photos
- Include watermarks or text overlays
- Use extremely dark or overexposed images
- Include photos where car is partially hidden
- Use blurry or motion-blurred images
- Mix with other car models

---

## Captioning Requirements

Each image needs a text description. Format: `image_name.txt` next to `image_name.jpg`

### Caption Template:
```
a photo of a [COLOR] Volkswagen Polo 6R, [ANGLE], [LOCATION], [LIGHTING], [ADDITIONAL DETAILS]
```

### Example Captions:
```
a photo of a white Volkswagen Polo 6R, front three-quarter view, parked on street, daylight, clean exterior

a photo of a white Volkswagen Polo 6R, rear view showing taillights, parking lot, overcast lighting

a photo of a white Volkswagen Polo 6R hatchback, side profile, urban setting, golden hour lighting, alloy wheels visible

close-up of Volkswagen Polo 6R front grille and VW badge, chrome details, white car
```

### Trigger Word
Choose a unique trigger word for your LoRA:
- Suggested: `vwpolo6r` or `polo6r` or `mypolo`
- Include this word in ALL captions

**Example with trigger:**
```
a photo of a white vwpolo6r Volkswagen Polo 6R, front view, daylight
```

---

## Folder Structure

Prepare your images in this structure:
```
C:\Users\spoko\www\ai\lora_training\
└── polo6r_dataset\
    ├── 01_front_34.jpg
    ├── 01_front_34.txt
    ├── 02_rear_34.jpg
    ├── 02_rear_34.txt
    ├── 03_side_left.jpg
    ├── 03_side_left.txt
    └── ... (more pairs)
```

---

## Technical Specifications (for training)

| Parameter | Recommended Value |
|-----------|-------------------|
| **Base Model** | Flux.1 Dev |
| **LoRA Rank** | 16-32 |
| **Learning Rate** | 1e-4 to 5e-4 |
| **Steps** | 1500-3000 |
| **Batch Size** | 1-2 (12GB VRAM) |
| **Resolution** | 1024x1024 |
| **Optimizer** | AdamW8bit |

---

## Checklist Before Training

- [ ] 15-40 high-quality images collected
- [ ] Images are sharp and well-lit
- [ ] Various angles covered (front, rear, side, 3/4)
- [ ] Each image has matching .txt caption file
- [ ] Trigger word included in all captions
- [ ] Images organized in single folder
- [ ] No watermarks or text overlays
- [ ] Car is clearly visible in all images

---

## Your Polo 6R Specifications

Fill in your car details:
- **Color:** _______________
- **Year:** _______________
- **Trim/Version:** _______________
- **Notable features:** _______________
- **Wheel type:** _______________

---

## Next Steps

1. Collect and organize photos per this guide
2. Create caption files for each image
3. Notify when ready - I'll help set up training environment
4. Train LoRA (~2-4 hours)
5. Test and iterate if needed

---

## Tools Needed (will install when ready)

- **kohya_ss** or **ai-toolkit** - LoRA training
- **BLIP/Florence** - Optional auto-captioning assist

---

*Created: 2026-01-02*

from PIL import Image
import os

files = [
    'output/flux_nsfw_00003_.png',  # SHARP
    'output/flux_nsfw_00004_.png',  # BLURRY
    'output/test_minimal/img_00001_.png'  # TEST
]

for f in files:
    if os.path.exists(f):
        img = Image.open(f)
        print(f"{f}: {img.size[0]}x{img.size[1]}")

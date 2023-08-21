# Mara Huldra 2023
# SPDX-License-Identifier: MIT
'''
Palettize an image.
'''
import os

import numpy as np
from PIL import Image
import torch

from comfy.nodes.package_typing import CustomNode

PALETTES_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'palettes')
PAL_EXT = '.png'

QUANTIZE_METHODS = {
    'median_cut': Image.Quantize.MEDIANCUT,
    'max_coverage': Image.Quantize.MAXCOVERAGE,
    'fast_octree': Image.Quantize.FASTOCTREE,
}


# Determine optimal number of colors.
# FROM: astropulse/sd-palettize
#
# Use FASTOCTREE for determining the best k, as it is
# - its faster
# - it does a better job fitting the image to lower color counts than the other options
# Max converge is best for reducing an image's colors more accurately, but
# since for best k we only care about the best number of colors, a faster more
# predictable method is better.
# (Astropulse, 2023-06-05)
def determine_best_k(image, max_k, quantize_method=Image.Quantize.FASTOCTREE):
    # Convert the image to RGB mode
    image = image.convert("RGB")

    # Prepare arrays for distortion calculation
    pixels = np.array(image)
    pixel_indices = np.reshape(pixels, (-1, 3))

    # Calculate distortion for different values of k
    distortions = []
    for k in range(1, max_k + 1):
        quantized_image = image.quantize(colors=k, method=quantize_method, kmeans=k, dither=0)
        centroids = np.array(quantized_image.getpalette()[:k * 3]).reshape(-1, 3)

        # Calculate distortions
        distances = np.linalg.norm(pixel_indices[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortions.append(np.sum(min_distances ** 2))

    # Calculate the rate of change of distortions
    rate_of_change = np.diff(distortions) / np.array(distortions[:-1])

    # Find the elbow point (best k value)
    if len(rate_of_change) == 0:
        best_k = 2
    else:
        elbow_index = np.argmax(rate_of_change) + 1
        best_k = elbow_index + 2

    return best_k


palette_warned = False


def list_palettes():
    global palette_warned
    palettes = []
    try:
        for filename in os.listdir(PALETTES_PATH):
            if filename.endswith(PAL_EXT):
                palettes.append(filename[0:-len(PAL_EXT)])
    except FileNotFoundError:
        pass
    if not palettes and not palette_warned:
        palette_warned = True
        print(
            "ImagePalettize warning: no fixed palettes found. You can put these in the palettes/ directory below the ComfyUI root.")
    return palettes


def get_image_colors(pal_img):
    palette = []
    pal_img = pal_img.convert('RGB')
    for i in pal_img.getcolors(16777216):
        palette.append(i[1][0])
        palette.append(i[1][1])
        palette.append(i[1][2])
    return palette


def load_palette(name):
    return get_image_colors(Image.open(os.path.join(PALETTES_PATH, name + PAL_EXT)))


class ImagePalettize(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette": (["auto_best_k", "auto_fixed_k"] + list_palettes(), {
                    "default": "auto_best_k",
                }),
                "max_k": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                }),
                "method": (list(QUANTIZE_METHODS.keys()), {
                    "default": "max_coverage",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "palettize"

    CATEGORY = "image/postprocessing"

    def palettize(self, image, palette, max_k, method):
        k = None
        pal_img = None
        if palette not in {'auto_best_k', 'auto_fixed_k'}:
            pal_entries = load_palette(palette)
            k = len(pal_entries) // 3
            pal_img = Image.new('P', (1, 1))  # image size doesn't matter it only holds the palette
            pal_img.putpalette(pal_entries)

        results = []

        for i in image:
            i = 255. * i.cpu().numpy()
            i = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if palette == 'auto_best_k':
                k = determine_best_k(i, max_k)
                print(f'Auto number of colors: {k}')
            elif palette == 'auto_fixed_k':
                k = max_k

            i = i.quantize(colors=k, method=QUANTIZE_METHODS[method], kmeans=k, dither=0, palette=pal_img)
            i = i.convert('RGB')

            results.append(np.array(i))

        result = np.array(results).astype(np.float32) / 255.0
        return (torch.from_numpy(result),)


NODE_CLASS_MAPPINGS = {
    "ImagePalettize": ImagePalettize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImagePalettize": "ImagePalettize"
}

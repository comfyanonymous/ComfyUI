"""
Adapted from https://github.com/WASasquatch/was-node-suite-comfyui/blob/main/LICENSE
MIT License

Copyright (c) 2023 Jordan Thompson (WASasquatch)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import json

from PIL import Image, ImageDraw, ImageFilter

from comfy.component_model.tensor_types import MaskBatch
from comfy.nodes.package_typing import CustomNode
from comfy.utils import pil2tensor


def gradient(size, mode='horizontal', colors=None, tolerance=0):
    if isinstance(colors, str):
        colors = json.loads(colors)

    if colors is None:
        colors = {0: [255, 0, 0], 50: [0, 255, 0], 100: [0, 0, 255]}

    colors = {int(k): [int(c) for c in v] for k, v in colors.items()}

    colors[0] = colors[min(colors.keys())]
    colors[255] = colors[max(colors.keys())]

    img = Image.new('RGB', size, color=(0, 0, 0))

    color_stop_positions = sorted(colors.keys())
    color_stop_count = len(color_stop_positions)
    spectrum = []
    for i in range(256):
        start_pos = max(p for p in color_stop_positions if p <= i)
        end_pos = min(p for p in color_stop_positions if p >= i)
        start = colors[start_pos]
        end = colors[end_pos]

        if start_pos == end_pos:
            factor = 0
        else:
            factor = (i - start_pos) / (end_pos - start_pos)

        r = round(start[0] + (end[0] - start[0]) * factor)
        g = round(start[1] + (end[1] - start[1]) * factor)
        b = round(start[2] + (end[2] - start[2]) * factor)
        spectrum.append((r, g, b))

    draw = ImageDraw.Draw(img)
    if mode == 'horizontal':
        for x in range(size[0]):
            pos = int(x * 100 / (size[0] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([round(c / tolerance) * tolerance for c in color])
            draw.line((x, 0, x, size[1]), fill=color)
    elif mode == 'vertical':
        for y in range(size[1]):
            pos = int(y * 100 / (size[1] - 1))
            color = spectrum[pos]
            if tolerance > 0:
                color = tuple([round(c / tolerance) * tolerance for c in color])
            draw.line((0, y, size[0], y), fill=color)

    blur = 1.5
    if size[0] > 512 or size[1] > 512:
        multiplier = max(size[0], size[1]) / 512
        if multiplier < 1.5:
            multiplier = 1.5
        blur = blur * multiplier

    img = img.filter(ImageFilter.GaussianBlur(radius=blur))

    return img


class ImageGenerateGradient(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        gradient_stops = '''0:255,0,0
25:255,255,255
50:0,255,0
75:0,0,255'''
        return {
            "required": {
                "width": ("INT", {"default": 512, "max": 4096, "min": 64, "step": 1}),
                "height": ("INT", {"default": 512, "max": 4096, "min": 64, "step": 1}),
                "direction": (["horizontal", "vertical"],),
                "tolerance": ("INT", {"default": 0, "max": 255, "min": 0, "step": 1}),
                "gradient_stops": ("STRING", {"default": gradient_stops, "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_gradient"

    CATEGORY = "image/generate"

    def image_gradient(self, gradient_stops, width=512, height=512, direction='horizontal', tolerance=0) -> tuple[MaskBatch]:
        import io

        colors_dict = {}
        stops = io.StringIO(gradient_stops.strip().replace(' ', ''))
        for stop in stops:
            parts = stop.split(':')
            colors = parts[1].replace('\n', '').split(',')
            colors_dict[parts[0].replace('\n', '')] = colors

        image = gradient((width, height), direction, colors_dict, tolerance)

        return (pil2tensor(image),)


NODE_CLASS_MAPPINGS = {
    "ImageGenerateGradient": ImageGenerateGradient,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGenerateGradient": "Image Generate Gradient",
}

import logging

import numpy as np
import torch
import vtracer
import logging
from PIL import Image

from comfy.nodes.package_typing import CustomNode
from comfy.utils import tensor2pil

logger = logging.getLogger(__name__)


def RGB2RGBA(image: Image, mask: Image) -> Image:
    (R, G, B) = image.convert('RGB').split()
    return Image.merge('RGBA', (R, G, B, mask.convert('L')))


class ImageToSVG(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormode": (["color", "binary"], {"default": "color"}),
                "hierarchical": (["stacked", "cutout"], {"default": "stacked"}),
                "mode": (["spline", "polygon", "none"], {"default": "spline"}),
                "filter_speckle": ("INT", {"default": 4, "min": 0, "max": 100}),
                "color_precision": ("INT", {"default": 6, "min": 0, "max": 10}),
                "layer_difference": ("INT", {"default": 16, "min": 0, "max": 256}),
                "corner_threshold": ("INT", {"default": 60, "min": 0, "max": 180}),
                "length_threshold": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 70}),
                "splice_threshold": ("INT", {"default": 45, "min": 0, "max": 180}),
                "path_precision": ("INT", {"default": 3, "min": 0, "max": 10}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SVG",)
    FUNCTION = "convert_to_svg"

    CATEGORY = "image/svg"

    def convert_to_svg(self, image, colormode, hierarchical, mode, filter_speckle, color_precision, layer_difference, corner_threshold, length_threshold, max_iterations, splice_threshold, path_precision):
        svg_strings = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = tensor2pil(i)

            if _image.mode != 'RGBA':
                alpha = Image.new('L', _image.size, 255)
                _image.putalpha(alpha)

            pixels = list(_image.getdata())

            size = _image.size

            svg_str = vtracer.convert_pixels_to_svg(
                pixels,
                size=size,
                colormode=colormode,
                hierarchical=hierarchical,
                mode=mode,
                filter_speckle=filter_speckle,
                color_precision=color_precision,
                layer_difference=layer_difference,
                corner_threshold=corner_threshold,
                length_threshold=length_threshold,
                max_iterations=max_iterations,
                splice_threshold=splice_threshold,
                path_precision=path_precision
            )

            svg_strings.append(svg_str)

        return (svg_strings,)


class SVGToImage(CustomNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg": ("STRING", {"forceInput": True}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_image"

    CATEGORY = "image/svg"

    def clean_svg_string(self, svg_string):
        svg_start = svg_string.find("<svg")
        if svg_start == -1:
            raise ValueError("No <svg> tag found in the input string")
        return svg_string[svg_start:]

    def convert_to_image(self, svg, scale):
        try:
            import skia
        except (ImportError, ModuleNotFoundError) as exc_info:
            logger.error("failed to import skia", exc_info=exc_info)
            return (torch.zeros((0, 1, 1, 3)),)
        raster_images = []

        for i, svg_string in enumerate(svg):
            stream = None
            try:
                cleaned_svg = self.clean_svg_string(svg_string)

                stream = skia.MemoryStream(cleaned_svg.encode('utf-8'), True)  # pylint: disable=c-extension-no-member
                svg_dom = skia.SVGDOM.MakeFromStream(stream)  # pylint: disable=c-extension-no-member

                if svg_dom is None:
                    raise ValueError(f"Failed to parse SVG content for image {i}")

                svg_width = svg_dom.containerSize().width()
                svg_height = svg_dom.containerSize().height()

                width = int(svg_width * scale)
                height = int(svg_height * scale)

                surface = skia.Surface(width, height)  # pylint: disable=c-extension-no-member
                with surface as canvas:
                    canvas.clear(skia.ColorTRANSPARENT)  # pylint: disable=c-extension-no-member

                    canvas.scale(scale, scale)
                    svg_dom.render(canvas)

                image = surface.makeImageSnapshot()
                img_array = np.array(image.toarray())

                # BGR to RGB
                img_array = img_array[..., :3][:, :, ::-1]
                img_tensor = torch.from_numpy(img_array.astype(np.float32) / 255.0)

                raster_images.append(img_tensor)
            except Exception as exc_info:
                logging.error("Error when trying to encode SVG, returning error rectangle instead", exc_info=exc_info)
                # Create a small red image to indicate error
                error_img = np.full((64, 64, 4), [255, 0, 0, 255], dtype=np.uint8)
                error_tensor = torch.from_numpy(error_img.astype(np.float32) / 255.0)
                raster_images.append(error_tensor)
            finally:
                if stream is not None:
                    del stream

        if not raster_images:
            raise ValueError("No valid images were generated from the input SVGs")

        batch = torch.stack(raster_images)

        return (batch,)


NODE_CLASS_MAPPINGS = {
    "ImageToSVG": ImageToSVG,
    "SVGToImage": SVGToImage,
}

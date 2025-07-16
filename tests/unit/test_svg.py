import pytest
import torch

from comfy_extras.nodes.nodes_svg import ImageToSVG, SVGToImage

SKIP = False
try:
    import skia
except (ImportError, ModuleNotFoundError):
    SKIP = True


@pytest.fixture
def sample_image():
    return torch.rand((1, 64, 64, 3))


@pytest.mark.skipif(SKIP, reason="skia import error")
def test_image_to_svg(sample_image):
    image_to_svg_node = ImageToSVG()

    svg_result, = image_to_svg_node.convert_to_svg(sample_image, "color", "stacked", "spline", 4, 6, 16, 60, 4.0, 10, 45, 3)
    assert isinstance(svg_result[0], str)
    assert svg_result[0].startswith('<?xml')

    svg_result, = image_to_svg_node.convert_to_svg(sample_image, "binary", "cutout", "polygon", 2, 8, 32, 90, 2.0, 5, 30, 5)
    assert isinstance(svg_result[0], str)
    assert svg_result[0].startswith('<?xml')


@pytest.mark.skipif(SKIP, reason="skia import error")
def test_svg_to_image():
    svg_to_image_node = SVGToImage()

    test_svg = '''<?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
        <rect width="100" height="100" fill="red" />
    </svg>'''

    image_result, = svg_to_image_node.convert_to_image([test_svg], 1.0)
    assert isinstance(image_result, torch.Tensor)
    assert image_result.shape == (1, 100, 100, 3)

    image_result, = svg_to_image_node.convert_to_image([test_svg], 2.0)
    assert isinstance(image_result, torch.Tensor)
    assert image_result.shape == (1, 200, 200, 3)

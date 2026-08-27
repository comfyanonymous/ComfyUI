import pytest

from comfy_extras.nodes_minimax_h3 import downscale_to_area


@pytest.mark.parametrize(
    ("width", "height", "max_pixels", "expected"),
    [
        (1344, 768, 960 * 544, (960, 544)),
        (768, 1344, 960 * 544, (544, 960)),
        (640, 352, 960 * 544, (640, 352)),
        (1000, 700, 1000 * 700, (992, 672)),
        (100, 100, 48 * 48, (32, 32)),
        (1024, 256, 32 * 32, (32, 32)),
    ],
)
def test_downscale_to_area(width, height, max_pixels, expected):
    actual = downscale_to_area(width, height, max_pixels)
    assert actual == expected
    assert actual[0] <= width
    assert actual[1] <= height
    assert actual[0] * actual[1] <= max_pixels

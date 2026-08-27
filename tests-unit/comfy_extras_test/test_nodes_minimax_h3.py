import pytest

from comfy_extras.nodes_minimax_h3 import downscale_to_area


@pytest.mark.parametrize(
    ("width", "height", "max_pixels", "expected"),
    [
        (1344, 768, 960 * 544, (960, 544)),
        (768, 1344, 960 * 544, (544, 960)),
        (640, 352, 960 * 544, (640, 352)),
    ],
)
def test_downscale_to_area(width, height, max_pixels, expected):
    assert downscale_to_area(width, height, max_pixels) == expected

import sys

import numpy as np
import pytest

import comfy.options


# The drawing helper does not need an accelerator, but importing this module
# initializes ComfyUI's device management on CPU-only test environments.
comfy.options.args_parsing = True
pytest_argv = sys.argv
sys.argv = [sys.argv[0], "--cpu"]

from comfy_extras.nodes_sdpose import KeypointDraw

sys.argv = pytest_argv
comfy.options.args_parsing = False


@pytest.mark.parametrize("index", [0, 10, 18, 24, 92, 113])
def test_draw_wholebody_skips_nonfinite_keypoints(index):
    keypoints = np.zeros((134, 2), dtype=np.float32)
    keypoints[index] = np.nan
    scores = np.ones(134, dtype=np.float32)
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)

    result = KeypointDraw().draw_wholebody_keypoints(canvas, keypoints, scores)

    assert result is canvas

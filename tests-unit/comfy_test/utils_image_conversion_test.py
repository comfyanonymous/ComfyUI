import warnings

import numpy as np
import torch

import comfy.utils


def test_image_to_uint8_sanitizes_nonfinite_values_without_runtime_warning():
    image = torch.tensor(
        [
            [
                [float("nan"), float("inf"), -float("inf")],
                [-1.0, 0.5, 2.0],
            ]
        ],
        dtype=torch.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = comfy.utils.image_to_uint8(image)

    assert result.dtype == np.uint8
    assert result.tolist() == [[[0, 255, 0], [0, 127, 255]]]


def test_image_to_uint8_does_not_modify_source_tensor():
    image = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.5])

    comfy.utils.image_to_uint8(image)

    assert torch.isnan(image[0])
    assert torch.isinf(image[1])
    assert torch.isinf(image[2])
    assert image[3] == 0.5

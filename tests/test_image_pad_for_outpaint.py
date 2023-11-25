import pytest
import torch
from nodes import ImagePadForOutpaint


def test_image_pad_for_outpaint():
    # Arrange
    input_image = [[1, 2], [3, 4]]
    expected_expanded_image = [[1, 2],
                               [3, 4]]

    # Convert the list to a PyTorch tensor and add two dimensions
    input_image_tensor = torch.tensor(input_image).unsqueeze(0).unsqueeze(0)

    # Act
    image_pad = ImagePadForOutpaint()
    result = image_pad.expand_image(input_image_tensor, 0, 0, 0, 0, 0)

    # Assert
    if isinstance(result, tuple) and len(result) > 0:
        result = result[0]
    assert result.squeeze().tolist() == expected_expanded_image

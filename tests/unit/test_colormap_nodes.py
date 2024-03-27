import pytest
import torch
from comfy_extras.nodes.nodes_apply_color_map import ImageApplyColorMap


@pytest.fixture
def input_image():
    # Create a 1x1x2x1 tensor representing an image with absolute distances of 1.3 meters and 300 meters
    return torch.tensor([[[[1.3], [300.0]]]], dtype=torch.float32)


def test_apply_colormap_grayscale(input_image):
    node = ImageApplyColorMap()
    colored_image, = node.execute(image=input_image, colormap="Grayscale", min_depth=1.3, max_depth=300.0)

    assert colored_image.shape == (1, 1, 2, 3)
    assert colored_image.dtype == torch.float32
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0, 0.0, 0.0]))


def test_apply_colormap_inferno(input_image):
    node = ImageApplyColorMap()
    colored_image, = node.execute(image=input_image, colormap="COLORMAP_INFERNO", min_depth=1.3, max_depth=300.0)

    assert colored_image.shape == (1, 1, 2, 3)
    assert colored_image.dtype == torch.float32
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([0.9882, 1.000, 0.6431]), atol=1e-4)
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0000, 0.0000, 0.0157]), atol=1e-4)


def test_apply_colormap_clipping(input_image):
    node = ImageApplyColorMap()

    colored_image, = node.execute(image=input_image, colormap="COLORMAP_INFERNO", clip_min=False, clip_max=False, min_depth=1.3, max_depth=300.0)
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([0.0, 0.0, 0.0157]), atol=1e-4)
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0, 0.0, 0.0157]), atol=1e-4)

    colored_image, = node.execute(image=input_image, colormap="COLORMAP_INFERNO", clip_min=True, clip_max=False, min_depth=1.3, max_depth=300.0)
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([0.9882, 1.0000, 0.6431]), atol=1e-4)
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0000, 0.0000, 0.0157]), atol=1e-4)

    colored_image, = node.execute(image=input_image, colormap="COLORMAP_INFERNO", clip_min=False, clip_max=True, min_depth=1.3, max_depth=200.0)
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([0.0, 0.0, 0.0157]), atol=1e-4)
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0, 0.0, 0.0157]), atol=1e-4)

    colored_image, = node.execute(image=input_image, colormap="COLORMAP_INFERNO", clip_min=True, clip_max=True, min_depth=1.3, max_depth=200.0)
    assert torch.allclose(colored_image[0, 0, 0], torch.tensor([0.9882, 1.0000, 0.6431]), atol=1e-4)
    assert torch.allclose(colored_image[0, 0, 1], torch.tensor([0.0000, 0.0000, 0.0157]), atol=1e-4)

import pytest
import torch
import numpy as np

# Assuming the node definitions are in a file named 'inpaint_nodes.py'
from comfy_extras.nodes.nodes_inpainting import CropAndFitInpaintToDiffusionSize, CompositeCroppedAndFittedInpaintResult, parse_margin

def create_circle_mask(height, width, center_y, center_x, radius):
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    distance = torch.sqrt((Y - center_y)**2 + (X - center_x)**2)
    return (distance <= radius).float().unsqueeze(0)

@pytest.fixture
def sample_image() -> torch.Tensor:
    gradient = torch.linspace(0, 1, 256).view(1, -1, 1, 1)
    return gradient.expand(1, 256, 256, 3)

@pytest.fixture
def rect_mask() -> torch.Tensor:
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:150, 80:180] = 1.0
    return mask

@pytest.fixture
def circle_mask() -> torch.Tensor:
    return create_circle_mask(256, 256, center_y=128, center_x=128, radius=50)

def test_crop_and_fit_overflow(sample_image, rect_mask):
    """Tests the overflow logic by placing the mask at an edge."""
    node = CropAndFitInpaintToDiffusionSize()
    edge_mask = torch.zeros_like(rect_mask)
    edge_mask[:, :20, :50] = 1.0
    _, _, ctx_no_overflow = node.crop_and_fit(sample_image, edge_mask, "SD1.5", "30", overflow=False)
    assert ctx_no_overflow == (0, 0, 80, 50)
    img, _, ctx_overflow = node.crop_and_fit(sample_image, edge_mask, "SD1.5", "30", overflow=True)
    assert ctx_overflow == (-30, -30, 110, 80)
    assert torch.allclose(img[0, 5, 5, :], torch.tensor([0.5, 0.5, 0.5]), atol=1e-3)

@pytest.mark.parametrize("mask_fixture, margin, overflow", [
    ("rect_mask", "16", False),
    ("circle_mask", "32", False),
    ("rect_mask", "64", True),
    ("circle_mask", "0", False),
])
def test_end_to_end_composition(request, sample_image, mask_fixture, margin, overflow):
    """Performs a full round-trip test of both nodes."""
    mask = request.getfixturevalue(mask_fixture)
    crop_node = CropAndFitInpaintToDiffusionSize()
    composite_node = CompositeCroppedAndFittedInpaintResult()

    # The resized mask from the first node is not needed for compositing.
    cropped_img, _, context = crop_node.crop_and_fit(sample_image, mask, "SD1.5", margin, overflow)

    h, w = cropped_img.shape[1:3]
    blue_color = torch.tensor([0.1, 0.2, 0.9]).view(1, 1, 1, 3)
    inpainted_sim = blue_color.expand(1, h, w, 3)

    # FIX: Pass the original, high-resolution mask as `source_mask`.
    final_image, = composite_node.composite_result(
        source_image=sample_image,
        source_mask=mask,
        inpainted_image=inpainted_sim,
        composite_context=context
    )

    assert final_image.shape == sample_image.shape

    bool_mask = mask.squeeze(0).bool()
    assert torch.allclose(final_image[0][bool_mask], blue_color.squeeze(), atol=1e-2)
    assert torch.allclose(final_image[0][~bool_mask], sample_image[0][~bool_mask], atol=1e-2)
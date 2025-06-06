import pytest
import torch
import numpy as np

# Assuming the node definitions are in a file named 'inpaint_nodes.py'
from comfy_extras.nodes.nodes_inpainting import CropAndFitInpaintToDiffusionSize, CompositeCroppedAndFittedInpaintResult, parse_margin


# Helper to create a circular mask
def create_circle_mask(height, width, center_y, center_x, radius):
    """Creates a boolean mask with a filled circle."""
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    distance = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    mask = (distance <= radius).float()
    return mask.unsqueeze(0)  # Add batch dimension


@pytest.fixture
def sample_image() -> torch.Tensor:
    """A 256x256 image with a vertical gradient."""
    gradient = torch.linspace(0, 1, 256).view(1, -1, 1, 1)
    image = gradient.expand(1, 256, 256, 3)  # (B, H, W, C)
    return image


@pytest.fixture
def rect_mask() -> torch.Tensor:
    """A rectangular mask in the center of a 256x256 image."""
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:150, 80:180] = 1.0
    return mask


@pytest.fixture
def circle_mask() -> torch.Tensor:
    """A circular mask in a 256x256 image."""
    return create_circle_mask(256, 256, center_y=128, center_x=128, radius=50)


def test_parse_margin():
    """Tests the margin parsing utility function."""
    assert parse_margin("10") == (10, 10, 10, 10)
    assert parse_margin(" 10 20 ") == (10, 20, 10, 20)
    assert parse_margin("10 20 30") == (10, 20, 30, 20)
    assert parse_margin("10 20 30 40") == (10, 20, 30, 40)
    with pytest.raises(ValueError):
        parse_margin("10 20 30 40 50")
    with pytest.raises(ValueError):
        parse_margin("not a number")


def test_crop_and_fit_basic(sample_image, rect_mask):
    """Tests the basic functionality of the cropping and fitting node."""
    node = CropAndFitInpaintToDiffusionSize()

    # Using SD1.5 resolutions for predictability in tests
    img, msk, ctx = node.crop_and_fit(sample_image, rect_mask, resolutions="SD1.5", margin="20", overflow=False)

    # Check output shapes
    assert img.shape[0] == 1 and img.shape[3] == 3
    assert msk.shape[0] == 1
    # Check if resized to a valid SD1.5 resolution
    assert (img.shape[2], img.shape[1]) in [(512, 512), (768, 512), (512, 768)]
    assert img.shape[1:3] == msk.shape[1:3]

    # Check context
    # Original mask bounds: y(100, 149), x(80, 179)
    # With margin 20: y(80, 169), x(60, 199)
    # context is (x, y, width, height)
    expected_x = 80 - 20
    expected_y = 100 - 20
    expected_width = (180 - 80) + 2 * 20
    expected_height = (150 - 100) + 2 * 20

    assert ctx == (expected_x, expected_y, expected_width, expected_height)


def test_crop_and_fit_overflow(sample_image, rect_mask):
    """Tests the overflow logic by placing the mask at an edge."""
    node = CropAndFitInpaintToDiffusionSize()
    edge_mask = torch.zeros_like(rect_mask)
    edge_mask[:, :20, :50] = 1.0  # Mask at the top-left corner

    # Test with overflow disabled (should clamp)
    _, _, ctx_no_overflow = node.crop_and_fit(sample_image, edge_mask, "SD1.5", "30", overflow=False)
    assert ctx_no_overflow == (0, 0, 50 + 30, 20 + 30)

    # Test with overflow enabled
    img, msk, ctx_overflow = node.crop_and_fit(sample_image, edge_mask, "SD1.5", "30", overflow=True)
    # Context should have negative coordinates
    # Original bounds: y(0, 19), x(0, 49)
    # Margin 30: y(-30, 49), x(-30, 79)
    assert ctx_overflow == (-30, -30, (50 - 0) + 60, (20 - 0) + 60)

    # Check that padded area is gray
    # The original image was placed inside a larger gray canvas.
    # We check a pixel that should be in the padded gray area of the *cropped* image.
    # The crop starts at y=-30, x=-30 relative to original image.
    # So, pixel (5,5) in the cropped image corresponds to (-25, -25) which is padding.
    assert torch.allclose(img[0, 5, 5, :], torch.tensor([0.5, 0.5, 0.5]))

    # Check that original image content is still there
    # Pixel (40, 40) in cropped image corresponds to (10, 10) in original image
    assert torch.allclose(img[0, 40, 40, :], sample_image[0, 10, 10, :])


def test_empty_mask_raises_error(sample_image):
    """Tests that an empty mask correctly raises a ValueError."""
    node = CropAndFitInpaintToDiffusionSize()
    empty_mask = torch.zeros(1, 256, 256)
    with pytest.raises(ValueError, match="Mask is empty"):
        node.crop_and_fit(sample_image, empty_mask, "SD1.5", "10", False)


@pytest.mark.parametrize("mask_fixture, margin, overflow", [
    ("rect_mask", "16", False),
    ("circle_mask", "32", False),
    ("rect_mask", "64", True),  # margin forces overflow
    ("circle_mask", "0", False),
])
def test_end_to_end_composition(request, sample_image, mask_fixture, margin, overflow):
    """Performs a full round-trip test of both nodes."""
    mask = request.getfixturevalue(mask_fixture)

    # --- 1. Crop and Fit ---
    crop_node = CropAndFitInpaintToDiffusionSize()
    cropped_img, cropped_mask, context = crop_node.crop_and_fit(
        sample_image, mask, "SD1.5", margin, overflow
    )

    # --- 2. Simulate Inpainting ---
    # Create a solid blue image as the "inpainted" result
    h, w = cropped_img.shape[1:3]
    blue_color = torch.tensor([0.1, 0.2, 0.9]).view(1, 1, 1, 3)
    inpainted_sim = blue_color.expand(1, h, w, 3)
    # The inpainted_mask is the mask output from the first node
    inpainted_mask = cropped_mask

    # --- 3. Composite Result ---
    composite_node = CompositeCroppedAndFittedInpaintResult()
    final_image, = composite_node.composite_result(
        source_image=sample_image,
        inpainted_image=inpainted_sim,
        inpainted_mask=inpainted_mask,
        composite_context=context
    )

    # --- 4. Verify Result ---
    assert final_image.shape == sample_image.shape

    # Create a boolean version of the original mask for easy indexing
    bool_mask = mask.squeeze(0).bool()  # H, W

    # Area *inside* the mask should be blue
    masked_area_in_final = final_image[0][bool_mask]
    assert torch.allclose(masked_area_in_final, blue_color.squeeze(), atol=1e-2)

    # Area *outside* the mask should be unchanged from the original
    unmasked_area_in_final = final_image[0][~bool_mask]
    unmasked_area_in_original = sample_image[0][~bool_mask]
    assert torch.allclose(unmasked_area_in_final, unmasked_area_in_original, atol=1e-2)

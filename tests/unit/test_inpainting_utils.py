import pytest
import torch

from comfy_extras.nodes.nodes_inpainting import CropAndFitInpaintToDiffusionSize, CompositeCroppedAndFittedInpaintResult


def create_circle_mask(height, width, center_y, center_x, radius):
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    distance = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    return (distance >= radius).float().unsqueeze(0)


@pytest.fixture
def sample_image() -> torch.Tensor:
    gradient = torch.linspace(0, 1, 256).view(1, -1, 1, 1)
    return gradient.expand(1, 256, 256, 3)


@pytest.fixture
def image_1024() -> torch.Tensor:
    gradient = torch.linspace(0, 1, 1024).view(1, -1, 1, 1)
    return gradient.expand(1, 1024, 1024, 3)


@pytest.fixture
def rect_mask() -> torch.Tensor:
    mask = torch.ones(1, 256, 256)
    mask[:, 100:150, 80:180] = 0.0
    return mask


@pytest.fixture
def circle_mask() -> torch.Tensor:
    return create_circle_mask(256, 256, center_y=128, center_x=128, radius=50)


def test_crop_and_fit_edge_clamp(sample_image):
    node = CropAndFitInpaintToDiffusionSize()
    edge_mask = torch.zeros(1, 256, 256)
    edge_mask[:, :20, :50] = 1.0

    _, _, context = node.crop_and_fit(sample_image, edge_mask, "SD1.5", "30")

    target_aspect_ratio = 1.0  # For SD1.5, the only valid resolution is 512x512
    actual_aspect_ratio = context.width / context.height
    assert abs(actual_aspect_ratio - target_aspect_ratio) < 1e-4


@pytest.mark.parametrize("mask_fixture, margin", [
    ("rect_mask", "16"),
    ("circle_mask", "32"),
    ("circle_mask", "0"),
])
def test_end_to_end_composition(request, sample_image, mask_fixture, margin):
    mask = request.getfixturevalue(mask_fixture)
    crop_node = CropAndFitInpaintToDiffusionSize()
    composite_node = CompositeCroppedAndFittedInpaintResult()

    cropped_img, _, context = crop_node.crop_and_fit(sample_image, mask, "SD1.5", margin)

    h, w = cropped_img.shape[1:3]
    blue_color = torch.tensor([0.1, 0.2, 0.9]).view(1, 1, 1, 3)
    inpainted_sim = blue_color.expand(1, h, w, 3)

    final_image, = composite_node.composite_result(
        source_image=sample_image,
        source_mask=mask,
        inpainted_image=inpainted_sim,
        composite_context=context
    )

    assert final_image.shape == sample_image.shape

    bool_mask = mask.squeeze(0).bool()

    assert torch.allclose(final_image[0][bool_mask], blue_color.squeeze(), atol=1e-2)
    assert torch.allclose(final_image[0][~bool_mask], sample_image[0][~bool_mask])


def test_wide_ideogram_composite(image_1024):
    """Tests the wide margin scenario. The node logic correctly chooses 1536x512."""
    source_image = image_1024
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 900:932, 950:982] = 1.0

    crop_node = CropAndFitInpaintToDiffusionSize()
    composite_node = CompositeCroppedAndFittedInpaintResult()

    margin = "64 64 64 400"

    cropped_img, _, context = crop_node.crop_and_fit(source_image, mask, "Ideogram", margin)
    assert cropped_img.shape[1:3] == (512, 1536)

    green_color = torch.tensor([0.1, 0.9, 0.2]).view(1, 1, 1, 3)
    inpainted_sim = green_color.expand(1, 512, 1536, 3)

    final_image, = composite_node.composite_result(
        source_image=source_image,
        source_mask=mask,
        inpainted_image=inpainted_sim,
        composite_context=context
    )

    assert final_image.shape == source_image.shape

    bool_mask = mask.squeeze(0).bool()

    final_pixels = final_image[0][bool_mask]
    assert torch.all(final_pixels[:, 1] > final_pixels[:, 0])
    assert torch.all(final_pixels[:, 1] > final_pixels[:, 2])

    assert torch.allclose(final_image[0, 916, 940, :], source_image[0, 916, 940, :])

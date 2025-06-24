import pytest
import torch

from comfy_extras.nodes.nodes_inpainting import CropAndFitInpaintToDiffusionSize, \
    CompositeCroppedAndFittedInpaintResult, CompositeContext

TEST_SCENARIOS = [
    # A standard, centered case with no complex adjustments.
    pytest.param(
        dict(
            test_id="standard_sd15_center",
            mask_rect=(400, 400, 200, 200),  # y, x, h, w
            margin="64",
            resolutions="SD1.5",
            expected_cropped_shape=(512, 512),
            expected_context=CompositeContext(x=336, y=336, width=328, height=328)
        ),
        id="standard_sd15_center"
    ),
    # The user-described wide-margin case.
    pytest.param(
        dict(
            test_id="wide_ideogram_right_edge",
            mask_rect=(900, 950, 32, 32),
            margin="64 64 64 400",
            resolutions="Ideogram",
            expected_cropped_shape=(512, 1536),  # Should select 1536x512 (AR=3.0)
            expected_context=CompositeContext(x=544, y=836, width=480, height=160)
        ),
        id="wide_ideogram_right_edge"
    ),
    # A new test for a tall mask, forcing a ~1:3 aspect ratio.
    pytest.param(
        dict(
            test_id="tall_ideogram_left_edge",
            mask_rect=(200, 20, 200, 50),
            margin="100",
            resolutions="Ideogram",
            expected_cropped_shape=(1536, 640),  # Should select 640x1536 (AR=0.416)
            expected_context=CompositeContext(x=0, y=96, width=170, height=408)
        ),
        id="tall_ideogram_left_edge"
    ),
    # A test where the covering rectangle must be shifted to stay in bounds.
    pytest.param(
        dict(
            test_id="shift_to_fit",
            mask_rect=(10, 10, 150, 50),
            margin="40",
            resolutions="Ideogram",
            expected_cropped_shape=(1408, 704),  # AR is exactly 0.5
            expected_context=CompositeContext(x=0, y=0, width=100, height=200)
        ),
        id="shift_to_fit"
    )
]


def create_circle_mask(height, width, center_y, center_x, radius):
    Y, X = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    distance = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
    return (distance < radius).float().unsqueeze(0)


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
    mask = torch.zeros(1, 256, 256)
    mask[:, 100:150, 80:180] = 1.0
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

    cropped_img, cropped_mask, context = crop_node.crop_and_fit(sample_image, mask, "SD1.5", margin)

    h, w = cropped_img.shape[1:3]
    blue_color = torch.tensor([0.1, 0.2, 0.9]).view(1, 1, 1, 3)
    inpainted_sim = blue_color.expand(1, h, w, 3)

    # Inpaint the cropped region with the new color
    inpainted_cropped = cropped_img * (1 - cropped_mask.unsqueeze(-1)) + inpainted_sim * cropped_mask.unsqueeze(-1)

    final_image, = composite_node.composite_result(
        source_image=sample_image,
        source_mask=mask,
        inpainted_image=inpainted_cropped,
        composite_context=context
    )

    assert final_image.shape == sample_image.shape

    bool_mask = mask.squeeze(0).bool()

    # The area inside the mask should be blue
    assert torch.allclose(final_image[0][bool_mask], blue_color.squeeze(), atol=1e-2)
    # The area outside the mask should be unchanged
    assert torch.allclose(final_image[0][~bool_mask], sample_image[0][~bool_mask], atol=1e-2)


def test_wide_ideogram_composite(image_1024):
    """Tests the wide margin scenario. The node logic correctly chooses 1536x512."""
    source_image = image_1024
    mask = torch.zeros(1, 1024, 1024)
    mask[:, 900:932, 950:982] = 1.0

    crop_node = CropAndFitInpaintToDiffusionSize()
    composite_node = CompositeCroppedAndFittedInpaintResult()

    margin = "64 64 64 400"

    cropped_img, cropped_mask, context = crop_node.crop_and_fit(source_image, mask, "Ideogram", margin)
    assert cropped_img.shape[1:3] == (512, 1536)

    green_color = torch.tensor([0.1, 0.9, 0.2]).view(1, 1, 1, 3)
    h, w = cropped_img.shape[1:3]
    inpainted_sim = green_color.expand(1, h, w, 3)

    inpainted_cropped = cropped_img * (1 - cropped_mask.unsqueeze(-1)) + inpainted_sim * cropped_mask.unsqueeze(-1)

    final_image, = composite_node.composite_result(
        source_image=source_image,
        source_mask=mask,
        inpainted_image=inpainted_cropped,
        composite_context=context
    )

    assert final_image.shape == source_image.shape

    bool_mask = mask.squeeze(0).bool()

    final_pixels = final_image[0][bool_mask]
    assert torch.all(final_pixels[:, 1] > final_pixels[:, 0])
    assert torch.all(final_pixels[:, 1] > final_pixels[:, 2])

    assert torch.allclose(final_image[0][~bool_mask], source_image[0][~bool_mask], atol=1e-2)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_end_to_end_scenarios(image_1024, scenario):
    """
    A single, comprehensive test to validate the full node pipeline against various scenarios.
    """
    source_image = image_1024

    # 1. Setup based on the scenario
    mask = torch.zeros_like(source_image[..., 0])
    y, x, h, w = scenario["mask_rect"]
    mask[:, y:y + h, x:x + w] = 1.0  # Area to inpaint is 1

    crop_node = CropAndFitInpaintToDiffusionSize()
    composite_node = CompositeCroppedAndFittedInpaintResult()

    # 2. Run the first node
    cropped_img, cropped_mask, context = crop_node.crop_and_fit(
        source_image, mask, scenario["resolutions"], scenario["margin"]
    )

    # 3. Assert the outputs of the first node
    assert cropped_img.shape[1:3] == scenario["expected_cropped_shape"]
    assert context == scenario["expected_context"]

    # 4. Simulate inpainting
    green_color = torch.tensor([0.1, 0.9, 0.2]).view(1, 1, 1, 3)
    sim_h, sim_w = cropped_img.shape[1:3]
    inpainted_sim = green_color.expand(1, sim_h, sim_w, -1)

    # Inpaint the cropped region with the new color, respecting the mask feathering
    inpainted_cropped = cropped_img * (1 - cropped_mask.unsqueeze(-1)) + inpainted_sim * cropped_mask.unsqueeze(-1)

    # 5. Run the second node
    final_image, = composite_node.composite_result(
        source_image=source_image,
        source_mask=mask,
        inpainted_image=inpainted_cropped,
        composite_context=context
    )

    # 6. Assert the final composited image
    assert final_image.shape == source_image.shape

    # Check that the area to be inpainted (mask==1) is now the green color.
    bool_mask_to_inpaint = (mask.squeeze(0) > 0.0)
    assert torch.allclose(final_image[0][bool_mask_to_inpaint], green_color.squeeze(), atol=1e-2)

    # Check that the area that was not masked is completely unchanged.
    assert torch.allclose(final_image[0][~bool_mask_to_inpaint], source_image[0][~bool_mask_to_inpaint], atol=1e-2)

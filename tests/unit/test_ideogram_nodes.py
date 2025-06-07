import os
import pytest
import torch

from comfy.component_model.tensor_types import RGBImageBatch
from comfy_extras.nodes.nodes_ideogram import (
    IdeogramGenerate,
    IdeogramEdit,
    IdeogramRemix,
    IdeogramDescribe,
)


@pytest.fixture
def api_key():
    key = os.environ.get('IDEOGRAM_API_KEY')
    if not key:
        pytest.skip("IDEOGRAM_API_KEY environment variable not set")
    return key


@pytest.fixture
def sample_image() -> RGBImageBatch:
    """A light gray 1024x1024 image."""
    return torch.ones((1, 1024, 1024, 3), dtype=torch.float32) * 0.8


@pytest.fixture
def black_square_image() -> RGBImageBatch:
    """A black square image (1 batch, 1024x1024 pixels, 3 channels)"""
    return torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)


@pytest.fixture
def red_style_image() -> RGBImageBatch:
    """A solid red 512x512 image to be used as a style reference."""
    red_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
    red_image[..., 0] = 1.0  # Set red channel to max
    return red_image


def test_ideogram_describe(api_key, black_square_image):
    node = IdeogramDescribe()
    descriptions_list, = node.describe(images=black_square_image, api_key=api_key)
    # todo: why does this do some wacky thing about buildings?
    assert len(descriptions_list) > 0


@pytest.mark.parametrize(
    "model, aspect_ratio, use_style_ref",
    [
        ("V_2_TURBO", "disabled", False),  # Test V2 model
        ("V_3", "disabled", False),  # Test V3 model, no special args
        ("V_3", "16x9", True),  # Test V3 model with style and aspect ratio
    ],
)
def test_ideogram_generate(api_key, model, aspect_ratio, use_style_ref, red_style_image):
    node = IdeogramGenerate()
    style_ref = red_style_image if use_style_ref else None

    image, = node.generate(
        prompt="a vibrant fantasy landscape",
        resolution="RESOLUTION_1024_1024",
        model=model,
        magic_prompt_option="AUTO",
        api_key=api_key,
        num_images=1,
        aspect_ratio=aspect_ratio,
        style_reference_images=style_ref,
    )

    assert isinstance(image, torch.Tensor)
    assert torch.all((image >= 0) & (image <= 1))

    if model == "V_3":
        if aspect_ratio == "16x9":
            # For a 16x9 aspect ratio, width should be greater than height. Shape is (B, H, W, C)
            assert image.shape[2] > image.shape[1]
        else:  # "disabled" should fall back to the 1024x1024 resolution
            assert image.shape[1:] == (1024, 1024, 3)

        if use_style_ref:
            # Check for red color influence from the style image
            red_channel_mean = image[..., 0].mean().item()
            assert red_channel_mean > 0.35, "Red channel should be prominent due to style reference"


@pytest.mark.parametrize(
    "model, use_style_ref",
    [
        ("V_2_TURBO", False),  # Test V2 model
        ("V_3", False),  # Test V3 model, no style ref
        ("V_3", True),  # Test V3 model with style ref
    ],
)
def test_ideogram_edit(api_key, sample_image, model, use_style_ref, red_style_image):
    node = IdeogramEdit()
    style_ref = red_style_image if use_style_ref else None

    mask = torch.zeros((1, 1024, 1024), dtype=torch.float32)
    # Create a black square in the middle to be repainted
    mask[:, 256:768, 256:768] = 1.0

    image, = node.edit(
        images=sample_image, masks=mask,
        prompt="a vibrant, colorful object",
        model=model, api_key=api_key, num_images=1,
        style_reference_images=style_ref,
    )

    assert isinstance(image, torch.Tensor)
    assert image.shape[1:] == (1024, 1024, 3)

    if model == "V_3" and use_style_ref:
        # Check for red color influence in the edited region
        edited_region = image[:, 256:768, 256:768, :]
        red_channel_mean = edited_region[..., 0].mean().item()
        assert red_channel_mean > 0.35, "Red channel should be prominent in the edited region"


@pytest.mark.parametrize(
    "model, aspect_ratio, use_style_ref",
    [
        ("V_2_TURBO", "disabled", False),
        ("V_3", "disabled", False),
        ("V_3", "16x9", True),
    ],
)
def test_ideogram_remix(api_key, sample_image, model, aspect_ratio, use_style_ref, red_style_image):
    node = IdeogramRemix()
    style_ref = red_style_image if use_style_ref else None

    image, = node.remix(
        images=sample_image,
        prompt="transform into a vibrant, colorful abstract scene",
        resolution="RESOLUTION_1024_1024",
        model=model, api_key=api_key, num_images=1,
        aspect_ratio=aspect_ratio,
        style_reference_images=style_ref,
    )

    assert isinstance(image, torch.Tensor)

    if model == "V_3":
        if aspect_ratio == "16x9":
            assert image.shape[2] > image.shape[1]
        else:
            assert image.shape[1:] == (1024, 1024, 3)

        if use_style_ref:
            red_channel_mean = image[..., 0].mean().item()
            assert red_channel_mean > 0.35, "Red channel should be prominent due to style reference"

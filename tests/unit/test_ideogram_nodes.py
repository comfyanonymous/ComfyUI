import os

import pytest
import torch

from comfy_extras.nodes.nodes_ideogram import (
    IdeogramGenerate,
    IdeogramEdit,
    IdeogramRemix
)


@pytest.fixture
def api_key():
    key = os.environ.get('IDEOGRAM_API_KEY')
    if not key:
        pytest.skip("IDEOGRAM_API_KEY environment variable not set")
    return key


@pytest.fixture
def sample_image():
    return torch.ones((1, 1024, 1024, 3)) * 0.8  # Light gray image


def test_ideogram_generate(api_key):
    node = IdeogramGenerate()

    image, = node.generate(
        prompt="a serene mountain landscape at sunset with snow-capped peaks",
        resolution="RESOLUTION_1024_1024",
        model="V_2_TURBO",
        magic_prompt_option="AUTO",
        api_key=api_key,
        num_images=1
    )

    # Verify output format
    assert isinstance(image, torch.Tensor)
    assert image.shape[1:] == (1024, 1024, 3)  # HxWxC format
    assert image.dtype == torch.float32
    assert torch.all((image >= 0) & (image <= 1))


def test_ideogram_edit(api_key, sample_image):
    node = IdeogramEdit()

    # white is areas to keep, black is areas to repaint
    mask = torch.full((1, 1024, 1024), fill_value=1.0)
    center_start = 386
    center_end = 640
    mask[:, center_start:center_end, center_start:center_end] = 0.0

    image, = node.edit(
        images=sample_image,
        masks=mask,
        magic_prompt_option="OFF",
        prompt="a solid black rectangle",
        model="V_2_TURBO",
        api_key=api_key,
        num_images=1,
    )

    # Verify output format
    assert isinstance(image, torch.Tensor)
    assert image.shape[1:] == (1024, 1024, 3)
    assert image.dtype == torch.float32
    assert torch.all((image >= 0) & (image <= 1))

    # Verify the center is darker than the original
    center_region = image[:, center_start:center_end, center_start:center_end, :]
    outer_region = image[:, :center_start, :, :]  # Use top portion for comparison

    center_mean = center_region.mean().item()
    outer_mean = outer_region.mean().item()

    assert center_mean < outer_mean, f"Center region ({center_mean:.3f}) should be darker than outer region ({outer_mean:.3f})"
    assert center_mean < 0.6, f"Center region ({center_mean:.3f}) should be dark"


def test_ideogram_remix(api_key, sample_image):
    node = IdeogramRemix()

    image, = node.remix(
        images=sample_image,
        prompt="transform into a vibrant blue ocean scene with waves",
        resolution="RESOLUTION_1024_1024",
        model="V_2_TURBO",
        api_key=api_key,
        num_images=1
    )

    # Verify output format
    assert isinstance(image, torch.Tensor)
    assert image.shape[1:] == (1024, 1024, 3)
    assert image.dtype == torch.float32
    assert torch.all((image >= 0) & (image <= 1))

    # Since we asked for a blue ocean scene, verify there's significant blue component
    blue_channel = image[..., 2]  # RGB where blue is index 2
    blue_mean = blue_channel.mean().item()
    assert blue_mean > 0.4, f"Blue channel mean ({blue_mean:.3f}) should be significant for an ocean scene"

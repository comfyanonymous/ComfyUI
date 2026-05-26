import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy_extras.nodes_seedvr as nodes_seedvr  # noqa: E402


def test_resize_simple_multiplier_resolves_upscaled_shorter_edge():
    images = torch.zeros(1, 3, 16, 20, 3)

    output = nodes_seedvr.SeedVR2Resize.execute(images, 4.0)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 64, 80, 3)
    assert input_pixels.min().item() == 0.0
    assert input_pixels.max().item() == 0.0
    assert original_image is images
    assert upscaled_shorter_edge == 64


def test_resize_simple_silent_spatial_padding_keeps_unpadded_edge_output():
    images = torch.zeros(1, 1, 16, 16, 3)

    output = nodes_seedvr.SeedVR2Resize.execute(images, 7.5)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 1, 128, 128, 3)
    assert original_image is images
    assert upscaled_shorter_edge == 120


def test_resize_simple_rejects_non_positive_multiplier():
    images = torch.zeros(1, 1, 16, 16, 3)

    try:
        nodes_seedvr.SeedVR2Resize.execute(images, 0.0)
    except ValueError as e:
        assert "multiplier must be > 0" in str(e)
    else:
        raise AssertionError("non-positive multiplier was not rejected")


def test_resize_simple_rejects_multiplier_resolving_to_too_small_edge():
    images = torch.zeros(1, 1, 16, 16, 3)

    try:
        nodes_seedvr.SeedVR2Resize.execute(images, 0.01)
    except ValueError as e:
        assert "multiplier resolved upscaled_shorter_edge" in str(e)
        assert "at least 2 pixels" in str(e)
    else:
        raise AssertionError("too-small resolved edge was not rejected")


def test_resize_advanced_takes_exact_shorter_edge():
    images = torch.zeros(1, 1, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAdvanced.execute(images, 120)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 1, 128, 128, 3)
    assert original_image is images
    assert upscaled_shorter_edge == 120


def test_resize_advanced_treats_4d_image_as_one_video_frame_sequence():
    images = torch.zeros(2, 16, 16, 3)

    output = nodes_seedvr.SeedVR2ResizeAdvanced.execute(images, 120)

    input_pixels, original_image, upscaled_shorter_edge = output.result
    assert tuple(input_pixels.shape) == (1, 5, 128, 128, 3)
    assert original_image is images
    assert upscaled_shorter_edge == 120


def test_resize_advanced_rejects_one_pixel_shorter_edge():
    images = torch.zeros(1, 1, 16, 16, 3)

    try:
        nodes_seedvr.SeedVR2ResizeAdvanced.execute(images, 1)
    except ValueError as e:
        assert "upscaled_shorter_edge must be at least 2 pixels" in str(e)
    else:
        raise AssertionError("one-pixel shorter_edge was not rejected")


def test_resize_node_schemas_and_execute_signatures_are_preprocess_only():
    simple = nodes_seedvr.SeedVR2Resize.define_schema()
    advanced = nodes_seedvr.SeedVR2ResizeAdvanced.define_schema()

    assert [item.id for item in simple.inputs] == ["images", "multiplier"]
    assert simple.inputs[1].default == 4.0
    assert [item.id for item in simple.outputs] == [
        "input_pixels",
        "original_image",
        "upscaled_shorter_edge",
    ]

    assert [item.id for item in advanced.inputs] == ["images", "shorter_edge"]
    assert advanced.inputs[1].min == 2
    assert advanced.inputs[1].step is None
    assert [item.id for item in advanced.outputs] == [
        "input_pixels",
        "original_image",
        "upscaled_shorter_edge",
    ]

import asyncio

import torch

from comfy.cli_args import args

args.cpu = True

from comfy_extras.nodes_mask import BlurMask, MaskExtension


def blur_mask(mask, sigma):
    return BlurMask.execute(mask, sigma)[0]


def test_blur_mask_softens_hard_mask_edges():
    mask = torch.zeros(1, 9, 9)
    mask[:, :, :4] = 1.0

    result = blur_mask(mask, 1.0)

    assert result.shape == mask.shape
    assert 0.1 < float(result[0, 4, 3]) < 0.9
    assert 0.1 < float(result[0, 4, 4]) < 0.9
    assert float(result[0, 4, 0]) > 0.99
    assert float(result[0, 4, 8]) < 0.01


def test_blur_mask_zero_sigma_returns_original_values():
    mask = torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4)

    result = blur_mask(mask, 0.0)

    assert torch.equal(result, mask)


def test_blur_mask_preserves_batch_and_intermediate_values():
    mask = torch.zeros(2, 5, 5)
    mask[:, :, :2] = 0.5

    result = blur_mask(mask, 0.5)

    assert result.shape == mask.shape
    assert torch.all((result >= 0.0) & (result <= 1.0))
    assert torch.all(result[:, :, 1] > 0.0)
    assert torch.all(result[:, :, 3] > 0.0)


def test_blur_mask_is_registered():
    nodes = asyncio.run(MaskExtension().get_node_list())

    assert BlurMask in nodes

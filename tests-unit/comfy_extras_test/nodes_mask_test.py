import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import comfy


def _repeat_to_batch_size(tensor, batch_size):
    return tensor


mock_utils = MagicMock()
mock_utils.repeat_to_batch_size = _repeat_to_batch_size

mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

_utils_sentinel = object()
_prior_utils = getattr(comfy, "utils", _utils_sentinel)
comfy.utils = mock_utils

with patch.dict(
    "sys.modules",
    {
        "nodes": mock_nodes,
        "comfy.model_management": MagicMock(),
        "comfy.utils": mock_utils,
    },
):
    sys.modules.pop("comfy_extras.nodes_mask", None)
    from comfy_extras import nodes_mask

if _prior_utils is _utils_sentinel:
    delattr(comfy, "utils")
else:
    comfy.utils = _prior_utils


@pytest.fixture()
def composite(monkeypatch):
    monkeypatch.setattr(nodes_mask, "comfy", SimpleNamespace(utils=mock_utils))
    return nodes_mask.composite


@pytest.mark.parametrize("multiplier,x,y", [(1, -1, -1), (8, -8, -8)])
def test_negative_offsets_composite_the_visible_source_rectangle(
    composite, multiplier, x, y
):
    destination = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    source = torch.ones(1, 1, 2, 2)
    expected = destination.clone()
    expected[0, 0, 0, 0] = 1.0

    result = composite(destination.clone(), source, x, y, multiplier=multiplier)

    assert torch.equal(result, expected)


def test_negative_offsets_crop_mask_with_source(composite):
    destination = torch.zeros(1, 1, 4, 4)
    source = torch.full((1, 1, 2, 2), 2.0)
    mask = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    ).reshape(1, 1, 2, 2)

    result = composite(destination.clone(), source, -1, -1, mask=mask, multiplier=1)

    assert result[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert result[0, 0, 1, 0].item() == pytest.approx(0.0)
    assert torch.count_nonzero(result).item() == 1


def test_positive_offsets_continue_to_clip_to_destination(composite):
    destination = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    source = torch.ones(1, 1, 2, 2)
    expected = destination.clone()
    expected[0, 0, 3, 3] = 1.0

    result = composite(destination.clone(), source, 3, 3, multiplier=1)

    assert torch.equal(result, expected)


def test_non_overlapping_composite_returns_destination_unchanged(composite):
    destination = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    source = torch.ones(1, 1, 2, 2)
    mask = torch.ones_like(source)

    result = composite(destination.clone(), source, -2, -4, mask=mask, multiplier=1)

    assert torch.equal(result, destination)


@pytest.mark.parametrize(
    "node_class",
    [nodes_mask.LatentCompositeMasked, nodes_mask.ImageCompositeMasked],
)
def test_composite_widgets_accept_negative_coordinates(node_class):
    schema = node_class.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert inputs["x"].min == -mock_nodes.MAX_RESOLUTION
    assert inputs["y"].min == -mock_nodes.MAX_RESOLUTION

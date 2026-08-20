import pytest
import torch

import nodes


class FakeStyleModel:
    def get_cond(self, clip_vision_output):
        return torch.zeros(1, 2, 3)


def _conditioning():
    return [(torch.zeros(1, 1, 3), {})]


@pytest.mark.parametrize("strength", [1.0, 1.0000000000000002])
def test_attn_bias_neutral_strength_does_not_create_a_mask(strength):
    result = nodes.StyleModelApply().apply_stylemodel(
        _conditioning(),
        FakeStyleModel(),
        object(),
        strength,
        "attn_bias",
    )

    assert "attention_mask" not in result[0][0][1]


def test_attn_bias_non_neutral_strength_creates_a_mask():
    result = nodes.StyleModelApply().apply_stylemodel(
        _conditioning(),
        FakeStyleModel(),
        object(),
        1.001,
        "attn_bias",
    )

    assert "attention_mask" in result[0][0][1]

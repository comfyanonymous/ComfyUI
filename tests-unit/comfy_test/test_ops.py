import sys
from unittest import mock

import torch


with mock.patch.object(sys, "argv", ["ComfyUI", "--cpu"]):
    from comfy.options import enable_args_parsing

    enable_args_parsing()
    import comfy.memory_management
    import comfy.ops
    enable_args_parsing(False)


def test_lazy_linear_placeholder_has_torch_linear_weight_shape():
    with mock.patch.object(comfy.memory_management, "aimdo_enabled", True):
        linear = comfy.ops.disable_weight_init.Linear(3, 5, bias=False)
        result = linear.load_state_dict({}, strict=False, assign=False)

    assert result.missing_keys == ["weight"]
    assert linear.weight.shape == (5, 3)
    assert linear(torch.arange(6, dtype=torch.float32).reshape(2, 3)).shape == (2, 5)

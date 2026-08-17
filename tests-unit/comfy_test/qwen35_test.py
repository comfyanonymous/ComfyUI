import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ops  # noqa: E402
from comfy.text_encoders.qwen35 import (  # noqa: E402
    GatedDeltaNet,
    Qwen35Config,
)


def _tiny_config():
    return Qwen35Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        conv_kernel_size=2,
        layer_types=["linear_attention"],
        final_norm=False,
    )


def _module():
    module = GatedDeltaNet(
        _tiny_config(),
        device=torch.device("cpu"),
        dtype=torch.float32,
        ops=comfy.ops.manual_cast,
    )
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(
                parameter.shape
            )
            parameter.copy_((values.remainder(17) - 8) * (0.002 + index * 0.0001))
    return module.eval()


def test_gated_delta_net_casts_unmanaged_parameters_for_forward(monkeypatch):
    module = _module()
    a_log_id = id(module.A_log)
    dt_bias_id = id(module.dt_bias)
    state_keys = tuple(module.state_dict())
    calls = []
    original_cast = comfy.model_management.cast_to_device

    def capture_cast(tensor, device, dtype, copy=False):
        if tensor is module.A_log or tensor is module.dt_bias:
            calls.append((id(tensor), torch.device(device), dtype, copy))
        return original_cast(tensor, device, dtype, copy=copy)

    monkeypatch.setattr(comfy.model_management, "cast_to_device", capture_cast)
    x = torch.linspace(-0.25, 0.25, steps=24, dtype=torch.float32).reshape(1, 3, 8)

    with torch.inference_mode():
        output, present_state = module(x)

    assert present_state is None
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert calls == [
        (a_log_id, x.device, torch.float32, False),
        (dt_bias_id, x.device, torch.float32, False),
    ]
    assert id(module.A_log) == a_log_id
    assert id(module.dt_bias) == dt_bias_id
    assert module.A_log.device.type == "cpu"
    assert module.dt_bias.device.type == "cpu"
    assert tuple(module.state_dict()) == state_keys

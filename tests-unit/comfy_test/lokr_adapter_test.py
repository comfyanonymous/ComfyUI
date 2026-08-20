import pytest
import torch

from comfy.cli_args import args


@pytest.fixture(autouse=True)
def use_cpu():
    previous_cpu = args.cpu
    args.cpu = True
    yield
    args.cpu = previous_cpu


def test_tucker_convolution_with_non_contiguous_factor_storage_is_applied():
    # This import must follow use_cpu so model management initializes on CPU.
    from comfy.weight_adapter.lokr import LoKrAdapter

    tucker_w2 = torch.einsum(
        "i j k l, j r, i p -> p r k l",
        torch.ones(4, 3, 3, 3),
        torch.ones(3, 3),
        torch.ones(4, 4),
    )
    w2 = tucker_w2.transpose(1, 2).contiguous().transpose(1, 2)
    lora = {
        "layer.lokr_w1": torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1,
        "layer.lokr_w2": w2,
    }
    loaded_keys = set()
    adapter = LoKrAdapter.load("layer", lora, 3.0, None, loaded_keys)

    weight = torch.zeros(8, 6, 3, 3)
    result = adapter.calculate_weight(
        weight,
        "layer.weight",
        strength=1.0,
        strength_model=1.0,
        offset=None,
        function=lambda value: value,
    )

    expected_diff = torch.kron(
        lora["layer.lokr_w1"].unsqueeze(2).unsqueeze(2), w2.contiguous()
    ).reshape(weight.shape)

    assert loaded_keys == {"layer.lokr_w1", "layer.lokr_w2"}
    assert not w2.is_contiguous()
    assert torch.equal(result, expected_diff)
    train_adapter = adapter.to_train()
    assert not train_adapter.w2.is_contiguous()
    train_result = train_adapter(torch.zeros_like(weight))
    assert torch.equal(train_result, expected_diff)

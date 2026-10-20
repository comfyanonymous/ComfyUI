import weakref

import pytest
import torch

import comfy.ops
from comfy.ldm.krea2.model import Attention, SwiGLU


@pytest.mark.parametrize("operations", [
    torch.nn,
    comfy.ops.disable_weight_init,
    comfy.ops.manual_cast,
    comfy.ops.mixed_precision_ops(compute_dtype=torch.float32),
])
def test_plain_linears_keep_forward_hooks_and_gradients(operations):
    layers = [operations.Linear(4, n, device="cpu", dtype=torch.float32) for n in (3, 2)]
    for layer, n in zip(layers, (3, 2)):
        layer.load_state_dict(torch.nn.Linear(4, n).state_dict())
    calls = []
    handles = [layer.register_forward_hook(
        lambda module, inputs, output: calls.append(module)
    ) for layer in layers]
    x = torch.randn(5, 4, requires_grad=True)
    expected = tuple(layer(x) for layer in layers)
    calls.clear()

    actual = comfy.ops.linear_group(x, layers)
    assert calls == layers
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(sum(a.sum() for a in actual), x)[0]
    expected_grad = torch.autograd.grad(sum(b.sum() for b in expected), x)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    for handle in handles:
        handle.remove()


def test_group_with_plain_torch_linear_uses_ordinary_calls():
    class PreparingLinear(comfy.ops.manual_cast.Linear):
        def prepare_input(self, x):
            raise AssertionError("Mixed groups must use ordinary calls")

    layers = [PreparingLinear(4, 3), torch.nn.Linear(4, 2)]
    layers[0].load_state_dict(torch.nn.Linear(4, 3).state_dict())
    x = torch.randn(5, 4)
    actual = comfy.ops.linear_group(x, layers)
    for layer, output in zip(layers, actual):
        torch.testing.assert_close(output, layer(x), rtol=0, atol=0)


def test_preparation_is_shared_and_lives_only_for_the_group():
    prepared_refs = []
    preparation_calls = []
    consumed = []

    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def __init__(self, factor, eligible):
            super().__init__()
            self.factor = factor
            self.eligible = eligible

        def prepare_input(self, x):
            if not self.eligible:
                return None
            preparation_calls.append(self)
            prepared = x.square()
            prepared_refs.append(weakref.ref(prepared))
            return prepared

        def forward_prepared(self, x, prepared):
            if not self.eligible:
                return self(x)
            consumed.append(id(prepared))
            return prepared * self.factor

        def forward(self, x):
            return x.square() * self.factor

    layers = [Linear(2, False), Linear(3, True), Linear(4, True)]
    for x in (torch.randn(3, 4), torch.randn(3, 4)):
        actual = comfy.ops.linear_group(x, layers)
        for layer, output in zip(layers, actual):
            torch.testing.assert_close(output, layer(x), rtol=0, atol=0)
        assert prepared_refs[-1]() is None
    assert preparation_calls == [layers[1], layers[1]]
    assert consumed[0] == consumed[1]
    assert consumed[2] == consumed[3]


@pytest.mark.parametrize("kind", ["pre", "post", "global_pre", "global_post", "force_forward"])
def test_hooks_that_change_shared_input_keep_separate_call_semantics(kind):
    class Linear(torch.nn.Module, comfy.ops.CastWeightBiasOp):
        def prepare_input(self, x):
            raise AssertionError("Hooked groups must use ordinary calls")

        def forward(self, x):
            return x.square()

    layers = [Linear(), Linear()]

    def change_input(module, inputs, *output):
        if module is layers[0]:
            inputs[0].add_(1)

    if kind == "pre":
        handle = layers[0].register_forward_pre_hook(change_input)
    elif kind == "post":
        handle = layers[0].register_forward_hook(change_input)
    elif kind == "global_pre":
        handle = torch.nn.modules.module.register_module_forward_pre_hook(change_input)
    elif kind == "global_post":
        handle = torch.nn.modules.module.register_module_forward_hook(change_input)
    else:
        layers[0].comfy_force_forward = True
        handle = None
    try:
        x = torch.randn(3, 4)
        reference = x.clone()
        expected = tuple(layer(reference) for layer in layers)
        actual = comfy.ops.linear_group(x, layers)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(x, reference, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()


@pytest.mark.parametrize("module", ["mlp", "attention"])
def test_krea_grouped_projections_match_separate_calls(module, monkeypatch):
    if module == "mlp":
        model = SwiGLU(32, 4, multiple=16, operations=comfy.ops.manual_cast)
    else:
        model = Attention(32, 4, kvheads=2, operations=comfy.ops.manual_cast)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.1)
    x = torch.randn(2, 5, 32)
    actual = model(x)
    monkeypatch.setattr(comfy.ops, "prepare_linear_group", lambda x, layers: None)
    expected = model(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_krea_mlp_releases_raw_gate_before_up_and_projections_before_down():
    model = SwiGLU(32, 4, multiple=16, operations=comfy.ops.manual_cast)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.1)
        parameter.requires_grad_(False)
    outputs = []
    for layer in (model.gate, model.up):
        layer.register_forward_hook(
            lambda module, inputs, output: outputs.append(weakref.ref(output))
        )

    def before_up(module, inputs):
        assert len(outputs) == 1
        assert outputs[0]() is None

    def before_down(module, inputs):
        assert len(outputs) == 2
        assert all(output() is None for output in outputs)

    model.up.register_forward_pre_hook(before_up)
    model.down.register_forward_pre_hook(before_down)
    model(torch.randn(2, 5, 32))

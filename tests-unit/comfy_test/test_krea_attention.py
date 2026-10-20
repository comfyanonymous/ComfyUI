import pytest
import torch

import comfy.model_patcher
import comfy.ops
from comfy.ldm.krea2.model import Attention
from comfy.ldm.modules import attention


@pytest.mark.parametrize("kvheads", [2, 4])
@pytest.mark.parametrize("backend,mask_kind", [
    ("pytorch", None), ("pytorch", "bool"), ("pytorch", "additive"), ("sub_quad", None),
])
def test_krea_passes_grouped_heads_to_selected_attention(kvheads, backend, mask_kind):
    torch.manual_seed(42)
    model = Attention(32, 4, kvheads=kvheads, operations=comfy.ops.manual_cast)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.1)
    patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    selected = attention.get_attention_function(backend)
    x = torch.randn(2, 5, 32)
    mask = None
    if mask_kind is not None:
        mask = torch.ones(5, 5, dtype=torch.bool)
        mask[:, -1] = False
        if mask_kind == "additive":
            mask = torch.zeros(5, 5).masked_fill(~mask, float("-inf"))
    events = []

    def input_patch(q, k, v, **kwargs):
        assert (q.shape[1], k.shape[1], v.shape[1]) == (4, kvheads, kvheads)
        events.append("input")
        return {"q": q * 0.75, "v": v + 0.1}

    def output_patch(out, extra_options):
        assert out.shape == (2, 5, 32)
        events.append("output")
        return out * 0.5

    def compact(q, k, v, heads, **kwargs):
        assert (q.shape[1], k.shape[1], v.shape[1]) == (4, kvheads, kvheads)
        assert kwargs["enable_gqa"] == (kvheads != heads)
        events.append("backend")
        return selected(q, k, v, heads, **kwargs)

    def expanded(q, k, v, heads, **kwargs):
        k = k.repeat_interleave(heads // kvheads, dim=1)
        v = v.repeat_interleave(heads // kvheads, dim=1)
        kwargs["enable_gqa"] = False
        events.append("backend")
        return selected(q, k, v, heads, **kwargs)

    patcher.set_model_attn1_patch(input_patch)
    patcher.set_model_attn1_output_patch(output_patch)
    patcher.set_model_optimized_attention(compact)
    options = {**patcher.model_options["transformer_options"], "block_index": 0}
    actual = model(x, mask=mask, transformer_options=options)
    assert events == ["input", "backend", "output"]
    events.clear()
    patcher.set_model_optimized_attention(expanded)
    options = {**patcher.model_options["transformer_options"], "block_index": 0}
    expected = model(x, mask=mask, transformer_options=options)
    assert events == ["input", "backend", "output"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("skip_reshape", [False, True])
@pytest.mark.parametrize("skip_output_reshape", [False, True])
def test_sol_expands_grouped_heads_for_its_backend(monkeypatch, skip_reshape, skip_output_reshape):
    torch.manual_seed(42)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)
    expected_k = k.repeat_interleave(2, dim=1).transpose(1, 2)
    expected_v = v.repeat_interleave(2, dim=1).transpose(1, 2)
    calls = []

    def sol(q, k, v, **kwargs):
        assert q.shape == k.shape == v.shape == (2, 5, 4, 8)
        torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(v, expected_v, rtol=0, atol=0)
        assert kwargs == {"tau": 1.2, "scale": 0.25}
        calls.append(True)
        return q

    monkeypatch.setattr(attention.comfy_kitchen, "sol_attn", sol)
    if not skip_reshape:
        q, k, v = (t.transpose(1, 2).flatten(2) for t in (q, k, v))
    actual = attention.attention_comfy_kitchen_sol(
        q, k, v, 4, enable_gqa=True, skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape, tau=1.2, scale=0.25,
    )
    assert calls == [True]
    expected = q if skip_reshape else q.unflatten(-1, (4, 8)).transpose(1, 2)
    if not skip_output_reshape:
        expected = expected.transpose(1, 2).flatten(2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

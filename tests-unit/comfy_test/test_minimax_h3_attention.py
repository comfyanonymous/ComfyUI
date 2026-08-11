import torch

import comfy.ldm.minimax.model as minimax_model
import comfy.ops


def test_h3_attention_makes_large_qkv_contiguous_only_on_gfx1151(monkeypatch):
    attention = minimax_model.Attention(
        hidden=8,
        heads=2,
        head_dim=128,
        eps=1e-5,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    x = torch.randn(5000, 8)
    layouts = []

    def capture_attention(q, k, v, heads, **kwargs):
        q, k, v = (x.peek() for x in (q, k, v))
        layouts.append(tuple(t.is_contiguous() for t in (q, k, v)))
        return q.transpose(1, 2).reshape(1, q.shape[2], heads * q.shape[3])

    monkeypatch.setattr(minimax_model, "optimized_attention", capture_attention)
    monkeypatch.setattr(minimax_model.comfy.model_management, "is_amd", lambda: True)

    arch = ["gfx1100"]
    monkeypatch.setattr(minimax_model, "_amd_arch", lambda device: arch[0])
    gfx1100_output = attention(x)

    arch[0] = "gfx1151"
    gfx1151_output = attention(x)

    assert layouts == [(False, False, False), (True, True, True)]
    torch.testing.assert_close(gfx1151_output, gfx1100_output)


def make_qkv(seq=5000, dim=128, contiguous=False):
    if contiguous:
        return tuple(torch.randn(1, 2, seq, dim) for _ in range(3))
    fused = torch.randn(seq, 6 * dim)
    return tuple(x.view(seq, 2, dim).transpose(0, 1).unsqueeze(0) for x in fused.split(2 * dim, dim=-1))


def test_h3_qkv_contiguous_gate_rejects_unvalidated_inputs(monkeypatch):
    cases = [
        (False, make_qkv()),
        (True, make_qkv(seq=4999)),
        (True, make_qkv(dim=64)),
        (True, make_qkv(contiguous=True)),
    ]

    for is_amd, inputs in cases:
        arch_calls = []
        monkeypatch.setattr(minimax_model.comfy.model_management, "is_amd", lambda: is_amd)
        monkeypatch.setattr(minimax_model, "_amd_arch", lambda device: arch_calls.append(device))
        outputs = minimax_model._contiguous_qkv_for_gfx1151(*inputs)
        assert arch_calls == []
        assert all(output is original for output, original in zip(outputs, inputs))


def test_h3_qkv_contiguous_gate_accepts_5000_boundary(monkeypatch):
    monkeypatch.setattr(minimax_model.comfy.model_management, "is_amd", lambda: True)
    arch_calls = []
    monkeypatch.setattr(minimax_model, "_amd_arch", lambda device: arch_calls.append(device) or "gfx1151")
    inputs = make_qkv(seq=5000)

    outputs = minimax_model._contiguous_qkv_for_gfx1151(*inputs)

    assert arch_calls == [inputs[0].device]
    assert all(output is not original for output, original in zip(outputs, inputs))
    assert all(output.is_contiguous() for output in outputs)


def test_h3_amd_arch_is_device_aware_cached(monkeypatch):
    minimax_model._AMD_ARCH_CACHE.clear()
    calls = []
    monkeypatch.setattr(minimax_model.torch.cuda, "device_count", lambda: 2)

    def get_properties(device):
        calls.append(device)
        return type("Props", (), {"gcnArchName": f"gfx115{device + 1}:sramecc+:xnack-"})()

    monkeypatch.setattr(minimax_model.torch.cuda, "get_device_properties", get_properties)

    assert minimax_model._amd_arch(torch.device("cpu")) is None
    assert calls == []
    assert minimax_model._amd_arch(torch.device("cuda:0")) == "gfx1151"
    assert minimax_model._amd_arch(torch.device("cuda:0")) == "gfx1151"
    assert minimax_model._amd_arch(torch.device("cuda:1")) == "gfx1152"
    assert minimax_model._amd_arch(torch.device("cuda:1")) == "gfx1152"
    assert calls == [0, 1]

    minimax_model._AMD_ARCH_CACHE.clear()


def test_h3_amd_arch_caches_invalid_device_miss(monkeypatch):
    minimax_model._AMD_ARCH_CACHE.clear()
    calls = []
    monkeypatch.setattr(minimax_model.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(minimax_model.torch.cuda, "get_device_properties", lambda device: calls.append(device))

    assert minimax_model._amd_arch(torch.device("cuda:1")) is None
    assert minimax_model._amd_arch(torch.device("cuda:1")) is None
    assert calls == []
    assert minimax_model._AMD_ARCH_CACHE == {("cuda", 1): None}

    minimax_model._AMD_ARCH_CACHE.clear()


def test_h3_amd_arch_propagates_property_errors(monkeypatch):
    minimax_model._AMD_ARCH_CACHE.clear()
    monkeypatch.setattr(minimax_model.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(minimax_model.torch.cuda, "get_device_properties", lambda device: (_ for _ in ()).throw(RuntimeError("probe failed")))

    try:
        minimax_model._amd_arch(torch.device("cuda:0"))
    except RuntimeError as error:
        assert str(error) == "probe failed"
    else:
        raise AssertionError("CUDA property error was silently ignored")
    assert minimax_model._AMD_ARCH_CACHE == {}

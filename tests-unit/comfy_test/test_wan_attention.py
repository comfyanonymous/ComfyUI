import torch

import comfy.ldm.wan.model as wan_model
import comfy.ops


def test_wan_self_attention_preserves_outputs_across_layout_paths(monkeypatch):
    attention = wan_model.WanSelfAttention(
        dim=16,
        num_heads=4,
        qk_norm=False,
        operation_settings={
            "operations": comfy.ops.disable_weight_init,
            "device": "cpu",
            "dtype": torch.float32,
        },
    )
    for parameter in attention.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    layouts = []

    def capture_attention(q, k, v, heads, skip_reshape=False, **kwargs):
        layouts.append((q.ndim, tuple(t.is_contiguous() for t in (q, k, v))))
        if not skip_reshape:
            batch, sequence, channels = q.shape
            head_dim = channels // heads
            q, k, v = (
                tensor.view(batch, sequence, heads, head_dim).transpose(1, 2)
                for tensor in (q, k, v)
            )
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)

    monkeypatch.setattr(wan_model, "optimized_attention", capture_attention)
    monkeypatch.setattr(wan_model, "apply_rope1", lambda tensor, freqs: tensor)
    monkeypatch.setattr(wan_model, "_should_make_qkv_contiguous", lambda q, k, v: False)
    x = torch.randn(1, 23, 16)
    original_output = attention(x, None)

    monkeypatch.setattr(
        wan_model,
        "_should_make_qkv_contiguous",
        lambda q, k, v: True,
    )
    optimized_output = attention(x, None)

    assert layouts == [(3, (True, True, True)), (4, (True, True, True))]
    torch.testing.assert_close(optimized_output, original_output, rtol=0, atol=0)


def make_qkv(seq=5000, dim=128, contiguous=False, device="cpu"):
    if contiguous:
        return tuple(torch.randn(1, 2, seq, dim, device=device) for _ in range(3))
    fused = torch.randn(1, seq, 6 * dim, device=device)
    return tuple(x.view(1, seq, 2, dim).transpose(1, 2) for x in fused.split(2 * dim, dim=-1))


def test_wan_qkv_gate_rejects_unvalidated_inputs(monkeypatch):
    probe_calls = []
    monkeypatch.setattr(wan_model, "_amd_arch", lambda device: probe_calls.append(device))
    cases = [make_qkv(seq=4096), make_qkv(dim=64), make_qkv(contiguous=True)]

    for inputs in cases:
        assert not wan_model._should_make_qkv_contiguous(*inputs)
    assert probe_calls == []


def test_wan_qkv_gate_rejects_other_architectures(monkeypatch):
    inputs = make_qkv()
    monkeypatch.setattr(wan_model, "_amd_arch", lambda device: "gfx1100")
    assert not wan_model._should_make_qkv_contiguous(*inputs)


def test_wan_qkv_gate_accepts_validated_shape_on_gfx1151(monkeypatch):
    inputs = make_qkv()
    monkeypatch.setattr(wan_model, "_amd_arch", lambda device: "gfx1151")
    assert wan_model._should_make_qkv_contiguous(*inputs)


def test_wan_amd_arch_probe_is_device_guarded_cached_and_fail_closed(monkeypatch):
    wan_model._amd_arch_cache.clear()
    calls = []
    monkeypatch.setattr(wan_model.comfy.model_management, "is_amd", lambda: True)
    monkeypatch.setattr(
        wan_model.torch.cuda,
        "get_device_properties",
        lambda device: calls.append(device) or type("Props", (), {"gcnArchName": "gfx1151:sramecc+"})(),
    )
    cuda = torch.device("cuda", 0)

    assert wan_model._amd_arch(torch.device("cpu")) is None
    assert wan_model._amd_arch(cuda) == "gfx1151"
    assert wan_model._amd_arch(cuda) == "gfx1151"
    assert calls == [cuda]

    wan_model._amd_arch_cache.clear()
    monkeypatch.setattr(
        wan_model.torch.cuda,
        "get_device_properties",
        lambda device: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    assert wan_model._amd_arch(cuda) is None
    assert wan_model._amd_arch(cuda) is None

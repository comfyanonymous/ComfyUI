import torch

import comfy.ldm.wan.model as wan_model
import comfy.ops


def test_wan_self_attention_makes_qkv_contiguous_only_on_amd(monkeypatch):
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

    x = torch.randn(1, 23, 16)
    monkeypatch.setattr(wan_model.comfy.model_management, "is_amd", lambda: False)
    non_amd_output = attention(x, None)

    monkeypatch.setattr(wan_model.comfy.model_management, "is_amd", lambda: True)
    amd_output = attention(x, None)

    assert layouts == [(3, (True, True, True)), (4, (True, True, True))]
    torch.testing.assert_close(amd_output, non_amd_output, rtol=0, atol=0)

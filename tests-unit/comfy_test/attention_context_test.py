import torch

from comfy.ldm.cosmos import predict2
from comfy.ldm.wan.ar_model import CausalWanModel


def test_cosmos_attention_passes_self_attention_context(monkeypatch):
    captured = {}

    def capture_attention(q, k, v, heads, **kwargs):
        captured.update(kwargs)
        return q

    monkeypatch.setattr(predict2, "optimized_attention", capture_attention)
    q = torch.zeros((1, 2, 3, 4))

    predict2.torch_attention_op(q, q, q, is_self_attention=True)

    assert captured["is_self_attention"] is True
    assert captured["attention_token_shape"] == (2,)


def test_causal_wan_forward_passes_transformer_options_to_ar_block():
    transformer_options = {
        "ar_state": {
            "start_frame": 2,
            "kv_caches": [object()],
            "crossattn_caches": [object()],
        },
        "optimized_attention_override": object(),
    }
    captured = {}

    class FakeCausalWan:
        def forward_block(self, **kwargs):
            captured.update(kwargs)
            return "result"

    result = CausalWanModel.forward(
        FakeCausalWan(),
        x=torch.zeros((1, 4, 3, 2, 2)),
        timestep=torch.zeros(1),
        context=torch.zeros((1, 1, 1)),
        transformer_options=transformer_options,
    )

    assert result == "result"
    assert captured["transformer_options"] is transformer_options

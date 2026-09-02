import torch

import comfy.ldm.minimax.model as minimax


class QKV(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x, x + 1, x + 2), dim=-1)


def test_attention_patch_accepts_tuple_and_mapping_callbacks(monkeypatch):
    attention = minimax.Attention(4, 2, 2, 1e-6, operations=torch.nn)
    attention.qkv_proj = QKV()
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.out_proj = torch.nn.Identity()
    seen = {}

    def tuple_patch(q, k, v, extra_options):
        assert extra_options["block_index"] == 3
        return q + 1, k + 1, v + 1

    def mapping_patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        assert pe is None
        assert attn_mask is None
        assert extra_options["n_heads"] == 2
        return {"q": q * 2, "v": v * 3}

    def output_patch(out, extra_options):
        assert extra_options["block_index"] == 3
        return out + 4

    def fake_attention(q, k, v, *args, **kwargs):
        seen.update(q=q, k=k, v=v)
        return v.transpose(1, 2).reshape(1, v.shape[2], -1)

    monkeypatch.setattr(minimax, "optimized_attention", fake_attention)
    x = torch.zeros(2, 4)
    output = attention(
        x,
        transformer_options={
            "block_index": 3,
            "patches": {
                "attn1_patch": [tuple_patch, mapping_patch],
                "attn1_output_patch": [output_patch],
            },
        },
    )

    assert torch.equal(seen["q"], torch.full((1, 2, 2, 2), 2.0))
    assert torch.equal(seen["k"], torch.full((1, 2, 2, 2), 2.0))
    assert torch.equal(seen["v"], torch.full((1, 2, 2, 2), 9.0))
    assert torch.equal(output, torch.full((2, 4), 13.0))

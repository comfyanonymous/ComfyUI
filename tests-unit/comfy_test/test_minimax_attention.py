import types

import torch

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.ldm.minimax import model as minimax_model


class FakeOperations:
    @staticmethod
    def Linear(*args, **kwargs):
        return FakeLinear()

    @staticmethod
    def RMSNorm(dim, eps, dtype=None, device=None):
        return FakeRMSNorm(eps)


class FakeLinear:
    def __init__(self, result=None):
        self.result = result

    def __call__(self, x):
        return x if self.result is None else self.result


class FakeRMSNorm:
    def __init__(self, eps):
        self.eps = eps
        self.weight = torch.ones(())

    def __call__(self, x):
        return x


def make_attention(qkv):
    attention = minimax_model.Attention(
        hidden=8,
        heads=2,
        head_dim=4,
        eps=1e-5,
        operations=FakeOperations,
    )
    attention.qkv_proj = FakeLinear(qkv)
    attention.q_norm = FakeRMSNorm(attention.q_norm.eps)
    attention.k_norm = FakeRMSNorm(attention.k_norm.eps)
    attention.out_proj = FakeLinear()
    return attention


def count_tensor_clones(monkeypatch):
    clones = []
    original_clone = torch.Tensor.clone

    def counting_clone(tensor, *args, **kwargs):
        clones.append(tensor)
        return original_clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)
    return clones


def test_minimax_attention_value_is_detached_and_preformatted(monkeypatch):
    """#15665: v must leave the fused qkv buffer with exactly one copy.

    The old v.clone() detached the buffer but kept [seq, heads, dim] layout,
    so every attention backend then paid a second full-size .contiguous() for
    the transposed view. The replacement materializes the backend layout
    directly: storage detached (#15486 preserved), tensor contiguous, and no
    clone anywhere on the path.
    """
    sequence_length = 5
    heads = 2
    head_dim = 4
    inner_dim = heads * head_dim
    qkv = torch.arange(sequence_length * inner_dim * 3, dtype=torch.float32).reshape(
        sequence_length, inner_dim * 3
    )
    expected_v = (
        qkv[:, inner_dim * 2 :]
        .view(sequence_length, heads, head_dim)
        .transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    captured = {}

    def fake_optimized_attention(q, k, v, heads, **kwargs):
        captured["v"] = v.take()
        q.take()
        k.take()
        return captured["v"].transpose(1, 2).reshape(1, sequence_length, inner_dim)

    monkeypatch.setattr(minimax_model, "optimized_attention", fake_optimized_attention)
    clones = count_tensor_clones(monkeypatch)
    attention = make_attention(qkv)

    output = attention(torch.zeros(sequence_length, 8))

    assert clones == [], "v must be produced by a single contiguous copy, not clone()"
    v = captured["v"]
    assert v.shape == (1, heads, sequence_length, head_dim)
    assert v.is_contiguous(), "backend-layout v must be contiguous so no second copy happens"
    assert v.untyped_storage().data_ptr() != qkv.untyped_storage().data_ptr(), (
        "v must not pin the 3x-wide fused qkv buffer through attention (#15486)"
    )
    torch.testing.assert_close(output, expected_v.transpose(1, 2).reshape(1, sequence_length, inner_dim).squeeze(0), rtol=0, atol=0)


def test_minimax_attention_value_is_isolated_from_inplace_rope_writes(monkeypatch):
    """In-place rope writes to the q/k views must never reach v (#15486)."""
    sequence_length = 5
    heads = 2
    head_dim = 4
    inner_dim = heads * head_dim
    qkv = torch.arange(sequence_length * inner_dim * 3, dtype=torch.float32).reshape(
        sequence_length, inner_dim * 3
    )
    expected_v = (
        qkv[:, inner_dim * 2 :]
        .view(sequence_length, heads, head_dim)
        .transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
    )
    captured = {}

    def fake_rope(q, k, rope_freqs, q_weight, k_weight, epsilon, rot_dim):
        captured["rope_q"] = q
        captured["rope_k"] = k
        q.add_(1.0)
        k.add_(2.0)
        return q, k

    def fake_optimized_attention(q, k, v, heads, **kwargs):
        captured["attention_v"] = v.take()
        q.take()
        k.take()
        return captured["attention_v"].transpose(1, 2).reshape(1, sequence_length, inner_dim)

    monkeypatch.setattr(
        minimax_model.comfy.quant_ops,
        "ck",
        types.SimpleNamespace(rms_rope_split_half=fake_rope),
        raising=False,
    )
    monkeypatch.setattr(minimax_model.comfy.model_management, "in_training", True)
    monkeypatch.setattr(minimax_model, "optimized_attention", fake_optimized_attention)
    clones = count_tensor_clones(monkeypatch)
    attention = make_attention(qkv)
    rope_freqs = torch.zeros(1, sequence_length, 1, 1, 2, 2)

    output = attention(torch.zeros(sequence_length, 8), rope_freqs=rope_freqs)

    assert clones == []
    assert captured["rope_q"].untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    assert captured["rope_k"].untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()
    v = captured["attention_v"]
    assert v.untyped_storage().data_ptr() != qkv.untyped_storage().data_ptr()
    torch.testing.assert_close(output, expected_v.transpose(1, 2).reshape(1, sequence_length, inner_dim).squeeze(0), rtol=0, atol=0)

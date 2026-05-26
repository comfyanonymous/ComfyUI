from __future__ import annotations

import torch
from torch import nn

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

import comfy.ldm.seedvr.model as seedvr_model  # noqa: E402


class _StubModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


def _capture_last_layer_flags(monkeypatch, vid_dim: int, txt_in_dim: int) -> list[bool]:
    flags = []

    class _Block(_StubModule):
        def __init__(self, *args, **kwargs):
            flags.append(kwargs["is_last_layer"])
            super().__init__()

    monkeypatch.setattr(seedvr_model, "NaPatchIn", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaPatchOut", _StubModule)
    monkeypatch.setattr(seedvr_model, "TimeEmbedding", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaMMSRTransformerBlock", _Block)

    seedvr_model.NaDiT(
        norm_eps=1e-5,
        qk_rope=None,
        num_layers=4,
        mlp_type="normal",
        vid_dim=vid_dim,
        txt_in_dim=txt_in_dim,
        heads=24,
        mm_layers=3,
    )

    return flags


def test_seedvr2_7b_keeps_final_block_text_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=3072, txt_in_dim=3072) == [
        False,
        False,
        False,
        False,
    ]


def test_seedvr2_3b_keeps_final_block_vid_only_path(monkeypatch):
    assert _capture_last_layer_flags(monkeypatch, vid_dim=2560, txt_in_dim=2560) == [
        False,
        False,
        False,
        True,
    ]


def _capture_block_attention_rope_type(monkeypatch, qk_rope):
    rope_types = []

    class _Attention(_StubModule):
        def __init__(self, *args, **kwargs):
            rope_types.append(kwargs["rope_type"])
            super().__init__()

    monkeypatch.setattr(seedvr_model, "MMModule", _StubModule)
    monkeypatch.setattr(seedvr_model, "NaSwinAttention", _Attention)

    seedvr_model.NaMMSRTransformerBlock(
        vid_dim=4,
        txt_dim=4,
        emb_dim=4,
        heads=1,
        head_dim=4,
        expand_ratio=1,
        norm=_StubModule,
        norm_eps=1e-5,
        ada=_StubModule,
        qk_bias=False,
        qk_rope=qk_rope,
        qk_norm=_StubModule,
        mlp_type="normal",
        shared_weights=False,
        rope_type="mmrope3d",
        rope_dim=4,
        is_last_layer=False,
        device="cpu",
        dtype=torch.float32,
        operations=seedvr_model.comfy.ops.disable_weight_init,
    )

    return rope_types


def test_seedvr2_3b_qk_rope_none_preserves_checkpoint_rope_buffers(monkeypatch):
    assert _capture_block_attention_rope_type(monkeypatch, qk_rope=None) == ["mmrope3d"]


def test_seedvr2_7b_qk_rope_true_preserves_attention_rope(monkeypatch):
    assert _capture_block_attention_rope_type(monkeypatch, qk_rope=True) == ["mmrope3d"]


def test_seedvr2_7b_rope3d_matches_checkpoint_buffer_shape():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)

    assert isinstance(rope, seedvr_model.NaRotaryEmbedding3d)
    assert tuple(rope.rope.freqs.shape) == (10,)


def test_seedvr2_7b_rope3d_preserves_qk_shape():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    q = torch.randn(4, 2, 128)
    k = torch.randn(4, 2, 128)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)

    q_out, k_out = rope(q, k, shape, seedvr_model.Cache(disable=True))

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_seedvr2_7b_rope3d_matches_wrapper_oracle():
    rope = seedvr_model.get_na_rope("rope3d", dim=64)
    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(4, 2, 128, generator=generator)
    k = torch.randn(4, 2, 128, generator=generator)
    shape = torch.tensor([[1, 2, 2]], dtype=torch.long)
    freqs = rope.get_axial_freqs(1, 2, 2).reshape(4, -1)

    expected_q = seedvr_model.apply_rotary_emb(
        freqs,
        q.permute(1, 0, 2).float(),
    ).to(q.dtype).permute(1, 0, 2)
    expected_k = seedvr_model.apply_rotary_emb(
        freqs,
        k.permute(1, 0, 2).float(),
    ).to(k.dtype).permute(1, 0, 2)

    actual_q, actual_k = rope(q.clone(), k.clone(), shape, seedvr_model.Cache(disable=True))

    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


def test_seedvr2_mmrope_handles_large_spatial_grid_without_truncation():
    rope = seedvr_model.NaMMRotaryEmbedding3d(dim=12)
    vid_shape = torch.tensor([[1, 129, 130]], dtype=torch.long)
    txt_shape = torch.tensor([[2]], dtype=torch.long)
    vid_tokens = int(vid_shape.prod().item())
    txt_tokens = int(txt_shape.prod().item())
    vid_q = torch.zeros(vid_tokens, 1, 12)
    vid_k = torch.zeros_like(vid_q)
    txt_q = torch.zeros(txt_tokens, 1, 12)
    txt_k = torch.zeros_like(txt_q)

    out = rope(vid_q, vid_k, vid_shape, txt_q, txt_k, txt_shape, seedvr_model.Cache(disable=True))

    assert [tuple(t.shape) for t in out] == [
        tuple(vid_q.shape),
        tuple(vid_k.shape),
        tuple(txt_q.shape),
        tuple(txt_k.shape),
    ]


def test_adasingle_init_preserves_supported_dtype():
    ada = seedvr_model.AdaSingle(
        dim=4,
        emb_dim=24,
        layers=["test"],
        modes=["in", "out"],
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert ada.test_shift.dtype is torch.bfloat16
    assert ada.test_scale.dtype is torch.bfloat16
    assert ada.test_gate.dtype is torch.bfloat16


def test_adasingle_init_uses_default_dtype_for_fp8():
    if not hasattr(torch, "float8_e4m3fn"):
        return

    ada = seedvr_model.AdaSingle(
        dim=4,
        emb_dim=24,
        layers=["test"],
        modes=["in", "out"],
        device="cpu",
        dtype=torch.float8_e4m3fn,
    )

    assert ada.test_shift.dtype is torch.float32
    assert ada.test_scale.dtype is torch.float32
    assert ada.test_gate.dtype is torch.float32


def test_adasingle_init_and_forward_share_fp8_dtype_set():
    expected = {
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float8_e5m2fnuz",
            "float8_e8m0fnu",
        )
        if hasattr(torch, name)
    }

    assert set(seedvr_model._torch_float8_types()) == expected

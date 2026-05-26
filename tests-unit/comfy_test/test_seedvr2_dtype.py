import inspect
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.ldm.modules.attention as attention
import comfy.sd
import comfy.supported_models
import comfy.ldm.seedvr.model as seedvr_model


def test_set_model_config_inference_dtype_preserves_legacy_signature():
    calls = []

    class LegacyConfig:
        def set_inference_dtype(self, dtype, manual_cast_dtype):
            calls.append((dtype, manual_cast_dtype))

    comfy.sd._set_model_config_inference_dtype(LegacyConfig(), torch.float16, None, object())

    assert calls == [(torch.float16, None)]


def test_set_model_config_inference_dtype_passes_device_when_supported():
    calls = []
    device = object()

    class DeviceAwareConfig:
        def set_inference_dtype(self, dtype, manual_cast_dtype, device=None):
            calls.append((dtype, manual_cast_dtype, device))

    comfy.sd._set_model_config_inference_dtype(DeviceAwareConfig(), torch.float16, None, device)

    assert calls == [(torch.float16, None, device)]


def test_set_model_config_inference_dtype_passes_device_to_kwargs_override():
    calls = []
    device = object()

    class KwargsConfig:
        def set_inference_dtype(self, dtype, manual_cast_dtype, **kwargs):
            calls.append((dtype, manual_cast_dtype, kwargs))

    comfy.sd._set_model_config_inference_dtype(KwargsConfig(), torch.float16, None, device)

    assert calls == [(torch.float16, None, {"device": device})]


def test_seedvr2_fp16_manual_cast_only_for_bf16_device(monkeypatch):
    bf16_device = object()
    fp16_device = object()

    monkeypatch.setattr(
        comfy.supported_models.comfy.model_management,
        "should_use_bf16",
        lambda device=None: device is bf16_device,
    )

    bf16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    bf16_config.set_inference_dtype(torch.float16, None, device=bf16_device)
    assert bf16_config.manual_cast_dtype is torch.bfloat16

    fp16_config = comfy.supported_models.SeedVR2({"image_model": "seedvr2"})
    fp16_config.set_inference_dtype(torch.float16, None, device=fp16_device)
    assert fp16_config.manual_cast_dtype is None


def test_apply_rope1_partial_preserves_full_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(8, dtype=torch.float16).reshape(1, 2, 4)
    original = t.clone()
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(out, (original.float() + 1.0).to(torch.float16))


def test_apply_rope1_partial_preserves_partial_rotation_input_dtype(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    t = torch.arange(12, dtype=torch.float16).reshape(1, 2, 6)
    original = t.clone()
    freqs_cis = torch.zeros(1, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert out.dtype is torch.float16
    torch.testing.assert_close(
        out[..., :4],
        (original[..., :4].float() + 1.0).to(torch.float16),
    )
    torch.testing.assert_close(out[..., 4:], original[..., 4:])


def test_apply_rope1_partial_chunks_sequence_dimension(monkeypatch):
    calls = []

    def fake_apply_rope1(t, freqs_cis):
        calls.append(t.shape[-2])
        return t.float() + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)
    monkeypatch.setattr(seedvr_model, "_ROPE1_PARTIAL_CHUNK_TOKENS", 2)

    t = torch.arange(30, dtype=torch.float16).reshape(1, 5, 6)
    original = t.clone()
    freqs_cis = torch.zeros(5, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)

    assert calls == [2, 2, 1]
    torch.testing.assert_close(out[..., :4], (original[..., :4].float() + 1.0).to(torch.float16))
    torch.testing.assert_close(out[..., 4:], original[..., 4:])


def test_apply_rope1_partial_clones_training_tensor(monkeypatch):
    def fake_apply_rope1(t, freqs_cis):
        return t + 1.0

    monkeypatch.setattr(seedvr_model, "apply_rope1", fake_apply_rope1)

    base = torch.arange(12, dtype=torch.float32, requires_grad=True)
    t = base.reshape(1, 2, 6)
    original = t.clone()
    freqs_cis = torch.zeros(2, 2, 2, 2)

    out = seedvr_model._apply_rope1_partial(t, freqs_cis)
    out.sum().backward()

    assert out is not t
    torch.testing.assert_close(t, original)
    torch.testing.assert_close(out[..., :4], original[..., :4] + 1.0)
    torch.testing.assert_close(out[..., 4:], original[..., 4:])
    assert base.grad is not None


def test_seedvr2_text_conditioning_accepts_cfg1_single_branch():
    context = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

    torch.testing.assert_close(txt, context.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3]], device=context.device))


def test_seedvr2_text_conditioning_accepts_batched_cfg1_single_branch():
    context = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0])

    torch.testing.assert_close(txt, context.flatten(0, -2))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_text_conditioning_accepts_multi_entry_cfg1_single_branch():
    context = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [0, 0])

    torch.testing.assert_close(txt, context.flatten(0, -2))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_text_conditioning_preserves_two_branch_swap_contract():
    neg = torch.full((1, 3, 2), -1.0)
    pos = torch.full((1, 3, 2), 1.0)
    context = torch.cat([neg, pos], dim=0)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context)

    torch.testing.assert_close(txt[:3], pos.squeeze(0))
    torch.testing.assert_close(txt[3:], neg.squeeze(0))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3]], device=context.device))


def test_seedvr2_text_conditioning_preserves_batched_two_branch_swap_contract():
    neg = torch.full((2, 3, 2), -1.0)
    pos = torch.full((2, 3, 2), 1.0)
    context = torch.cat([neg, pos], dim=0)

    txt, txt_shape = seedvr_model.NaDiT._resolve_text_conditioning(object(), context, [1, 0])

    torch.testing.assert_close(txt[:6], pos.flatten(0, -2))
    torch.testing.assert_close(txt[6:], neg.flatten(0, -2))
    torch.testing.assert_close(txt_shape, torch.tensor([[3], [3], [3], [3]], device=context.device))


def test_seedvr2_cfg1_single_branch_output_is_not_swapped():
    out = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    swapped = seedvr_model.NaDiT._swap_pos_neg_halves(object(), out, [0])

    torch.testing.assert_close(swapped, out)


def test_seedvr2_multi_entry_cfg1_output_is_not_swapped():
    out = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    swapped = seedvr_model.NaDiT._swap_pos_neg_halves(object(), out, [0, 0])

    torch.testing.assert_close(swapped, out)


def test_seedvr2_conditioning_keeps_comfy_cfg1_optimization_enabled():
    source = (Path(__file__).resolve().parents[2] / "comfy_extras" / "nodes_seedvr.py").read_text(encoding="utf-8")

    assert "disable_model_cfg1_optimization()" not in source


def test_seedvr2_split_var_attention_matches_nested_var_attention():
    torch.manual_seed(1)
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)

    torch_fx_logger = logging.getLogger("torch.fx._symbolic_trace")
    old_torch_fx_level = torch_fx_logger.level
    torch_fx_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The PyTorch API of nested tensors is in prototype stage.*",
                category=UserWarning,
            )
            nested = attention.var_attention_pytorch(
                q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                skip_reshape=True, skip_output_reshape=True,
            )
    finally:
        torch_fx_logger.setLevel(old_torch_fx_level)
    split = attention.var_attention_pytorch_split(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        skip_reshape=True, skip_output_reshape=True,
    )

    torch.testing.assert_close(split, nested, rtol=1e-5, atol=1e-5)


def test_seedvr2_split_var_attention_preserves_flat_output_shape():
    torch.manual_seed(2)
    q = torch.randn(5, 8)
    k = torch.randn(7, 8)
    v = torch.randn(7, 8)
    cu_q = torch.tensor([0, 1, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 7], dtype=torch.int32)

    nested = attention.var_attention_pytorch(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )
    split = attention.var_attention_pytorch_split(
        q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
    )

    assert split.shape == q.shape
    torch.testing.assert_close(split, nested, rtol=1e-5, atol=1e-5)


def test_seedvr2_split_var_attention_rejects_mismatched_sequence_count():
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    cu_q = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_k = torch.tensor([0, 3, 5, 7], dtype=torch.int32)

    try:
        attention.var_attention_pytorch_split(
            q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            skip_reshape=True, skip_output_reshape=True,
        )
    except ValueError as exc:
        assert "same sequence count" in str(exc)
    else:
        raise AssertionError("mismatched cu_seqlens sequence counts must fail")


def test_seedvr2_split_var_attention_rejects_malformed_offsets():
    q = torch.randn(5, 2, 4)
    k = torch.randn(7, 2, 4)
    v = torch.randn(7, 2, 4)
    cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)

    malformed_cases = (
        (torch.tensor([1, 2, 5], dtype=torch.int32), "start at 0"),
        (torch.tensor([0, 2, 2, 5], dtype=torch.int32), "strictly increasing"),
        (torch.tensor([0.0, 2.0, 5.0], dtype=torch.float32), "integer dtype"),
    )

    for cu_q, message in malformed_cases:
        try:
            attention.var_attention_pytorch_split(
                q, k, v, heads=2, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                skip_reshape=True, skip_output_reshape=True,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("malformed cu_seqlens must fail")


def test_seedvr2_7b_window_attention_handles_mm_rope_source():
    source = inspect.getsource(seedvr_model.NaSwinAttention.forward)

    assert "if self.rope.mm" in source
    assert "txt_q_repeat" in source


def test_seedvr2_7b_window_attention_routes_to_split_var_attention():
    source = inspect.getsource(seedvr_model.NaSwinAttention.forward)

    assert "_seedvr2_7b_window_attention_split" in source
    assert "if self.version_7b" in source


def test_seedvr2_7b_window_attention_split_matches_concat_path():
    torch.manual_seed(3)
    vid_len_win = torch.tensor([1, 2, 3], dtype=torch.int64)
    txt_len = torch.tensor([2, 3], dtype=torch.int64)
    window_count = torch.tensor([2, 1], dtype=torch.int64)
    heads = 2
    dim = 4

    vid_total = int(vid_len_win.sum().item())
    txt_total = int(txt_len.sum().item())
    vid_q = torch.randn(vid_total, heads, dim)
    vid_k = torch.randn(vid_total, heads, dim)
    vid_v = torch.randn(vid_total, heads, dim)
    txt_q = torch.randn(txt_total, heads, dim)
    txt_k = torch.randn(txt_total, heads, dim)
    txt_v = torch.randn(txt_total, heads, dim)

    concat_win, unconcat_win = seedvr_model.repeat_concat_idx(vid_len_win, txt_len, window_count)
    all_len_win = vid_len_win + txt_len.repeat_interleave(window_count)
    cu_seqlens = torch.nn.functional.pad(all_len_win.cumsum(0), (1, 0)).int()
    concat_out = attention.var_attention_pytorch_split(
        concat_win(vid_q, txt_q),
        concat_win(vid_k, txt_k),
        concat_win(vid_v, txt_v),
        heads=heads,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        skip_reshape=True,
        skip_output_reshape=True,
    )
    expected_vid, expected_txt = unconcat_win(concat_out)

    split_vid, split_txt = seedvr_model._seedvr2_7b_window_attention_split(
        vid_q, txt_q, vid_k, txt_k, vid_v, txt_v,
        vid_len_win, txt_len, window_count,
    )

    torch.testing.assert_close(split_vid, expected_vid, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(split_txt, expected_txt, rtol=1e-5, atol=1e-5)


def test_seedvr2_7b_window_attention_split_preserves_autograd():
    torch.manual_seed(4)
    vid_len_win = torch.tensor([1, 2, 3], dtype=torch.int64)
    txt_len = torch.tensor([2, 3], dtype=torch.int64)
    window_count = torch.tensor([2, 1], dtype=torch.int64)
    heads = 2
    dim = 4

    vid_total = int(vid_len_win.sum().item())
    txt_total = int(txt_len.sum().item())
    vid_q = torch.randn(vid_total, heads, dim, requires_grad=True)
    vid_k = torch.randn(vid_total, heads, dim, requires_grad=True)
    vid_v = torch.randn(vid_total, heads, dim, requires_grad=True)
    txt_q = torch.randn(txt_total, heads, dim, requires_grad=True)
    txt_k = torch.randn(txt_total, heads, dim, requires_grad=True)
    txt_v = torch.randn(txt_total, heads, dim, requires_grad=True)

    split_vid, split_txt = seedvr_model._seedvr2_7b_window_attention_split(
        vid_q, txt_q, vid_k, txt_k, vid_v, txt_v,
        vid_len_win, txt_len, window_count,
    )
    (split_vid.sum() + split_txt.sum()).backward()

    for tensor in (vid_q, vid_k, vid_v, txt_q, txt_k, txt_v):
        assert tensor.grad is not None


def test_seedvr2_7b_mlp_chunks_video_tokens(monkeypatch):
    class TrackingModule(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale
            self.calls = []

        def forward(self, x):
            self.calls.append(x.shape[0])
            return x * self.scale

    monkeypatch.setattr(seedvr_model, "SEEDVR2_7B_MLP_CHUNK", 2)

    vid_module = TrackingModule(2.0)
    txt_module = TrackingModule(3.0)
    block = SimpleNamespace(
        mlp=SimpleNamespace(
            shared_weights=False,
            vid_only=False,
            vid=vid_module,
            txt=txt_module,
        )
    )
    vid = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    txt = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    out_vid, out_txt = seedvr_model.NaMMSRTransformerBlock._seedvr2_7b_mlp(block, vid, txt)

    assert vid_module.calls == [2, 2, 2]
    assert txt_module.calls == [3]
    torch.testing.assert_close(out_vid, vid * 2.0)
    torch.testing.assert_close(out_txt, txt * 3.0)


def test_seedvr2_7b_mlp_preserves_video_autograd(monkeypatch):
    class TrackingModule(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    monkeypatch.setattr(seedvr_model, "SEEDVR2_7B_MLP_CHUNK", 2)

    block = SimpleNamespace(
        mlp=SimpleNamespace(
            shared_weights=False,
            vid_only=True,
            vid=TrackingModule(),
        )
    )
    vid_base = torch.arange(24, dtype=torch.float32, requires_grad=True)
    vid = vid_base.reshape(6, 4)
    txt = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    out_vid, _ = seedvr_model.NaMMSRTransformerBlock._seedvr2_7b_mlp(block, vid, txt)
    out_vid.sum().backward()

    assert vid_base.grad is not None


def test_seedvr2_7b_block_routes_mlp_to_chunk_helper():
    source = inspect.getsource(seedvr_model.NaMMSRTransformerBlock.forward)

    assert "if self.version" in source
    assert "_seedvr2_7b_mlp" in source


def test_seedvr2_vae_decode_memory_covers_full_frame_lab_transfer():
    estimate = comfy.sd._seedvr2_vae_decode_memory_used((1, 16, 26, 120, 160))
    old_estimate = 16 * 120 * 160 * (4 * 8 * 8) * 2

    assert estimate == 101 * 960 * 1280 * 160
    assert estimate > 15 * 1024 ** 3
    assert estimate > old_estimate * 100


def test_seedvr2_vae_decode_memory_estimate_is_per_sample():
    single = comfy.sd._seedvr2_vae_decode_memory_used((1, 16, 26, 120, 160))
    batch = comfy.sd._seedvr2_vae_decode_memory_used((2, 16, 26, 120, 160))

    assert batch == single


def test_seedvr2_vae_decode_memory_accepts_channel_last_tiled_latents():
    channel_first = comfy.sd._seedvr2_vae_decode_memory_used((1, 16, 26, 120, 160))
    channel_last = comfy.sd._seedvr2_vae_decode_memory_used((1, 26, 120, 160, 16))

    assert channel_last == channel_first


def test_seedvr2_vae_decode_memory_rounds_malformed_collapsed_channels_up():
    malformed = comfy.sd._seedvr2_vae_decode_memory_used((1, 17, 120, 160))
    expected = comfy.sd._seedvr2_vae_decode_output_pixels(2, 120, 160) * comfy.sd.SEEDVR2_VAE_DECODE_BYTES_PER_OUTPUT_PIXEL

    assert malformed == expected


def test_seedvr2_vae_decode_memory_uses_conservative_ambiguous_5d_layout():
    ambiguous = comfy.sd._seedvr2_vae_decode_memory_used((1, 16, 120, 160, 16))
    channel_first = comfy.sd._seedvr2_vae_decode_output_pixels(120, 160, 16) * comfy.sd.SEEDVR2_VAE_DECODE_BYTES_PER_OUTPUT_PIXEL
    channel_last = comfy.sd._seedvr2_vae_decode_output_pixels(16, 120, 160) * comfy.sd.SEEDVR2_VAE_DECODE_BYTES_PER_OUTPUT_PIXEL

    assert ambiguous == max(channel_first, channel_last)

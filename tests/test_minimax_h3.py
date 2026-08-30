import os
import unittest
from unittest.mock import MagicMock, patch

import torch

import comfy.ldm.minimax.model as m
import comfy.ldm.minimax.audio_vae as avae
import comfy.model_detection as md


class TestMiniMaxH3Optimizations(unittest.TestCase):
    def test_patchify_unpatchify_roundtrip(self):
        latent = torch.randn(1, 24, 5, 16, 16)
        patched = m.patchify_video(latent)
        unpatched = m.unpatchify_video(patched, 5, 8, 8, 24)
        self.assertTrue(torch.allclose(latent, unpatched))

    def test_snake_activation(self):
        x = torch.randn(2, 32, 100)
        alpha = torch.randn(1, 32, 1)
        beta = torch.randn(1, 32, 1).abs() + 0.1
        out = avae.snake(x, alpha, beta)
        self.assertEqual(out.shape, x.shape)

    def test_model_forward_with_cached_layout_and_tensor_only_replacement(self):
        model = m.MiniMaxH3Model(
            hidden_size=256,
            num_layers=2,
            num_attention_heads=4,
            attention_head_dim=128,
            ffn_hidden_size=512,
            time_embed_hidden_size=256,
            time_embed_dim=256,
            operations=m.comfy.ops.manual_cast,
        )
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.fill_(1.0)

        x_video = torch.randn(1, 24, 2, 8, 8)
        x_audio = torch.randn(1, 32, 2, 10)
        context = torch.randn(1, 16, 256)
        timestep = torch.tensor([500.0])

        layout = m.PackedLayout(context.shape[1], x_video.shape[2], x_video.shape[3], x_video.shape[4], x_audio.shape[-1])
        payload = {"layout": layout}

        def tensor_only_replacement(args, extra):
            args = {key: value for key, value in args.items() if key != "mod_segments"}
            return extra["original_block"](args)

        transformer_options = {"patches_replace": {"dit": {("double_block", 0): tensor_only_replacement}}}
        with (
            patch.object(m, "PackedLayout", wraps=m.PackedLayout) as packed_layout,
            patch.object(model, "rope_freqs", wraps=model.rope_freqs) as rope_freqs,
            patch.object(m.comfy.quant_ops.ck, "rms_rope_split_half_"),
        ):
            out1 = model([x_video, x_audio], timestep, context, transformer_options=transformer_options, minimax_payload=payload)
            out2 = model([x_video, x_audio], timestep, context, transformer_options=transformer_options, minimax_payload=payload)

        packed_layout.assert_not_called()
        self.assertEqual(rope_freqs.call_count, 2)
        self.assertTrue(all(call.args[0] is layout.position_ids for call in rope_freqs.call_args_list))
        self.assertIs(payload["layout"], layout)
        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-5))
        self.assertTrue(torch.allclose(out1[1], out2[1], atol=1e-5))

    def test_tensor_segment_modulation_matches_tuple_path(self):
        hidden = 16
        norm = m.comfy.ops.disable_weight_init.RMSNorm(hidden, eps=1e-6)
        norm.weight.detach().copy_(torch.linspace(0.5, 1.5, hidden))
        x = torch.randn(12, hidden)
        shift = torch.randn(3, hidden)
        scale = torch.randn(3, hidden)
        gate = torch.randn(3, hidden)
        other = torch.randn_like(x)
        segments = [(0, 4, 0), (4, 9, 1), (9, 12, 2)]
        segment_ids = torch.tensor([0] * 4 + [1] * 5 + [2] * 3)

        tuple_out = m._mod_scale_shift(norm(x), shift, scale, segments)
        tensor_out = m._rms_adaln(norm, x, shift, scale, segment_ids)
        tuple_gated = m._mod_gate(x.clone(), gate, other, segments)
        tensor_gated = m._mod_gate(x.clone(), gate, other, segment_ids)

        self.assertEqual(tensor_out.shape, x.shape)
        self.assertEqual(tensor_out.dtype, x.dtype)
        self.assertEqual(tensor_out.device, x.device)
        self.assertTrue(torch.allclose(tuple_out, tensor_out, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(tuple_gated, tensor_gated, rtol=1e-5, atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP16 modulation coverage")
    def test_tensor_segment_modulation_cuda_fp16(self):
        device = torch.device("cuda")
        dtype = torch.float16
        seq_len = 4096
        hidden = 1536
        num_segments = 16
        segment_len = seq_len // num_segments
        segments = [(i * segment_len, (i + 1) * segment_len, i) for i in range(num_segments)]
        segment_ids = torch.arange(num_segments, device=device).repeat_interleave(segment_len)
        norm = m.comfy.ops.disable_weight_init.RMSNorm(hidden, eps=1e-6, dtype=dtype, device=device)
        norm.weight.detach().copy_(torch.linspace(0.5, 1.5, hidden, device=device, dtype=dtype))
        x = torch.randn(seq_len, hidden, device=device, dtype=dtype)
        shift = torch.randn(num_segments, hidden, device=device, dtype=dtype) * 0.1
        scale = torch.randn(num_segments, hidden, device=device, dtype=dtype) * 0.1
        gate = torch.randn(num_segments, hidden, device=device, dtype=dtype) * 0.1
        other = torch.randn_like(x)

        tuple_out = m._mod_scale_shift(norm(x), shift, scale, segments)
        tensor_out = m._rms_adaln(norm, x, shift, scale, segment_ids)
        tuple_gated = m._mod_gate(x.clone(), gate, other, segments)
        tensor_gated = m._mod_gate(x.clone(), gate, other, segment_ids)

        self.assertEqual(tensor_out.shape, x.shape)
        self.assertEqual(tensor_out.dtype, dtype)
        self.assertEqual(tensor_out.device, x.device)
        self.assertTrue(torch.allclose(tuple_out, tensor_out, rtol=2e-3, atol=2e-3))
        self.assertTrue(torch.allclose(tuple_gated, tensor_gated, rtol=2e-3, atol=2e-3))

        def benchmark(fn):
            for _ in range(2):
                fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 10

        tuple_ms = benchmark(lambda: m._mod_scale_shift(norm(x), shift, scale, segments))
        tensor_ms = benchmark(lambda: m._rms_adaln(norm, x, shift, scale, segment_ids))
        self.assertGreater(tuple_ms, 0.0)
        self.assertGreater(tensor_ms, 0.0)

    def test_quantized_model_detection(self):
        quant_weight = MagicMock()
        quant_weight.shape = torch.Size([4608, 768])
        quant_weight.tensor_shape = torch.Size([4608, 1536])

        state_dict = {
            "video_patch_proj.weight": torch.empty(1536, 96),
            "audio_patch_proj.weight": torch.empty(1536, 64),
            "final_layer.video_out.weight": torch.empty(96, 1536),
            "final_layer.audio_out.weight": torch.empty(64, 1536),
            "blocks.0.attn.q_norm.weight": torch.empty(128),
            "blocks.0.attn.qkv_proj.weight": quant_weight,
            "blocks.0.mlp.fc1.weight": torch.empty(8192, 1536),
            "condition_proj.weight": torch.empty(1536, 4096),
            "time_embedder.proj_in.weight": torch.empty(512, 256),
            "time_embedder.proj_out.weight": torch.empty(512, 512),
            "rope.inv_freq": torch.empty(16),
        }
        config = md.detect_unet_config(state_dict, "")
        self.assertIsNotNone(config)
        self.assertEqual(config.get("image_model"), "minimax_h3")
        self.assertEqual(config.get("hidden_size"), 1536)
        self.assertEqual(config.get("text_dim"), 4096)


if __name__ == "__main__":
    unittest.main()

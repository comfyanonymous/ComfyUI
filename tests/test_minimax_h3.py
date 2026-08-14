import os
import unittest
from unittest.mock import MagicMock

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

    def test_model_forward_and_caching(self):
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
            p.requires_grad = False

        x_video = torch.randn(1, 24, 2, 8, 8)
        x_audio = torch.randn(1, 32, 2, 10)
        context = torch.randn(1, 16, 256)
        timestep = torch.tensor([500.0])

        payload = {}
        out1 = model([x_video, x_audio], timestep, context, minimax_payload=payload)
        out2 = model([x_video, x_audio], timestep, context, minimax_payload=payload)

        self.assertTrue(torch.allclose(out1[0], out2[0], atol=1e-5))
        self.assertTrue(torch.allclose(out1[1], out2[1], atol=1e-5))

    def test_quantized_model_detection(self):
        # Mock NVFP4 4-bit packed linear weight (shape[1] is 768, tensor_shape[1] is 1536)
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

        # Test key-guarding when optional keys (condition_proj, time_embedder, rope) are omitted
        minimal_state_dict = {
            "video_patch_proj.weight": torch.empty(1536, 96),
            "audio_patch_proj.weight": torch.empty(1536, 64),
            "final_layer.video_out.weight": torch.empty(96, 1536),
            "final_layer.audio_out.weight": torch.empty(64, 1536),
            "blocks.0.attn.q_norm.weight": torch.empty(128),
            "blocks.0.attn.qkv_proj.weight": quant_weight,
            "blocks.0.mlp.fc1.weight": torch.empty(8192, 1536),
        }
        config_min = md.detect_unet_config(minimal_state_dict, "")
        self.assertIsNotNone(config_min)
        self.assertEqual(config_min.get("image_model"), "minimax_h3")
        self.assertEqual(config_min.get("hidden_size"), 1536)
        self.assertIsNone(config_min.get("text_dim"))


if __name__ == "__main__":
    unittest.main()

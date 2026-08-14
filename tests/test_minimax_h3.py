import unittest
from unittest.mock import MagicMock

import torch

import comfy.model_detection as md


class TestMiniMaxH3ModelDetection(unittest.TestCase):
    def test_quantized_model_detection(self):
        packed_condition_weight = MagicMock()
        packed_condition_weight.shape = torch.Size([1536, 2048])
        packed_condition_weight.tensor_shape = torch.Size([1536, 4096])

        state_dict = {
            "video_patch_proj.weight": torch.empty(1536, 96),
            "audio_patch_proj.weight": torch.empty(1536, 64),
            "final_layer.video_out.weight": torch.empty(96, 1536),
            "final_layer.audio_out.weight": torch.empty(64, 1536),
            "blocks.0.attn.q_norm.weight": torch.empty(128),
            "blocks.0.attn.qkv_proj.weight": torch.empty(4608, 768),
            "blocks.0.mlp.fc1.weight": torch.empty(8192, 1536),
            "condition_proj.weight": packed_condition_weight,
            "time_embedder.proj_in.weight": torch.empty(512, 256),
            "time_embedder.proj_out.weight": torch.empty(512, 512),
            "rope.inv_freq": torch.empty(16),
        }
        config = md.detect_unet_config(state_dict, "")
        self.assertIsNotNone(config)
        self.assertEqual(config.get("image_model"), "minimax_h3")
        self.assertEqual(config.get("hidden_size"), 1536)
        self.assertEqual(config.get("text_dim"), 4096)
        self.assertNotEqual(packed_condition_weight.shape[1], config["text_dim"])

        omission_cases = {
            "condition_proj": (
                ("condition_proj.weight",),
                ("text_dim",),
            ),
            "time_embedder_proj_in": (
                ("time_embedder.proj_in.weight",),
                ("timestep_input_dim", "time_embed_hidden_size"),
            ),
            "time_embedder_proj_out": (
                ("time_embedder.proj_out.weight",),
                ("time_embed_dim",),
            ),
            "time_embedder": (
                ("time_embedder.proj_in.weight", "time_embedder.proj_out.weight"),
                ("timestep_input_dim", "time_embed_hidden_size", "time_embed_dim"),
            ),
            "rope": (
                ("rope.inv_freq",),
                ("rope_inv_freq_len",),
            ),
        }
        for name, (missing_keys, missing_config) in omission_cases.items():
            with self.subTest(name=name):
                incomplete_state_dict = dict(state_dict)
                for key in missing_keys:
                    incomplete_state_dict.pop(key)
                incomplete_config = md.detect_unet_config(incomplete_state_dict, "")
                self.assertEqual(incomplete_config.get("image_model"), "minimax_h3")
                self.assertEqual(incomplete_config.get("hidden_size"), 1536)
                for key in missing_config:
                    self.assertNotIn(key, incomplete_config)


if __name__ == "__main__":
    unittest.main()

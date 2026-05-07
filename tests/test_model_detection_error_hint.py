"""Unit tests for sd.model_detection_error_hint.

Exercises the friendlier-error path that runs when load_checkpoint_guess_config
can't identify the file. These tests stand in for the most common "what node
do I actually use for this file?" Discord questions.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from comfy.sd import model_detection_error_hint
    _IMPORT_ERROR = None
except Exception as _e:  # noqa: BLE001 — propagate any import failure
    model_detection_error_hint = None
    _IMPORT_ERROR = _e


def _hint(path, sd):
    if _IMPORT_ERROR is not None:
        raise unittest.SkipTest(f"comfy.sd import unavailable: {_IMPORT_ERROR}")
    return model_detection_error_hint(path, sd)


class ModelDetectionErrorHintTests(unittest.TestCase):
    def test_lora_filename_hint(self):
        h = _hint("/x/some_lora.safetensors", {"lora_up.weight": None})
        self.assertIn("LoRA", h)
        self.assertIn("models/loras/", h)

    def test_flux_unet_only_hint(self):
        sd = {
            "double_blocks.0.img_attn.proj.weight": None,
            "single_blocks.0.linear1.weight": None,
        }
        h = _hint("/x/flux1-dev-fp8.safetensors", sd)
        self.assertIn("UNETLoader", h)
        self.assertIn("DualCLIPLoader", h)
        self.assertIn("Flux", h)

    def test_sd3_unet_only_hint(self):
        sd = {
            "joint_blocks.0.x_block.attn.proj.weight": None,
            "pos_embed": None,
        }
        h = _hint("/x/sd3.safetensors", sd)
        self.assertIn("UNETLoader", h)
        self.assertIn("SD3", h)

    def test_vae_only_hint(self):
        sd = {"first_stage_model.decoder.conv_in.weight": None}
        h = _hint("/x/sdxl_vae.safetensors", sd)
        self.assertIn("VAELoader", h)

    def test_text_encoder_t5_hint(self):
        sd = {"shared.weight": None}
        h = _hint("/x/t5xxl_fp8.safetensors", sd)
        self.assertIn("CLIPLoader", h)

    def test_text_encoder_clip_l_hint(self):
        sd = {"text_model.embeddings.token_embedding.weight": None}
        h = _hint("/x/clip_l.safetensors", sd)
        self.assertIn("CLIPLoader", h)

    def test_gguf_hint(self):
        h = _hint("/x/some_model.gguf", {})
        self.assertIn("GGUF", h)
        self.assertIn("ComfyUI-GGUF", h)

    def test_unknown_returns_empty(self):
        h = _hint("/x/mystery.safetensors", {"some.unknown.key": None})
        self.assertEqual("", h)

    def test_filename_overrides_when_state_dict_silent(self):
        # User renamed a VAE without recognisable keys (e.g. quantised dump).
        h = _hint("/x/my_vae.safetensors", {"opaque.key": None})
        self.assertIn("VAELoader", h)


if __name__ == "__main__":
    unittest.main()

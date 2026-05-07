"""Unit tests for comfy_execution.workflow_to_prompt.

Stubs nodes.NODE_CLASS_MAPPINGS so the conversion logic can be exercised
without booting the entire ComfyUI module graph (torch, model_management,
etc.).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _FakeNode:
    """Minimal stand-in for a NODE_CLASS_MAPPINGS entry."""
    def __init__(self, input_types):
        self._inputs = input_types

    def INPUT_TYPES(self):
        return self._inputs


# --- Test fixtures ---------------------------------------------------------

KSAMPLER = _FakeNode({
    "required": {
        "model": ("MODEL",),
        "seed": ("INT", {"default": 0}),
        "steps": ("INT", {"default": 20}),
        "cfg": ("FLOAT", {"default": 8.0}),
        "sampler_name": (["euler", "dpmpp_2m"], ),
        "scheduler": (["normal", "karras"], ),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "latent_image": ("LATENT",),
        "denoise": ("FLOAT", {"default": 1.0}),
    },
})

CHECKPOINT = _FakeNode({
    "required": {
        "ckpt_name": (["model.safetensors"], ),
    },
})

CLIPTEXT = _FakeNode({
    "required": {
        "text": ("STRING", {"multiline": True}),
        "clip": ("CLIP",),
    },
})


def _stub_modules():
    """Replace nodes / heavy imports so workflow_to_prompt can import."""
    fake_nodes = mock.MagicMock()
    fake_nodes.NODE_CLASS_MAPPINGS = {
        "KSampler": KSAMPLER,
        "CheckpointLoaderSimple": CHECKPOINT,
        "CLIPTextEncode": CLIPTEXT,
    }
    return {"nodes": fake_nodes}


def _import_under_stub():
    with mock.patch.dict(sys.modules, _stub_modules()):
        sys.modules.pop("comfy_execution.workflow_to_prompt", None)
        from comfy_execution import workflow_to_prompt  # noqa: F401
        return workflow_to_prompt


class WorkflowToPromptTests(unittest.TestCase):
    def setUp(self):
        self.mod = _import_under_stub()
        # Re-stub for each call too — workflow_to_prompt re-reads nodes
        # at call time via attribute lookup.
        self._patcher = mock.patch.dict(sys.modules, _stub_modules())
        self._patcher.start()
        # Re-import a fresh module reference that has the patched nodes baked
        # into its closure.
        sys.modules.pop("comfy_execution.workflow_to_prompt", None)
        from comfy_execution import workflow_to_prompt
        self.fn = workflow_to_prompt.workflow_to_prompt

    def tearDown(self):
        self._patcher.stop()

    def test_widget_only_node(self):
        wf = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["v1-5-pruned.ckpt"], "inputs": []},
            ],
            "links": [],
        }
        out = self.fn(wf)
        self.assertEqual(out, {"1": {"class_type": "CheckpointLoaderSimple",
                                       "inputs": {"ckpt_name": "v1-5-pruned.ckpt"}}})

    def test_int_seed_consumes_extra_widget_value(self):
        # KSampler widget order: seed, control_after_generate (implicit),
        # steps, cfg, sampler_name, scheduler, denoise. (8 widget values)
        wf = {
            "nodes": [
                {"id": 7, "type": "KSampler",
                 "widgets_values": [12345, "fixed", 25, 7.5, "euler", "karras", 0.8],
                 "inputs": [
                     {"name": "model", "type": "MODEL", "link": 1},
                     {"name": "positive", "type": "CONDITIONING", "link": 2},
                     {"name": "negative", "type": "CONDITIONING", "link": 3},
                     {"name": "latent_image", "type": "LATENT", "link": 4},
                 ]},
            ],
            "links": [
                [1, 100, 0, 7, 0, "MODEL"],
                [2, 101, 0, 7, 1, "CONDITIONING"],
                [3, 102, 0, 7, 2, "CONDITIONING"],
                [4, 103, 0, 7, 3, "LATENT"],
            ],
        }
        out = self.fn(wf)
        self.assertEqual(out["7"]["inputs"]["seed"], 12345)
        self.assertEqual(out["7"]["inputs"]["steps"], 25)
        self.assertEqual(out["7"]["inputs"]["cfg"], 7.5)
        self.assertEqual(out["7"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(out["7"]["inputs"]["scheduler"], "karras")
        self.assertEqual(out["7"]["inputs"]["denoise"], 0.8)
        self.assertEqual(out["7"]["inputs"]["model"], ["100", 0])

    def test_link_resolution(self):
        wf = {
            "nodes": [
                {"id": 5, "type": "CLIPTextEncode",
                 "widgets_values": ["a cat"],
                 "inputs": [{"name": "clip", "type": "CLIP", "link": 9}]},
                {"id": 1, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["m.ckpt"], "inputs": []},
            ],
            "links": [[9, 1, 1, 5, 0, "CLIP"]],  # checkpoint clip out -> textenc clip in
        }
        out = self.fn(wf)
        self.assertEqual(out["5"]["inputs"]["text"], "a cat")
        self.assertEqual(out["5"]["inputs"]["clip"], ["1", 1])

    def test_bypass_and_never_nodes_skipped(self):
        # Both BYPASS (mode=4) and NEVER (mode=2) match frontend semantics
        # of "don't ship to executor" — confirm both are filtered.
        wf = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["bypass.ckpt"], "inputs": [], "mode": 4},
                {"id": 2, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["never.ckpt"], "inputs": [], "mode": 2},
                {"id": 3, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["always.ckpt"], "inputs": [], "mode": 0},
            ],
            "links": [],
        }
        out = self.fn(wf)
        self.assertNotIn("1", out)
        self.assertNotIn("2", out)
        self.assertIn("3", out)

    def test_frontend_only_nodes_skipped(self):
        wf = {
            "nodes": [
                {"id": 1, "type": "Note", "widgets_values": ["hi"], "inputs": []},
                {"id": 2, "type": "Reroute", "widgets_values": [], "inputs": []},
                {"id": 3, "type": "PrimitiveNode", "widgets_values": [42], "inputs": []},
                {"id": 4, "type": "CheckpointLoaderSimple",
                 "widgets_values": ["x.ckpt"], "inputs": []},
            ],
            "links": [],
        }
        out = self.fn(wf)
        self.assertEqual(set(out.keys()), {"4"})

    def test_unknown_class_skipped(self):
        wf = {"nodes": [{"id": 1, "type": "DefinitelyNotARealNode",
                          "widgets_values": [], "inputs": []}],
              "links": []}
        out = self.fn(wf)
        self.assertEqual(out, {})

    def test_widget_converted_to_input(self):
        # ckpt_name has been promoted from widget to input slot — no widget
        # value present, but a link exists.
        wf = {
            "nodes": [
                {"id": 2, "type": "CheckpointLoaderSimple",
                 "widgets_values": [],
                 "inputs": [{"name": "ckpt_name", "type": "STRING", "link": 1}]},
            ],
            "links": [[1, 99, 0, 2, 0, "STRING"]],
        }
        out = self.fn(wf)
        self.assertEqual(out["2"]["inputs"]["ckpt_name"], ["99", 0])

    def test_empty_workflow(self):
        self.assertEqual(self.fn({}), {})
        self.assertEqual(self.fn({"nodes": [], "links": []}), {})

    def test_dict_form_links(self):
        # Some frontend versions emit links as dicts rather than tuples.
        wf = {
            "nodes": [
                {"id": 5, "type": "CLIPTextEncode",
                 "widgets_values": ["a"],
                 "inputs": [{"name": "clip", "type": "CLIP", "link": 1}]},
            ],
            "links": [{"id": 1, "origin_id": 9, "origin_slot": 1,
                       "target_id": 5, "target_slot": 0, "type": "CLIP"}],
        }
        out = self.fn(wf)
        self.assertEqual(out["5"]["inputs"]["clip"], ["9", 1])


if __name__ == "__main__":
    unittest.main()

import unittest
import torch
import sys
import os
import json

# Add comfy to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def has_gpu():
    return torch.cuda.is_available()

from comfy.cli_args import args
if not has_gpu():
    args.cpu = True

from comfy import ops
from comfy.quant_ops import QuantizedTensor
import comfy.utils
import comfy.model_detection as model_detection


def marker_json(state_dict, key):
    """Decode a `<key>.comfy_quant` marker tensor back into its layer_conf dict."""
    return json.loads(state_dict[key].numpy().tobytes())


class SimpleModel(torch.nn.Module):
    """Mirrors tests-unit/comfy_quant/test_mixed_precision.py::SimpleModel."""

    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        return self.layer1(x)


class TestConvertOldQuantsPrefixAware(unittest.TestCase):
    """Regression tests for GitHub #11864 / #13328: convert_old_quants()'s
    new-format (_quantization_metadata) branch must match the layer key
    convention actually used by the state_dict it is given, regardless of
    whether that convention is prefixed or already stripped, and regardless
    of whether model_prefix is stripped/added/empty at the call site.
    """

    # ---- scenario 1: metadata key and sd key both carry the prefix (aligned) ----
    def test_scenario1_prefixed_metadata_matches_prefixed_sd(self):
        layer_quant_config = {"model.diffusion_model.proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "model.diffusion_model.proj_in.weight_scale": torch.tensor(1.0),
            "model.diffusion_model.other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        self.assertIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertNotIn("proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "model.diffusion_model.proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- scenario 2: sd already stripped of prefix, metadata key still prefixed (the bug) ----
    def test_scenario2_prefixed_metadata_matches_stripped_sd_after_fix(self):
        layer_quant_config = {"model.diffusion_model.proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
            "other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        # Before the fix this would have blindly written
        # "model.diffusion_model.proj_in.comfy_quant", which never matches
        # "proj_in.weight" -> detect_layer_quantization()/MixedPrecisionOps
        # would find no marker and load the layer as a plain dtype tensor.
        self.assertIn("proj_in.comfy_quant", out_sd)
        self.assertNotIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- scenario 3: metadata key already stripped, sd already stripped (today's working path) ----
    def test_scenario3_stripped_metadata_matches_stripped_sd_unchanged(self):
        layer_quant_config = {"proj_in": {"format": "float8_e4m3fn"}}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        # Zero behavior change vs. today: direct match succeeds immediately,
        # key is written exactly as it always was.
        self.assertIn("proj_in.comfy_quant", out_sd)
        self.assertEqual(marker_json(out_sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # Same as scenario 3 but with model_prefix="" (the literal value passed
    # by comfy/sd.py::load_diffusion_model_state_dict at both call sites).
    def test_scenario3b_stripped_metadata_empty_model_prefix_unchanged(self):
        layer_quant_config = {"proj_in": {"format": "nvfp4"}}
        state_dict = {
            "proj_in.weight": torch.randint(0, 255, (4, 2), dtype=torch.uint8),
            "proj_in.weight_scale": torch.tensor(1.0),
            "proj_in.weight_scale_2": torch.tensor(1.0),
        }
        out_sd, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        self.assertIn("proj_in.comfy_quant", out_sd)

    # ---- scenario 4: legacy scaled_fp8 branch must be completely unaffected ----
    def test_scenario4_legacy_scaled_fp8_branch_unaffected(self):
        state_dict = {
            "model.diffusion_model.scaled_fp8": torch.tensor([0.0], dtype=torch.float32),
            "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "model.diffusion_model.proj_in.scale_weight": torch.tensor(2.0),
            "model.diffusion_model.other.weight": torch.randn(4, 4, dtype=torch.bfloat16),
        }
        out_sd, metadata = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={},
        )
        # Old-format branch derives layer keys straight from state_dict's own
        # (already correctly prefixed) keys, so resolution is a same-key
        # direct match every time -- this path must be byte-for-byte identical
        # to pre-fix behavior.
        self.assertNotIn("model.diffusion_model.scaled_fp8", out_sd)
        self.assertIn("model.diffusion_model.proj_in.weight_scale", out_sd)
        self.assertIn("model.diffusion_model.proj_in.comfy_quant", out_sd)
        self.assertEqual(
            marker_json(out_sd, "model.diffusion_model.proj_in.comfy_quant")["format"],
            "float8_e4m3fn",
        )
        self.assertNotIn("proj_in.comfy_quant", out_sd)  # not stripped/mismatched

    # ---- extra: idempotency across the exact two-call pattern comfy/sd.py uses ----
    def test_two_call_pattern_mirrors_load_diffusion_model_state_dict(self):
        """Simulates comfy/sd.py::load_diffusion_model_state_dict() verbatim:
        convert_old_quants(sd, "", metadata=metadata) is called once before
        the diffusion_model_prefix strip and once after, both with an empty
        model_prefix. This must work for BOTH metadata conventions without
        any change to the call site (see PR #13328, closed for reordering
        the calls and breaking the other convention instead)."""
        for convention, layer_key in (
            ("prefixed", "model.diffusion_model.proj_in"),
            ("stripped", "proj_in"),
        ):
            with self.subTest(convention=convention):
                metadata = {"_quantization_metadata": json.dumps(
                    {"layers": {layer_key: {"format": "float8_e4m3fn"}}}
                )}
                sd = {
                    "model.diffusion_model.proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
                    "model.diffusion_model.proj_in.weight_scale": torch.tensor(1.0),
                    "unrelated.top_level.weight": torch.randn(2, 2, dtype=torch.bfloat16),
                }

                # call 1: before stripping, model_prefix="" (as sd.py does)
                sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                # simulate state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
                prefix = "model.diffusion_model."
                temp_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
                self.assertGreater(len(temp_sd), 0)
                sd = temp_sd

                # call 2: after stripping, model_prefix="" again (as sd.py does)
                sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

                self.assertIn("proj_in.comfy_quant", sd,
                               f"{convention} metadata convention did not resolve after the two-call dance")
                self.assertEqual(marker_json(sd, "proj_in.comfy_quant")["format"], "float8_e4m3fn")

    # ---- extra: repeated calls with identical inputs don't duplicate/clobber ----
    def test_marker_write_is_idempotent(self):
        layer_quant_config = {"proj_in": {"format": "float8_e4m3fn"}}
        metadata = {"_quantization_metadata": json.dumps({"layers": layer_quant_config})}
        state_dict = {
            "proj_in.weight": torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn),
            "proj_in.weight_scale": torch.tensor(1.0),
        }
        out_sd1, _ = comfy.utils.convert_old_quants(dict(state_dict), model_prefix="", metadata=dict(metadata))
        out_sd2, _ = comfy.utils.convert_old_quants(out_sd1, model_prefix="", metadata=dict(metadata))
        self.assertEqual(set(out_sd1.keys()), set(out_sd2.keys()))
        self.assertTrue(torch.equal(out_sd1["proj_in.comfy_quant"], out_sd2["proj_in.comfy_quant"]))

    # ---- extra: functional end-to-end, proving the fixed layer actually loads as QuantizedTensor ----
    def test_functional_load_after_prefix_mismatch_fix(self):
        layer_quant_config = {"model.diffusion_model.layer1": {"format": "float8_e4m3fn"}}
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        # sd already stripped of "model.diffusion_model." (as it is by the
        # time comfy/sd.py's second convert_old_quants call runs), while
        # metadata still carries the full prefix.
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
        }
        state_dict, _ = comfy.utils.convert_old_quants(
            state_dict,
            model_prefix="model.diffusion_model.",
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)


class TestKnownResidualGap(unittest.TestCase):
    """Documents a related but DISTINCT failure mode this PR does not close,
    found while writing the regression tests above. Kept as an
    expectedFailure so it stays visible instead of silently passing or
    breaking CI.

    load_diffusion_model_state_dict() passes model_prefix="" (a string
    literal, not the real diffusion-model prefix) to both of its
    convert_old_quants() calls -- see comfy/sd.py, unchanged by this PR since
    editing that call site is the path #13328 was closed for. When a
    checkpoint's real weight keys carry NO wrapper prefix at all but its
    _quantization_metadata layer keys DO carry a "model.diffusion_model."
    style prefix, convert_old_quants() (both before and after this fix) has
    no way to recognize the mismatch on the first call with model_prefix="",
    so its fallback write -- required to be byte-for-byte identical to
    today's behavior for the "neither convention matches" case -- pollutes
    state_dict with spurious "model.diffusion_model.*.comfy_quant" keys.
    comfy.model_detection.unet_prefix_from_state_dict() (a different
    function, out of this PR's scope) then falsely detects
    "model.diffusion_model." as the prefix from those spurious keys alone,
    and state_dict_prefix_replace(..., filter_keys=True) strips using that
    wrong prefix, discarding every real .weight/.weight_scale tensor.

    This is pre-existing: it reproduces identically with and without this
    PR's fix (verified manually), because the root cause here is
    unet_prefix_from_state_dict()/the sd.py call site, not the marker-key
    resolution this PR changes. Closing it would require either passing the
    real prefix into convert_old_quants() from sd.py (a call-site change) or
    hardening unet_prefix_from_state_dict() to ignore .comfy_quant keys --
    both outside "only touch convert_old_quants()".
    """

    @unittest.expectedFailure
    def test_empty_model_prefix_cannot_prevent_real_prefix_detection_poisoning(self):
        prefix = "model.diffusion_model."
        num_layers = 10  # unet_prefix_from_state_dict requires > 5 matches
        sd = {}
        layers_meta = {}
        for i in range(num_layers):
            local = f"block{i}"
            sd[f"{local}.weight"] = torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
            sd[f"{local}.weight_scale"] = torch.tensor(1.0)
            layers_meta[f"{prefix}{local}"] = {"format": "float8_e4m3fn"}
        metadata = {"_quantization_metadata": json.dumps({"layers": layers_meta})}

        # exact two-call dance from comfy/sd.py::load_diffusion_model_state_dict
        sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)
        diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
        temp_sd = comfy.utils.state_dict_prefix_replace(sd, {diffusion_model_prefix: ""}, filter_keys=True)
        if len(temp_sd) > 0:
            sd = temp_sd
            sd, metadata = comfy.utils.convert_old_quants(sd, "", metadata=metadata)

        remaining_weights = [k for k in sd if k.endswith(".weight")]
        self.assertEqual(len(remaining_weights), num_layers)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for io.DynamicGroup: expansion/reconstruction (0-row and N-row cases)."""
import sys
import types
import pytest

# Stub torch (type-hint only in _io.py; real torch not available in unit-test env)
if "torch" not in sys.modules:
    _torch_stub = types.ModuleType("torch")
    _torch_stub.Tensor = object  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch_stub

from comfy_api.latest._io import (  # noqa: E402
    DynamicGroup,
    Float,
    Int,
    String,
    Boolean,
    get_finalized_class_inputs,
    build_nested_inputs,
    create_input_dict_v1,
    setup_dynamic_input_funcs,
)

# Make sure dynamic input funcs are registered (may already be done at import time)
setup_dynamic_input_funcs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_class_inputs(group_input: DynamicGroup.Input) -> dict:
    """Wrap a DynamicGroup.Input into the required/optional dict structure."""
    return create_input_dict_v1([group_input])


def _run(group_input: DynamicGroup.Input, live_values: dict) -> dict:
    """End-to-end helper: expand schema + reconstruct values.

    Mirrors the production split in execution.py:
      1. get_finalized_class_inputs  (schema expansion, line 162)
      2. build_nested_inputs          (value reconstruction, line 281)

    The two steps are separate in production because the engine resolves
    linked node outputs between them, but in tests we supply values directly.
    """
    class_inputs = _make_class_inputs(group_input)
    _, _, v3_data = get_finalized_class_inputs(class_inputs, live_values)
    return build_nested_inputs(dict(live_values), v3_data)


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

class TestDynamicGroupInputConstruction:
    def test_basic_construction(self):
        inp = DynamicGroup.Input(
            "loras",
            template=[
                Float.Input("strength", default=1.0),
                String.Input("name"),
            ],
            min=0,
            max=10,
        )
        assert inp.id == "loras"
        assert inp.min == 0
        assert inp.max == 10
        assert len(inp.template) == 2

    def test_get_all_includes_self_and_template(self):
        inp = DynamicGroup.Input(
            "items",
            template=[Float.Input("value")],
        )
        all_inputs = inp.get_all()
        assert all_inputs[0] is inp
        assert all_inputs[1].id == "value"

    def test_as_dict_has_template_min_max(self):
        inp = DynamicGroup.Input(
            "items",
            template=[Float.Input("val", default=0.5)],
            min=1,
            max=5,
        )
        d = inp.as_dict()
        assert "template" in d
        assert d["min"] == 1
        assert d["max"] == 5

    def test_duplicate_field_ids_raises(self):
        with pytest.raises(AssertionError):
            DynamicGroup.Input(
                "bad",
                template=[Float.Input("x"), Float.Input("x")],
            )

    def test_empty_template_raises(self):
        with pytest.raises(AssertionError):
            DynamicGroup.Input("bad", template=[])

    def test_min_gt_max_raises(self):
        with pytest.raises(AssertionError):
            DynamicGroup.Input("bad", template=[Float.Input("x")], min=5, max=3)

    def test_max_exceeds_limit_raises(self):
        with pytest.raises(AssertionError):
            DynamicGroup.Input("bad", template=[Float.Input("x")], max=101)

    def test_dynamic_input_in_template_raises(self):
        with pytest.raises(AssertionError):
            DynamicGroup.Input(
                "bad",
                template=[DynamicGroup.Input("nested", template=[Float.Input("x")])],
            )

    def test_validate_calls_through(self):
        inp = DynamicGroup.Input("items", template=[Float.Input("val", min=-1.0, max=1.0)])
        inp.validate()  # should not raise


# ---------------------------------------------------------------------------
# 0-row case
# ---------------------------------------------------------------------------

class TestZeroRows:
    def test_empty_live_inputs_produces_empty_list(self):
        """With min=0 and no live values, the result should be an empty list."""
        inp = DynamicGroup.Input("loras", template=[Float.Input("strength", default=1.0)], min=0, max=10)
        assert _run(inp, {}).get("loras") == []

    def test_min_zero_with_values(self):
        """min=0 but 2 rows of live data."""
        inp = DynamicGroup.Input("loras", template=[Float.Input("strength", default=1.0)], min=0, max=10)
        result = _run(inp, {"loras.0.strength": 0.8, "loras.1.strength": 0.5})
        assert result["loras"] == [{"strength": 0.8}, {"strength": 0.5}]


# ---------------------------------------------------------------------------
# N-row case
# ---------------------------------------------------------------------------

class TestNRows:
    def test_two_rows_two_fields(self):
        """Two rows with two fields each produce a list[dict]."""
        inp = DynamicGroup.Input(
            "loras",
            template=[String.Input("lora_name"), Float.Input("strength", default=1.0)],
            min=0, max=50,
        )
        result = _run(inp, {
            "loras.0.lora_name": "model_a.safetensors", "loras.0.strength": 0.9,
            "loras.1.lora_name": "model_b.safetensors", "loras.1.strength": 0.4,
        })
        assert result["loras"] == [
            {"lora_name": "model_a.safetensors", "strength": 0.9},
            {"lora_name": "model_b.safetensors", "strength": 0.4},
        ]

    def test_rows_are_sorted_by_index(self):
        """Rows must be in ascending index order even if dict iteration is unordered."""
        inp = DynamicGroup.Input("items", template=[Int.Input("v", default=0)], min=0, max=10)
        result = _run(inp, {"items.0.v": 10, "items.2.v": 30, "items.1.v": 20})
        assert [row["v"] for row in result["items"]] == [10, 20, 30]

    def test_min_rows_schema_slots(self):
        """With min=2 and no live data, 2 slots must appear in the expanded schema."""
        inp = DynamicGroup.Input("items", template=[Float.Input("val", default=0.0)], min=2, max=5)
        out, _, _ = get_finalized_class_inputs(_make_class_inputs(inp), {})
        all_slots = {**out.get("required", {}), **out.get("optional", {})}
        assert "items.0.val" in all_slots
        assert "items.1.val" in all_slots

    def test_min_rows_reconstructs_when_no_values(self):
        """min=2 with NO live values must still yield a 2-element list,
        not collapse to [] (regression: parent-path clobber)."""
        inp = DynamicGroup.Input("items", template=[Float.Input("val", default=0.0)], min=2, max=5)
        result = _run(inp, {})
        assert len(result["items"]) == 2
        assert all("val" in row for row in result["items"])

    def test_min_rows_reconstructs_with_partial_values(self):
        """min=2 with only the first row's value present still yields 2 rows."""
        inp = DynamicGroup.Input("items", template=[Float.Input("val", default=0.0)], min=2, max=5)
        result = _run(inp, {"items.0.val": 0.7})
        assert len(result["items"]) == 2
        assert result["items"][0]["val"] == 0.7
        assert result["items"][1]["val"] is None

    def test_list_paths_in_v3_data(self):
        """list_paths must contain the group id so build_nested_inputs knows to convert."""
        inp = DynamicGroup.Input("things", template=[Boolean.Input("flag")], min=0, max=5)
        _, _, v3_data = get_finalized_class_inputs(_make_class_inputs(inp), {})
        assert "things" in v3_data.get("list_paths", set())

    def test_no_leftover_flat_keys(self):
        """Flat keys must be consumed; only the reconstructed list remains."""
        inp = DynamicGroup.Input("rows", template=[Float.Input("x", default=0.0)], min=0, max=5)
        result = _run(inp, {"rows.0.x": 1.0, "rows.1.x": 2.0})
        assert "rows.0.x" not in result
        assert "rows.1.x" not in result
        assert isinstance(result["rows"], list)

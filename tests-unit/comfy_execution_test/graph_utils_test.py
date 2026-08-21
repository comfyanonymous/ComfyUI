import pytest

from comfy_execution.graph_utils import GraphBuilder


def test_node_flattens_dynamic_combo_inputs():
    builder = GraphBuilder(prefix="dynamic_combo")

    builder.node(
        "ResizeImageMaskNode",
        resize_type={"resize_type": "scale by multiplier", "multiplier": 2.0},
    )

    assert builder.finalize()["dynamic_combo1"]["inputs"] == {
        "resize_type": "scale by multiplier",
        "resize_type.multiplier": 2.0,
    }


def test_node_flattens_nested_dynamic_combo_inputs():
    builder = GraphBuilder(prefix="nested_dynamic_combo")

    builder.node(
        "NestedDynamicComboNode",
        combo={
            "combo": "option4",
            "subcombo": {
                "subcombo": "opt1",
                "float_x": 1.5,
                "float_y": 2.5,
            },
        },
    )

    assert builder.finalize()["nested_dynamic_combo1"]["inputs"] == {
        "combo": "option4",
        "combo.subcombo": "opt1",
        "combo.subcombo.float_x": 1.5,
        "combo.subcombo.float_y": 2.5,
    }


def test_node_preserves_regular_dict_inputs():
    builder = GraphBuilder(prefix="regular_dict")
    value = {"key": "value"}

    builder.node("TestNode", value=value)

    assert builder.finalize()["regular_dict1"]["inputs"]["value"] == value


def test_node_rejects_conflicting_dynamic_combo_keys():
    builder = GraphBuilder(prefix="conflict")

    with pytest.raises(ValueError, match="conflicting input 'combo.string'"):
        builder.node(
            "NestedDynamicComboNode",
            combo={"combo": "option1", "string": "hello"},
            **{"combo.string": "conflict"},
        )

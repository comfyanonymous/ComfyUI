"""Prospective public-API regressions for bounded per-invocation outputs; UNRUN."""
from dataclasses import FrozenInstanceError

import pytest

from comfy_api.latest import io


def schema(**kwargs):
    return io.Schema(
        node_id="RepeatedOutputPublic_TEST",
        inputs=[],
        accept_all_inputs=True,
        outputs=[
            io.String.Output("fixed", display_name="Fixed"),
            io.RepeatedOutput("values", io.AnyType.Output(), "keys",
                              max_outputs=4, max_selector_bytes=16),
        ],
        **kwargs,
    )


@pytest.mark.parametrize("inputs,names", [
    ({}, []),
    ({"keys": None}, []),
    ({"keys": ""}, [""]),
    ({"keys": "A,A,missing"}, ["A", "A", "missing"]),
    ({"keys": "A, B,"}, ["A", " B", ""]),
])
def test_literal_names_and_frozen_instance_layout(inputs, names):
    declaration = schema()
    declaration.validate()
    layout = declaration.resolve_outputs(inputs)
    assert layout.slots[0] == ("fixed", "STRING", "Fixed", False)
    assert [slot[2] for slot in layout.slots[1:]] == names
    assert [slot[0] for slot in layout.slots[1:]] == [f"values:{i}" for i in range(len(names))]
    assert layout.as_dict() == {"version": 1, "slots": [list(slot) for slot in layout.slots]}
    with pytest.raises(FrozenInstanceError):
        layout.slots = ()
    # Resolving another prompt must not alter the first invocation or declaration.
    declaration.resolve_outputs({"keys": "different"})
    assert [slot[2] for slot in layout.slots[1:]] == names
    assert len(declaration.outputs) == 2


@pytest.mark.parametrize("value", [["source", 0], 1, True, "A,B,C,D,E", "x" * 17, "€" * 6])
def test_bad_selector_or_bounds_are_denied_without_truncation(value):
    with pytest.raises((TypeError, ValueError)):
        schema().resolve_outputs({"keys": value})


@pytest.mark.parametrize("bounds", [
    {"max_outputs": False, "max_selector_bytes": 16},
    {"max_outputs": 0, "max_selector_bytes": 16},
    {"max_outputs": 1025, "max_selector_bytes": 16},
    {"max_outputs": 4, "max_selector_bytes": True},
    {"max_outputs": 4, "max_selector_bytes": 65537},
])
def test_exact_integer_declaration_limits(bounds):
    with pytest.raises(ValueError):
        io.RepeatedOutput("values", io.AnyType.Output(), "keys", **bounds)


def test_list_flag_and_multibyte_boundary():
    declaration = io.Schema(node_id="RepeatedList_TEST", accept_all_inputs=True, outputs=[
        io.RepeatedOutput("values", io.AnyType.Output(is_output_list=True), "keys",
                          max_outputs=2, max_selector_bytes=4),
    ])
    declaration.validate()
    assert declaration.resolve_outputs({"keys": "éé"}).slots == (("values:0", "*", "éé", True),)
    assert declaration.resolve_outputs({"keys": ","}).slots == (
        ("values:0", "*", "", True), ("values:1", "*", "", True))


@pytest.mark.parametrize("kwargs", [{"is_input_list": True}, {"enable_expand": True}])
def test_unqualified_execution_shapes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        schema(**kwargs).validate()


def test_fixed_socket_id_collision_is_rejected():
    declaration = io.Schema(node_id="RepeatedCollision_TEST", accept_all_inputs=True, outputs=[
        io.AnyType.Output("values:0"),
        io.RepeatedOutput("values", io.AnyType.Output(), "keys",
                          max_outputs=2, max_selector_bytes=4),
    ])
    declaration.validate()
    with pytest.raises(ValueError):
        declaration.resolve_outputs({"keys": "A"})

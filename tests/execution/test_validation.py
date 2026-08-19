from __future__ import annotations

import asyncio

import pytest

from comfy import cli_args

# model_management probes the accelerator at import time; these tests never
# execute nodes, so force the CPU code path before importing execution.
cli_args.args.cpu = True

import execution
import nodes


class ValidationTestSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "testing/validation"

    def execute(self):
        return ("ok",)


class ValidationTestOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("STRING",)}}

    RETURN_TYPES = ()
    FUNCTION = "execute"
    CATEGORY = "testing/validation"
    OUTPUT_NODE = True

    def execute(self, value):
        return ()


@pytest.fixture
def validation_nodes():
    classes = {
        "ValidationTestSource": ValidationTestSource,
        "ValidationTestOutput": ValidationTestOutput,
    }
    for name, cls in classes.items():
        nodes.NODE_CLASS_MAPPINGS[name] = cls
    yield classes
    for name in classes:
        nodes.NODE_CLASS_MAPPINGS.pop(name, None)


def run_validate_prompt(prompt: dict) -> tuple[bool, dict | None, list, dict]:
    return asyncio.run(execution.validate_prompt("test-prompt", prompt, None))


def build_prompt(slot: int) -> dict:
    return {
        "1": {"class_type": "ValidationTestSource", "inputs": {}},
        "2": {"class_type": "ValidationTestOutput", "inputs": {"value": ["1", slot]}},
    }


def test_in_range_slot_validates(validation_nodes) -> None:
    valid, error, _, node_errors = run_validate_prompt(build_prompt(0))
    assert valid is True
    assert error is None
    assert node_errors == {}


def test_out_of_range_slot_reports_typed_error(validation_nodes) -> None:
    valid, error, _, node_errors = run_validate_prompt(build_prompt(1))

    assert valid is False
    assert error is not None
    assert error["type"] == "prompt_outputs_failed_validation"
    errors = node_errors["2"]["errors"]
    assert errors[0]["type"] == "return_slot_out_of_range"
    assert errors[0]["extra_info"]["linked_node"] == ["1", 1]
    assert errors[0]["extra_info"]["output_count"] == 1


def test_negative_slot_reports_typed_error(validation_nodes) -> None:
    valid, _, _, node_errors = run_validate_prompt(build_prompt(-1))

    assert valid is False
    errors = node_errors["2"]["errors"]
    assert errors[0]["type"] == "return_slot_out_of_range"

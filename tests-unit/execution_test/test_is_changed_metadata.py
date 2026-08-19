import asyncio
import json
import math

import pytest

from comfy.cli_args import args

args.cpu = True

import execution
from comfy_execution.graph import DynamicPrompt


class TestNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
    ],
)
def test_is_changed_metadata_is_valid_json(monkeypatch, value, expected):
    prompt = {"node": {"class_type": "TestNode", "inputs": {}}}
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "TestNode", TestNode)

    async def map_node(*args, **kwargs):
        return [value]

    async def resolve_result(result):
        return result

    monkeypatch.setattr(execution, "_async_map_node_over_list", map_node)
    monkeypatch.setattr(execution, "resolve_map_node_over_list_results", resolve_result)

    cache = execution.IsChangedCache("prompt", DynamicPrompt(prompt), None)
    cache_value = asyncio.run(cache.get("node"))

    if expected == "NaN":
        assert math.isnan(cache_value[0])
    else:
        assert cache_value[0] == value
    metadata = json.dumps(prompt, allow_nan=False)
    assert json.loads(metadata)["node"]["is_changed"] == [expected]


def test_is_changed_evaluation_error_metadata_is_valid_json(monkeypatch):
    prompt = {"node": {"class_type": "TestNode", "inputs": {}}}
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "TestNode", TestNode)

    async def map_node(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(execution, "_async_map_node_over_list", map_node)

    cache = execution.IsChangedCache("prompt", DynamicPrompt(prompt), None)
    cache_value = asyncio.run(cache.get("node"))

    assert math.isnan(cache_value)
    metadata = json.dumps(prompt, allow_nan=False)
    assert json.loads(metadata)["node"]["is_changed"] == "NaN"

import asyncio

from comfy.cli_args import args

previous_cpu_setting = args.cpu
args.cpu = True

import execution
import nodes

args.cpu = previous_cpu_setting


class NodeWithDefault:
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 7, "min": 0, "max": 10}),
            }
        }


class NodeWithoutDefault:
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"min": 0, "max": 10}),
            }
        }


class NodeWithListDefault:
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": (["alpha", "beta", "gamma"], {"default": ["alpha", "gamma"], "multiselect": True}),
            }
        }


def test_required_input_with_default_is_filled(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "NodeWithDefault", NodeWithDefault)
    prompt = {"node": {"class_type": "NodeWithDefault", "inputs": {}}}

    valid, error, outputs, node_errors = asyncio.run(
        execution.validate_prompt("test", prompt, None)
    )

    assert valid
    assert error is None
    assert outputs == ["node"]
    assert node_errors == {}
    assert prompt["node"]["inputs"]["value"] == 7


def test_required_input_without_default_is_not_filled(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "NodeWithoutDefault", NodeWithoutDefault)
    prompt = {"node": {"class_type": "NodeWithoutDefault", "inputs": {}}}

    valid, _, _, node_errors = asyncio.run(
        execution.validate_prompt("test", prompt, None)
    )

    assert not valid
    assert node_errors["node"]["errors"][0]["type"] == "required_input_missing"
    assert "value" not in prompt["node"]["inputs"]


def test_list_default_is_not_treated_as_a_link(monkeypatch):
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "NodeWithListDefault", NodeWithListDefault)
    prompt = {"node": {"class_type": "NodeWithListDefault", "inputs": {}}}

    valid, error, outputs, node_errors = asyncio.run(
        execution.validate_prompt("test", prompt, None)
    )

    assert valid
    assert error is None
    assert outputs == ["node"]
    assert node_errors == {}
    assert prompt["node"]["inputs"]["tags"] == ["alpha", "gamma"]

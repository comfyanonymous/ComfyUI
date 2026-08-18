import asyncio

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import execution  # noqa: E402


def test_validate_inputs_deep_acyclic_chain(monkeypatch):
    class Source:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    class Link:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {})}}

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "DeepValidationSource", Source)
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "DeepValidationLink", Link)

    size = 2_000
    prompt = {"0": {"class_type": "DeepValidationSource", "inputs": {}}}
    for index in range(1, size):
        prompt[str(index)] = {
            "class_type": "DeepValidationLink",
            "inputs": {"value": [str(index - 1), 0]},
        }

    validated = {}
    result = execution.validate_inputs("test", prompt, str(size - 1), validated)
    result = asyncio.run(result)

    assert result == (True, [], str(size - 1))
    assert len(validated) == size


def test_validate_inputs_reports_dependency_cycles(monkeypatch):
    class Link:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {})}}

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "CycleValidationLink", Link)
    prompt = {
        "0": {"class_type": "CycleValidationLink", "inputs": {"value": ["1", 0]}},
        "1": {"class_type": "CycleValidationLink", "inputs": {"value": ["0", 0]}},
    }

    validated = {}
    result = asyncio.run(execution.validate_inputs("test", prompt, "0", validated))

    assert result[0] is False
    assert result[2] == "0"
    assert result[1][0]["type"] == "dependency_cycle"
    assert result[1][0]["details"] == "0 (CycleValidationLink) -> 1 (CycleValidationLink) -> 0 (CycleValidationLink)"
    assert [validated[node_id][0] for node_id in prompt] == [False, False]


def test_validate_inputs_awaits_custom_validator(monkeypatch):
    class Validated:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {"default": 7})}}

        @classmethod
        async def VALIDATE_INPUTS(cls, value):
            await asyncio.sleep(0)
            return value == 7

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "AsyncValidationValue", Validated)
    prompt = {"0": {"class_type": "AsyncValidationValue", "inputs": {"value": 7}}}

    result = asyncio.run(execution.validate_inputs("test", prompt, "0", {}))

    assert result == (True, [], "0")


def test_validate_inputs_reports_dependency_validation_exception(monkeypatch):
    class Source:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    class Broken:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {})}}

        @classmethod
        async def VALIDATE_INPUTS(cls, value):
            await asyncio.sleep(0)
            raise RuntimeError("validation failed")

    class Consumer:
        RETURN_TYPES = ("INT",)

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("INT", {})}}

    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "ExceptionValidationSource", Source)
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "ExceptionValidationLink", Broken)
    monkeypatch.setitem(execution.nodes.NODE_CLASS_MAPPINGS, "ExceptionValidationConsumer", Consumer)
    prompt = {
        "0": {"class_type": "ExceptionValidationSource", "inputs": {}},
        "1": {"class_type": "ExceptionValidationLink", "inputs": {"value": ["0", 0]}},
        "2": {"class_type": "ExceptionValidationConsumer", "inputs": {"value": ["1", 0]}},
    }

    validated = {}
    result = asyncio.run(execution.validate_inputs("test", prompt, "2", validated))

    assert result[0] is False
    assert result[2] == "2"
    error = validated["1"][1][0]
    assert error["type"] == "exception_during_inner_validation"
    assert error["extra_info"]["exception_message"] == "validation failed"
    assert validated["0"][0] is True

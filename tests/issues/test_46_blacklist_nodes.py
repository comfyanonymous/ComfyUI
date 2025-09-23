import pytest
from importlib.resources import files

from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy

_TEST_WORKFLOW_1 = {
    "0": {
        "inputs": {},
        "class_type": "ShouldNotExist",
        "_meta": {
            "title": ""
        }
    },
    "1": {
        "inputs": {},
        "class_type": "TestPath",
        "_meta": {
            "title": ""
        }
    }
}

_TEST_WORKFLOW_2 = {
    "1": {
        "inputs": {},
        "class_type": "TestPath",
        "_meta": {
            "title": ""
        }
    }
}


@pytest.mark.asyncio
async def test_blacklist_node():
    config = Configuration(blacklist_custom_nodes=['issue_46'])
    # for finding the custom nodes
    config.base_paths = [str(files(__package__))]

    async with Comfy(config) as client:
        from comfy.cmd.execution import validate_prompt
        res = await validate_prompt("1", prompt=_TEST_WORKFLOW_1, partial_execution_list=[])
        assert "ShouldNotExist" in res.error["message"]
        assert res.error["type"] == "invalid_prompt"
        res = await validate_prompt("2", prompt=_TEST_WORKFLOW_2, partial_execution_list=[])
        assert "TestPath" not in res.error, "successfully loaded issue_25 nodes"

from importlib.resources import files

import pytest

from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.execution_context import context_configuration

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
    config = default_configuration()
    config.blacklist_custom_nodes = ['issue_46']
    # for finding the custom nodes
    config.base_paths = [str(files(__package__))]

    with context_configuration(config):
        from comfy.nodes_context import get_nodes
        nodes = get_nodes()
        assert "ShouldNotExist" not in nodes.NODE_CLASS_MAPPINGS
    async with Comfy(config) as client:
        from comfy.cmd.execution import validate_prompt
        res = await validate_prompt("1", prompt=_TEST_WORKFLOW_1, partial_execution_list=[])
        assert "ShouldNotExist" in res.error["message"]
        assert res.error["type"] == "invalid_prompt"
        res = await validate_prompt("2", prompt=_TEST_WORKFLOW_2, partial_execution_list=[])
        assert "TestPath" not in res.error, "successfully loaded issue_25 nodes"

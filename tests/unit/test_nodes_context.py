from unittest.mock import patch

import pytest
import sys

from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.make_mutable import make_mutable
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.execution_context import context_add_custom_nodes
from comfy.nodes.package_typing import CustomNode, ExportedNodes
from tests.unit.test_panics import ThrowsExceptionNode


def disable_vanilla(*args):
    patch_disable_vanilla = globals()['prepare_vanilla_environment'] = patch('comfy_compatibility.vanilla.prepare_vanilla_environment', lambda: "patched")
    patch_disable_vanilla.start()
    from comfy_compatibility.vanilla import prepare_vanilla_environment
    assert prepare_vanilla_environment() == "patched"


def enable_vanilla(*args):
    patch_disable_vanilla = globals()['prepare_vanilla_environment']
    patch_disable_vanilla.stop()


class AssertVanillaImportFails(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "assert_import_fails"
    CATEGORY = "Testing/Nodes"

    def assert_import_fails(self) -> tuple[str]:
        try:
            # sometimes, other code like fluxtapoz has a directly called nodes, and for
            # development purposes, its source directory is added to path, and this
            # can be imported, so import nodes cannot be used
            if 'nodes' in sys.modules:
                assert 'NODE_CLASS_MAPPINGS' not in sys.modules['nodes'] or 'SplitImageWithAlpha' not in sys.modules['nodes'].NODE_CLASS_MAPPINGS
                del sys.modules['nodes']

        except ModuleNotFoundError:
            pass
        return ("dummy",)


class PrepareVanillaEnvironment(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input": ("STRING", {}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "prepare"
    CATEGORY = "Testing/Nodes"

    def prepare(self, input: str) -> tuple[str]:
        enable_vanilla()
        from comfy_compatibility import vanilla
        vanilla.prepare_vanilla_environment()
        assert "nodes" in sys.modules
        return ("dummy",)


class AssertVanillaImportSucceeds(CustomNode):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "input": ("STRING", {}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "assert_import_succeeds"
    CATEGORY = "Testing/Nodes"

    def assert_import_succeeds(self, input: str) -> tuple[str]:
        import nodes
        assert "SplitImageWithAlpha" in nodes.NODE_CLASS_MAPPINGS
        return ("",)


def create_nodes_context_workflow():
    """Create a workflow that uses our test node to raise an exception"""
    return make_mutable({
        "1": {"class_type": "AssertVanillaImportFails", "inputs": {}},
        "2": {"class_type": "PrepareVanillaEnvironment", "inputs": {"input": ["1", 0]}},
        "3": {"class_type": "AssertVanillaImportSucceeds", "inputs": {"input": ["2", 0]}},
        "4": {"class_type": "PreviewString", "inputs": {"value": ["3", 0]}},
    })


TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestExceptionNode": "Test Exception Node",
    "AssertVanillaImportFails": "Assert Vanilla Import Fails",
    "PrepareVanillaEnvironment": "Prepare Vanilla Environment",
    "AssertVanillaImportSucceeds": "Assert Vanilla Import Succeeds",
}

EXECUTOR_FACTORIES = [
    (ProcessPoolExecutor, {"max_workers": 1}),
]


@pytest.mark.asyncio
async def test_nodes_context_shim():
    """Test panic behavior with different executor types"""

    # Initialize the specific executor
    executor = ProcessPoolExecutor(max_workers=1, initializer=disable_vanilla)

    if 'nodes' in sys.modules:
        # something else imported it
        del sys.modules['nodes']
    assert 'nodes' not in sys.modules
    with context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS={
        "TestExceptionNode": ThrowsExceptionNode,
        "AssertVanillaImportFails": AssertVanillaImportFails,
        "PrepareVanillaEnvironment": PrepareVanillaEnvironment,
        "AssertVanillaImportSucceeds": AssertVanillaImportSucceeds,
    }, NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS)):
        async with Comfy(executor=executor) as client:
            # Queue our failing workflow
            workflow = create_nodes_context_workflow()
            await client.queue_prompt(workflow)
    assert 'nodes' not in sys.modules

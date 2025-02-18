import asyncio
import threading
from unittest.mock import patch

import pytest
import torch

from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.cmd.execution import nodes
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.nodes.package_typing import CustomNode, ExportedNodes


@pytest.mark.asyncio
async def test_event_loop_callbacks():
    """Test to understand event loop callback behavior in pytest-asyncio"""
    callback_executed = False
    current_thread = threading.current_thread()
    current_loop = asyncio.get_running_loop()

    def callback(*args):
        nonlocal callback_executed
        print(f"Callback executing in thread: {threading.current_thread()}")
        print(f"Original thread was: {current_thread}")
        callback_executed = True

    print(f"Test running in thread: {current_thread}")
    print(f"Test using event loop: {current_loop}")

    # Try different ways of scheduling the callback
    current_loop.call_soon(callback)
    await asyncio.sleep(0)
    print(f"After sleep(0), callback_executed: {callback_executed}")

    if not callback_executed:
        current_loop.call_soon_threadsafe(callback)
        await asyncio.sleep(0)
        print(f"After threadsafe callback, callback_executed: {callback_executed}")

    if not callback_executed:
        # Try running callback in event loop directly
        await asyncio.get_event_loop().run_in_executor(None, callback)
        print(f"After run_in_executor, callback_executed: {callback_executed}")

    assert callback_executed, "Callback was never executed"


@pytest.mark.asyncio
async def test_separate_thread_callback():
    """Test callbacks scheduled from a separate thread"""
    callback_executed = False
    event = threading.Event()
    main_loop = asyncio.get_running_loop()

    def thread_func():
        print(f"Thread function running in: {threading.current_thread()}")
        main_loop.call_soon_threadsafe(lambda *_: event.set())

    print(f"Test running in thread: {threading.current_thread()}")
    print(f"Test using event loop: {main_loop}")

    # Start thread that will schedule callback
    thread = threading.Thread(target=thread_func)
    thread.start()

    # Wait for event with timeout
    try:
        await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, event.wait),
            timeout=1.0
        )
        print("Event was set!")
    except asyncio.TimeoutError:
        print("Timed out waiting for event!")
        assert False, "Event was never set"

    thread.join()


# Custom test exception that we'll configure to panic on
class TestUnrecoverableError(Exception):
    pass


class TestExceptionNode(CustomNode):
    """Node that raises a specific exception for testing"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "should_raise": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)  # Make it an output node by returning IMAGE
    FUNCTION = "raise_exception"
    CATEGORY = "Testing/Nodes"
    OUTPUT_NODE = True

    def raise_exception(self, should_raise=True) -> tuple[RGBImageBatch]:
        if should_raise:
            raise TestUnrecoverableError("Test exception from node")
        else:
            # Return a dummy image if not raising
            return (torch.zeros([1, 64, 64, 3]),)


# Export the node mappings
TEST_NODE_CLASS_MAPPINGS = {
    "TestExceptionNode": TestExceptionNode,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestExceptionNode": "Test Exception Node",
}


def create_failing_workflow():
    """Create a workflow that uses our test node to raise an exception"""
    return make_mutable({
        "1": {
            "class_type": "TestExceptionNode",
            "inputs": {
                "should_raise": True
            }
        }
    })


@pytest.mark.asyncio
async def test_panic_on_exception():
    # Set up the test nodes
    nodes.update(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS))

    # Create configuration with our test exception in panic_when
    config = Configuration()
    config.panic_when = [f"{__name__}.TestUnrecoverableError"]

    # Mock sys.exit to prevent actual exit and verify it's called
    with patch('sys.exit') as mock_exit:
        try:
            async with EmbeddedComfyClient(configuration=config) as client:
                # Queue our failing workflow
                await client.queue_prompt(create_failing_workflow())
        except TestUnrecoverableError:
            # We expect the exception to be raised here
            pass

        # Give the event loop a chance to process the exit callback
        await asyncio.sleep(0)

        # Verify sys.exit was called with code 1
        mock_exit.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_no_panic_when_disabled():
    """Verify that the same exception doesn't trigger exit when not in panic_when"""
    # Set up the test nodes
    nodes.update(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS))

    # Create configuration without the exception in panic_when
    config = Configuration()

    # Mock sys.exit to verify it's not called
    with patch('sys.exit') as mock_exit:
        try:
            async with EmbeddedComfyClient(configuration=config) as client:
                # Queue our failing workflow
                await client.queue_prompt(create_failing_workflow())
        except TestUnrecoverableError:
            # We expect the exception to be raised here
            pass

        # Give the event loop a chance to process any callbacks
        await asyncio.sleep(0.1)

        # Verify sys.exit was not called
        mock_exit.assert_not_called()

import asyncio
import threading
from unittest.mock import patch

import pytest
import torch
import pebble.common.types

from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.tensor_types import RGBImageBatch
from comfy.distributed.executors import ContextVarExecutor
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.execution_context import context_add_custom_nodes
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
class UnrecoverableError(Exception):
    pass


class ThrowsExceptionNode(CustomNode):
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
            raise UnrecoverableError("Test exception from node")
        else:
            # Return a dummy image if not raising
            return (torch.zeros([1, 64, 64, 3]),)


# Export the node mappings
TEST_NODE_CLASS_MAPPINGS = {
    "TestExceptionNode": ThrowsExceptionNode,
}

TEST_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestExceptionNode": "Test Exception Node",
}

EXECUTOR_FACTORIES = [
    (ContextVarExecutor, {"max_workers": 1}),
    (ProcessPoolExecutor, {"max_workers": 1}),
]


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
@pytest.mark.parametrize("executor_cls,executor_kwargs", EXECUTOR_FACTORIES)
async def test_panic_on_exception_with_executor(executor_cls, executor_kwargs):
    """Test panic behavior with different executor types"""
    # Create configuration with our test exception in panic_when
    config = Configuration()
    config.panic_when = [f"{__name__}.UnrecoverableError"]

    # Initialize the specific executor
    executor = executor_cls(**executor_kwargs)

    # Mock sys.exit to prevent actual exit and verify it's called
    with (context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS,
                                                 NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS)),
          patch('sys.exit') as mock_exit):
        try:
            async with Comfy(configuration=config, executor=executor) as client:
                # Queue our failing workflow
                await client.queue_prompt(create_failing_workflow())
        except (SystemExit, pebble.common.types.ProcessExpired):
            sys_exit_called = True
        except UnrecoverableError:
            # We expect the exception to be raised here
            sys_exit_called = False

        # Give the event loop a chance to process the exit callback
        await asyncio.sleep(0)

        # Verify sys.exit was called with code 1
        if executor_cls == ProcessPoolExecutor:
            assert sys_exit_called
        else:
            mock_exit.assert_called_once_with(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_cls,executor_kwargs", EXECUTOR_FACTORIES)
async def test_no_panic_when_disabled_with_executor(executor_cls, executor_kwargs):
    """Test no-panic behavior with different executor types"""

    # Create configuration without the exception in panic_when
    config = Configuration()

    # Initialize the specific executor
    executor = executor_cls(**executor_kwargs)

    # Mock sys.exit to verify it's not called
    with (context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS,
                                                 NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS)),
          patch('sys.exit') as mock_exit):
        try:
            async with Comfy(configuration=config, executor=executor) as client:
                from comfy.cli_args import args
                assert len(args.panic_when) == 0
                # Queue our failing workflow
                await client.queue_prompt(create_failing_workflow())
        except SystemExit:
            sys_exit_called = True
        except UnrecoverableError:
            # We expect the exception to be raised here
            sys_exit_called = False

        # Give the event loop a chance to process any callbacks
        await asyncio.sleep(0)

        # Verify sys.exit was not called
        mock_exit.assert_not_called()
        assert not sys_exit_called


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_cls,executor_kwargs", EXECUTOR_FACTORIES)
async def test_executor_cleanup(executor_cls, executor_kwargs):
    """Test that executors are properly cleaned up after use"""
    executor = executor_cls(**executor_kwargs)

    with context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS,
                                                NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS)):
        async with Comfy(executor=executor) as client:
            # Create a simple workflow that doesn't raise
            workflow = create_failing_workflow()
            workflow["1"]["inputs"]["should_raise"] = False

            # Run it
            result = await client.queue_prompt(workflow)
            assert isinstance(result, dict), "Expected workflow to return results"


# Add a test for parallel execution to verify multi-worker behavior
@pytest.mark.asyncio
@pytest.mark.parametrize("executor_cls,executor_kwargs", [
    (ContextVarExecutor, {"max_workers": 2}),
    (ProcessPoolExecutor, {"max_workers": 2}),
])
async def test_parallel_execution(executor_cls, executor_kwargs):
    """Test that executors can handle multiple workflows in parallel"""
    executor = executor_cls(**executor_kwargs)

    with context_add_custom_nodes(ExportedNodes(NODE_CLASS_MAPPINGS=TEST_NODE_CLASS_MAPPINGS,
                                                NODE_DISPLAY_NAME_MAPPINGS=TEST_NODE_DISPLAY_NAME_MAPPINGS)):
        async with Comfy(executor=executor) as client:
            # Create multiple non-failing workflows
            workflow = create_failing_workflow()
            workflow["1"]["inputs"]["should_raise"] = False

            # Run multiple workflows concurrently
            results = await asyncio.gather(*[
                client.queue_prompt(workflow)
                for _ in range(3)
            ])

            assert len(results) == 3, "Expected all workflows to complete"
            assert all(isinstance(r, dict) for r in results), "Expected all workflows to return results"

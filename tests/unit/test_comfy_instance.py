import pytest
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.distributed.executors import ContextVarExecutor


@pytest.mark.asyncio
async def test_comfy_defaults():
    """Test that default initialization uses ContextVarExecutor."""
    client = Comfy()
    assert isinstance(client._executor, ContextVarExecutor)
    assert client._owns_executor


@pytest.mark.asyncio
async def test_comfy_config_triggers_process_pool():
    """Test that configurations affecting global state trigger ProcessPoolExecutor."""
    # "lowvram" is in MODEL_MANAGEMENT_ARGS
    client = Comfy(configuration={"lowvram": True})
    assert isinstance(client._executor, ProcessPoolExecutor)
    assert client._owns_executor


@pytest.mark.asyncio
async def test_comfy_config_unrelated_does_not_trigger():
    """Test that unrelated configuration keys do NOT trigger ProcessPoolExecutor."""
    # "some_random_ui_setting" is likely not in MODEL_MANAGEMENT_ARGS
    client = Comfy(configuration={"some_random_ui_setting": "value"})
    assert isinstance(client._executor, ContextVarExecutor)


@pytest.mark.asyncio
async def test_comfy_explicit_executor_string():
    """Test explicitly requesting an executor by string."""
    client = Comfy(executor="ProcessPoolExecutor")
    assert isinstance(client._executor, ProcessPoolExecutor)
    assert client._owns_executor

    client2 = Comfy(executor="ContextVarExecutor")
    assert isinstance(client2._executor, ContextVarExecutor)
    assert client2._owns_executor


@pytest.mark.asyncio
async def test_comfy_explicit_executor_instance():
    """Test passing an executor instance."""
    executor = ContextVarExecutor(max_workers=1)
    client = Comfy(executor=executor)
    assert client._executor is executor
    assert not client._owns_executor


@pytest.mark.asyncio
async def test_comfy_mismatch_string_raises():
    """Test that valid config requiring ProcessPoolExecutor raises error if ContextVarExecutor is forced via string."""
    with pytest.raises(ValueError, match="Configuration requires ProcessPoolExecutor"):
        Comfy(configuration={"lowvram": True}, executor="ContextVarExecutor")


@pytest.mark.asyncio
async def test_comfy_mismatch_instance_raises():
    """Test that valid config requiring ProcessPoolExecutor raises error if ContextVarExecutor instance is passed."""
    executor = ContextVarExecutor(max_workers=1)
    with pytest.raises(ValueError, match="Configuration requires ProcessPoolExecutor"):
        Comfy(configuration={"lowvram": True}, executor=executor)


@pytest.mark.asyncio
async def test_comfy_context_manager():
    """Test the async context manager behavior."""
    async with Comfy() as client:
        assert client.is_running
        assert isinstance(client._executor, ContextVarExecutor)
    assert not client.is_running

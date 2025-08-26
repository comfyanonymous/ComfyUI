import asyncio

import pytest
import torch

from comfy.cli_args_types import Configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner


@pytest.mark.asyncio
async def test_cuda_memory_usage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    device = torch.device("cuda")
    starting_memory = torch.cuda.memory_allocated(device)

    async with Comfy() as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs = await client.queue_prompt(prompt)
        assert outputs["13"]["images"][0]["abs_path"] is not None
        memory_after_workflow = torch.cuda.memory_allocated(device)
        assert memory_after_workflow > starting_memory, "Expected CUDA memory to increase after running the workflow"

    ending_memory = torch.cuda.memory_allocated(device)
    assert abs(ending_memory - starting_memory) < 1e7, "Expected CUDA memory to return close to starting memory after cleanup"


@pytest.mark.asyncio
async def test_embedded_comfy():
    async with Comfy() as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs = await client.queue_prompt(prompt)
        assert outputs["13"]["images"][0]["abs_path"] is not None

@pytest.mark.asyncio
async def test_configuration_options():
    config = Configuration()
    async with Comfy(configuration=config) as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs = await client.queue_prompt(prompt)
        assert outputs["13"]["images"][0]["abs_path"] is not None

@pytest.mark.asyncio
async def test_multithreaded_comfy():
    async with Comfy(max_workers=2) as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs_iter = await asyncio.gather(*[client.queue_prompt(prompt) for _ in range(4)])
        assert all(outputs["13"]["images"][0]["abs_path"] is not None for outputs in outputs_iter)


@pytest.mark.asyncio
async def test_progress_notifications():
    async with Comfy() as client:
        prompt = sdxl_workflow_with_refiner("test")
        task = client.queue_with_progress(prompt)

        notifications_received = []
        async for notification in task.progress():
            notifications_received.append(notification)

        assert len(notifications_received) > 0, "Should have received progress notifications"

        result = await task.get()
        assert result.outputs["13"]["images"][0]["abs_path"] is not None
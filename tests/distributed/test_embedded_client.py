import pytest
import torch

from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner


@pytest.mark.asyncio
async def test_cuda_memory_usage():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment")

    device = torch.device("cuda")
    starting_memory = torch.cuda.memory_allocated(device)

    async with EmbeddedComfyClient() as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs = await client.queue_prompt(prompt)
        assert outputs["13"]["images"][0]["abs_path"] is not None
        memory_after_workflow = torch.cuda.memory_allocated(device)
        assert memory_after_workflow > starting_memory, "Expected CUDA memory to increase after running the workflow"

    ending_memory = torch.cuda.memory_allocated(device)
    assert abs(
        ending_memory - starting_memory) < 1e7, "Expected CUDA memory to return close to starting memory after cleanup"


@pytest.mark.asyncio
async def test_embedded_comfy():
    async with EmbeddedComfyClient() as client:
        prompt = sdxl_workflow_with_refiner("test")
        outputs = await client.queue_prompt(prompt)
        assert outputs["13"]["images"][0]["abs_path"] is not None

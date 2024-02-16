import random

import pytest
from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner


@pytest.mark.asyncio
async def test_completes_prompt(comfy_background_server):
    client = AsyncRemoteComfyClient()
    prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
    png_image_bytes = await client.queue_prompt(prompt)
    assert len(png_image_bytes) > 1000


@pytest.mark.asyncio
async def test_completes_prompt_with_ui(comfy_background_server):
    client = AsyncRemoteComfyClient()
    prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
    result_dict = await client.queue_prompt_ui(prompt)
    # should contain one output
    assert len(result_dict) == 1


@pytest.mark.asyncio
async def test_completes_prompt_with_image_urls(comfy_background_server):
    client = AsyncRemoteComfyClient()
    random_seed = random.randint(1,4294967295)
    prompt = sdxl_workflow_with_refiner("test", inference_steps=1, seed=random_seed, refiner_steps=1)
    result_list = await client.queue_prompt_uris(prompt)
    assert len(result_list) == 3
    result_list = await client.queue_prompt_uris(prompt)
    # cached
    assert len(result_list) == 1

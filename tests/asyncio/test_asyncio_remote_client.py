import random
from urllib.parse import parse_qsl

import aiohttp
import pytest
from can_ada import URL, parse

from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner


@pytest.mark.asyncio
async def test_completes_prompt(comfy_background_server):
    async with AsyncRemoteComfyClient() as client:
        random_seed = random.randint(1, 4294967295)
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, seed=random_seed, refiner_steps=1)
        png_image_bytes = await client.queue_prompt(prompt)
    assert len(png_image_bytes) > 1000


@pytest.mark.asyncio
async def test_completes_prompt_with_ui(comfy_background_server):
    async with AsyncRemoteComfyClient() as client:
        random_seed = random.randint(1, 4294967295)
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, seed=random_seed, refiner_steps=1)
        result_dict = await client.queue_prompt_ui(prompt)
    # should contain one output
    assert len(result_dict) == 1


@pytest.mark.asyncio
async def test_completes_prompt_with_image_urls(comfy_background_server):
    async with AsyncRemoteComfyClient() as client:
        random_seed = random.randint(1, 4294967295)
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, seed=random_seed, refiner_steps=1, filename_prefix="subdirtest/sdxl")
        result = await client.queue_prompt_api(prompt)
    assert len(result.urls) == 2
    for url_str in result.urls:
        url: URL = parse(url_str)
        assert url.hostname == "localhost" or url.hostname == "127.0.0.1" or url.hostname == "::1"
        assert url.pathname == "/view"
        search = {k: v for (k, v) in parse_qsl(url.search[1:])}
        assert str(search["filename"]).startswith("sdxl")
        assert search["subfolder"] == "subdirtest"
        assert search["type"] == "output"
        # get the actual image file and assert it works
        async with aiohttp.ClientSession() as session:
            async with session.get(url_str) as response:
                assert response.status == 200
                assert response.headers['Content-Type'] == 'image/png'
                content = await response.read()
                assert len(content) > 1000
    assert len(result.outputs) == 1
    assert len(result.outputs["13"]["images"]) == 1

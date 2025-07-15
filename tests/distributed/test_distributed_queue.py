import asyncio
import logging

logging.basicConfig(level=logging.ERROR)

import uuid
from typing import Callable

import jwt
import pytest
from aiohttp import ClientSession
from testcontainers.rabbitmq import RabbitMqContainer

from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy.client.embedded_comfy_client import Comfy
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner
from comfy.component_model.executor_types import Executor
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.queue_types import QueueItem, QueueTuple, TaskInvocation, NamedQueueTuple, ExecutionStatus
from comfy.distributed.distributed_prompt_worker import DistributedPromptWorker
from comfy.distributed.executors import ContextVarExecutor
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.distributed.server_stub import ServerStub


async def create_test_prompt() -> QueueItem:
    from comfy.cmd.execution import validate_prompt

    prompt = make_mutable(sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1))
    item_id = str(uuid.uuid4())

    validation_tuple = await validate_prompt(item_id, prompt)
    queue_tuple: QueueTuple = (0, item_id, prompt, {}, validation_tuple[2])
    return QueueItem(queue_tuple, None)


@pytest.mark.asyncio
async def test_sign_jwt_auth_none():
    client_id = str(uuid.uuid4())
    user_token_str = jwt.encode({"sub": client_id}, None, algorithm="none")
    user_token = jwt.decode(user_token_str, None, algorithms=["none"], options={"verify_signature": False})
    assert user_token["sub"] == client_id


_executor_factories: tuple[Executor] = (ContextVarExecutor, ProcessPoolExecutor)


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_factory", _executor_factories)
async def test_basic_queue_worker(executor_factory: Callable[..., Executor]) -> None:
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        async with DistributedPromptWorker(connection_uri=f"amqp://guest:guest@127.0.0.1:{params.port}", executor=executor_factory(max_workers=1)):
            # this unfortunately does a bunch of initialization on the test thread
            from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
            # now submit some jobs
            distributed_queue = DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=f"amqp://guest:guest@127.0.0.1:{params.port}")
            await distributed_queue.init()
            queue_item = await create_test_prompt()
            res: TaskInvocation = await distributed_queue.put_async(queue_item)
            assert res.item_id == queue_item.prompt_id
            assert len(res.outputs) == 1
            assert res.status is not None
            assert res.status.status_str == "success"
            await distributed_queue.close()


@pytest.mark.asyncio
async def test_distributed_prompt_queues_same_process():
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
        async with DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=connection_uri) as frontend:
            async with DistributedPromptQueue(ServerStub(), is_callee=True, is_caller=False, connection_uri=connection_uri) as worker:
                test_prompt = await create_test_prompt()
                test_prompt.completed = asyncio.Future()

                frontend.put(test_prompt)

                # start a worker thread
                thread_pool = ContextVarExecutor(max_workers=1)

                async def in_thread():
                    incoming, incoming_prompt_id = worker.get()
                    assert incoming is not None
                    incoming_named = NamedQueueTuple(incoming)
                    assert incoming_named.prompt_id == incoming_prompt_id
                    async with Comfy() as embedded_comfy_client:
                        outputs = await embedded_comfy_client.queue_prompt(incoming_named.prompt,
                                                                           incoming_named.prompt_id)
                    worker.task_done(incoming_named.prompt_id, outputs, ExecutionStatus("success", True, []))

                thread_pool.submit(lambda: asyncio.run(in_thread()))
                # this was completed over the comfyui queue interface, so it should be a task invocation
                frontend_pov_result: TaskInvocation = await test_prompt.completed
                assert frontend_pov_result is not None
                assert frontend_pov_result.item_id == test_prompt.prompt_id
                assert frontend_pov_result.outputs is not None
                assert len(frontend_pov_result.outputs) == 1
                assert frontend_pov_result.status is not None


@pytest.mark.asyncio
async def test_frontend_backend_workers(frontend_backend_worker_with_rabbitmq):
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
        png_image_bytes = await client.queue_prompt(prompt)
        len_queue_after = await client.len_queue()
    assert len_queue_after == 0
    assert len(png_image_bytes) > 1000, "expected an image, but got nothing"


@pytest.mark.asyncio
async def test_frontend_backend_workers_validation_error_raises(frontend_backend_worker_with_rabbitmq):
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1, sdxl_refiner_checkpoint_name="unknown.safetensors")
        with pytest.raises(Exception):
            await client.queue_prompt(prompt)


async def check_health(url: str, max_retries: int = 5, retry_delay: float = 1.0):
    async with ClientSession() as session:
        for _ in range(max_retries):
            try:
                async with session.get(url, timeout=1) as response:
                    if response.status == 200:
                        return True
            except Exception as exc_info:
                pass
            await asyncio.sleep(retry_delay)
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("executor_factory", _executor_factories)
async def test_basic_queue_worker_with_health_check(executor_factory):
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"
        health_check_port = 9090

        async with DistributedPromptWorker(connection_uri=connection_uri, health_check_port=health_check_port, executor=executor_factory(max_workers=1)) as worker:
            health_check_url = f"http://localhost:{health_check_port}/health"

            health_check_ok = await check_health(health_check_url)
            assert health_check_ok, "Health check server did not start properly"


@pytest.mark.asyncio
async def test_queue_and_forget_prompt_api_integration(frontend_backend_worker_with_rabbitmq):
    # Create the client using the server address from the fixture
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:

        # Create a test prompt
        prompt = sdxl_workflow_with_refiner("test prompt", inference_steps=1, refiner_steps=1)

        # Queue the prompt
        task_id = await client.queue_and_forget_prompt_api(prompt)

        assert task_id is not None, "Failed to get a valid task ID"

        # Poll for the result
        max_attempts = 60  # Increase max attempts for integration test
        poll_interval = 1  # Increase poll interval for integration test
        for _ in range(max_attempts):
            try:
                response = await client.session.get(f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts/{task_id}")
                if response.status == 200:
                    result = await response.json()
                    assert result is not None, "Received empty result"

                    # Find the first output node with images
                    output_node = next((node for node in result.values() if 'images' in node), None)
                    assert output_node is not None, "No output node with images found"

                    assert len(output_node['images']) > 0, "No images in output node"
                    assert 'filename' in output_node['images'][0], "No filename in image output"
                    assert 'subfolder' in output_node['images'][0], "No subfolder in image output"
                    assert 'type' in output_node['images'][0], "No type in image output"

                    # Check if we can access the image
                    image_url = f"{client.server_address}/view?filename={output_node['images'][0]['filename']}&type={output_node['images'][0]['type']}&subfolder={output_node['images'][0]['subfolder']}"
                    image_response = await client.session.get(image_url)
                    assert image_response.status == 200, f"Failed to retrieve image from {image_url}"

                    return  # Test passed
                elif response.status == 204:
                    await asyncio.sleep(poll_interval)
                else:
                    response.raise_for_status()
            except _:
                await asyncio.sleep(poll_interval)

    pytest.fail("Failed to get a 200 response with valid data within the timeout period")


class Worker(DistributedPromptWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processed_workflows: set[str] = set()

    async def on_will_complete_work_item(self, request: dict):
        workflow_id = request.get('prompt_id', 'unknown')
        self.processed_workflows.add(workflow_id)
        await super().on_will_complete_work_item(request)


@pytest.mark.asyncio
async def test_two_workers_distinct_requests():
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        # Start two test workers
        workers: list[Worker] = []
        for i in range(2):
            worker = Worker(connection_uri=connection_uri, health_check_port=9090 + i, executor=ProcessPoolExecutor(max_workers=1))
            await worker.init()
            workers.append(worker)

        from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
        queue = DistributedPromptQueue(is_callee=False, is_caller=True, connection_uri=connection_uri)
        await queue.init()

        # Submit two prompts
        task1 = asyncio.create_task(queue.put_async(await create_test_prompt()))
        task2 = asyncio.create_task(queue.put_async(await create_test_prompt()))

        # Wait for tasks to complete
        await asyncio.gather(task1, task2)

        # Clean up
        for worker in workers:
            await worker.close()
        await queue.close()

        # Assert that each worker processed exactly one distinct workflow
        all_workflows = set()
        for worker in workers:
            assert len(worker.processed_workflows) == 1, f"Worker processed {len(worker.processed_workflows)} workflows instead of 1"
            all_workflows.update(worker.processed_workflows)

        assert len(all_workflows) == 2, f"Expected 2 distinct workflows, but got {len(all_workflows)}"

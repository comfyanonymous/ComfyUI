import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

import jwt
import pytest
from aiohttp import ClientSession, ClientConnectorError
from testcontainers.rabbitmq import RabbitMqContainer

from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.queue_types import QueueItem, QueueTuple, TaskInvocation, NamedQueueTuple, ExecutionStatus
from comfy.distributed.distributed_prompt_worker import DistributedPromptWorker
from comfy.distributed.server_stub import ServerStub


def create_test_prompt() -> QueueItem:
    from comfy.cmd.execution import validate_prompt

    prompt = make_mutable(sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1))
    validation_tuple = validate_prompt(prompt)
    item_id = str(uuid.uuid4())
    queue_tuple: QueueTuple = (0, item_id, prompt, {}, validation_tuple[2])
    return QueueItem(queue_tuple, None)


@pytest.mark.asyncio
async def test_sign_jwt_auth_none():
    client_id = str(uuid.uuid4())
    user_token_str = jwt.encode({"sub": client_id}, None, algorithm="none")
    user_token = jwt.decode(user_token_str, None, algorithms=["none"], options={"verify_signature": False})
    assert user_token["sub"] == client_id


@pytest.mark.asyncio
async def test_basic_queue_worker() -> None:
    # there are lots of side effects from importing that we have to deal with

    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        async with DistributedPromptWorker(connection_uri=f"amqp://guest:guest@127.0.0.1:{params.port}"):
            # this unfortunately does a bunch of initialization on the test thread
            from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
            # now submit some jobs
            distributed_queue = DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=f"amqp://guest:guest@127.0.0.1:{params.port}")
            await distributed_queue.init()
            queue_item = create_test_prompt()
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
                test_prompt = create_test_prompt()
                test_prompt.completed = asyncio.Future()

                frontend.put(test_prompt)

                # start a worker thread
                thread_pool = ThreadPoolExecutor(max_workers=1)

                async def in_thread():
                    incoming, incoming_prompt_id = worker.get()
                    assert incoming is not None
                    incoming_named = NamedQueueTuple(incoming)
                    assert incoming_named.prompt_id == incoming_prompt_id
                    async with EmbeddedComfyClient() as embedded_comfy_client:
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
    client = AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq)
    prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
    png_image_bytes = await client.queue_prompt(prompt)
    len_queue_after = await client.len_queue()
    assert len_queue_after == 0
    assert len(png_image_bytes) > 1000, "expected an image, but got nothing"


@pytest.mark.asyncio
async def test_frontend_backend_workers_validation_error_raises(frontend_backend_worker_with_rabbitmq):
    client = AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq)

    prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1, sdxl_refiner_checkpoint_name="unknown.safetensors")
    with pytest.raises(Exception):
        await client.queue_prompt(prompt)


async def check_health(url: str, max_retries: int = 5, retry_delay: float = 1.0):
    async with ClientSession() as session:
        for _ in range(max_retries):
            try:
                async with session.get(url, timeout=1) as response:
                    if response.status == 200 and await response.text() == "OK":
                        return True
            except Exception as exc_info:
                pass
            await asyncio.sleep(retry_delay)
    return False


@pytest.mark.asyncio
async def test_basic_queue_worker_with_health_check():
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"
        health_check_port = 9090

        async with DistributedPromptWorker(connection_uri=connection_uri, health_check_port=health_check_port) as worker:
            # Test health check
            health_check_url = f"http://localhost:{health_check_port}/health"

            health_check_ok = await check_health(health_check_url)
            assert health_check_ok, "Health check server did not start properly"

            # Test the actual worker functionality
            from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
            distributed_queue = DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=connection_uri)
            await distributed_queue.init()

            queue_item = create_test_prompt()
            res = await distributed_queue.put_async(queue_item)

            assert res.item_id == queue_item.prompt_id
            assert len(res.outputs) == 1
            assert res.status is not None
            assert res.status.status_str == "success"

            await distributed_queue.close()

        # Test that the health check server is stopped after the worker is closed
        health_check_stopped = not await check_health(health_check_url, max_retries=1)
        assert health_check_stopped, "Health check server did not stop properly"


@pytest.mark.asyncio
async def test_health_check_port_conflict():
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"
        health_check_port = 9090

        # Start a simple server to occupy the health check port
        from aiohttp import web
        async def dummy_handler(request):
            return web.Response(text="Dummy")

        app = web.Application()
        app.router.add_get('/', dummy_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', health_check_port)
        await site.start()

        try:
            # Now try to start the DistributedPromptWorker
            async with DistributedPromptWorker(connection_uri=connection_uri, health_check_port=health_check_port) as worker:
                # The health check should be disabled, but the worker should still function
                from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
                distributed_queue = DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=connection_uri)
                await distributed_queue.init()

                queue_item = create_test_prompt()
                res = await distributed_queue.put_async(queue_item)

                assert res.item_id == queue_item.prompt_id
                assert len(res.outputs) == 1
                assert res.status is not None
                assert res.status.status_str == "success"

                await distributed_queue.close()

            # The original server should still be running
            async with ClientSession() as session:
                async with session.get(f"http://localhost:{health_check_port}") as response:
                    assert response.status == 200
                    assert await response.text() == "Dummy"

        finally:
            await runner.cleanup()

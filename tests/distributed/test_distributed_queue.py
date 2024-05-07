import asyncio
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

import jwt
import pytest
import requests
from testcontainers.rabbitmq import RabbitMqContainer

from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy.client.embedded_comfy_client import EmbeddedComfyClient
from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.queue_types import QueueItem, QueueTuple, TaskInvocation, NamedQueueTuple, ExecutionStatus
from comfy.distributed.distributed_prompt_worker import DistributedPromptWorker
from comfy.distributed.server_stub import ServerStub

# fixes issues with running the testcontainers rabbitmqcontainer on Windows
os.environ["TC_HOST"] = "localhost"


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
async def test_frontend_backend_workers():
    processes_to_close: List[subprocess.Popen] = []
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        try:
            params = rabbitmq.get_connection_params()
            connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

            frontend_command = [
                "comfyui",
                "--listen=0.0.0.0",
                "--port=9001",
                "--cpu",
                "--distributed-queue-frontend",
                f"--distributed-queue-connection-uri={connection_uri}",
            ]

            processes_to_close.append(subprocess.Popen(frontend_command, stdout=sys.stdout, stderr=sys.stderr))
            backend_command = [
                "comfyui-worker",
                "--port=9002",
                f"--distributed-queue-connection-uri={connection_uri}",
            ]

            processes_to_close.append(subprocess.Popen(backend_command, stdout=sys.stdout, stderr=sys.stderr))
            server_address = f"http://{get_lan_ip()}:8188"
            start_time = time.time()
            while time.time() - start_time < 60:
                try:
                    response = requests.get(server_address)
                    if response.status_code == 200:
                        break
                except ConnectionRefusedError:
                    pass
                except Exception as exc:
                    logging.warning("", exc_info=exc)
                time.sleep(1)

            client = AsyncRemoteComfyClient(server_address=server_address)
            prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
            png_image_bytes = await client.queue_prompt(prompt)
            assert len(png_image_bytes) > 1000, "expected an image, but got nothing"
        finally:
            for process in processes_to_close:
                process.terminate()


def get_lan_ip():
    """
    Finds the host's IP address on the LAN it's connected to.

    Returns:
        str: The IP address of the host on the LAN.
    """
    # Create a dummy socket
    s = None
    try:
        # Connect to a dummy address (Here, Google's public DNS server)
        # The actual connection is not made, but this allows finding out the LAN IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        if s is not None:
            s.close()
    return ip

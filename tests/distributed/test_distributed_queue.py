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


@pytest.mark.asyncio
async def test_api_error_reporting_blocking_request(frontend_backend_worker_with_rabbitmq):
    """Test error reporting with blocking request (no async preference)"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create an invalid prompt that will cause a validation error
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
        # Make the prompt invalid by referencing a non-existent checkpoint
        prompt["4"]["inputs"]["ckpt_name"] = "nonexistent_checkpoint.safetensors"

        # Post with blocking behavior (no prefer header for async)
        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)
        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as response:
            # Should return 400 for validation error (invalid checkpoint)
            assert response.status == 400, f"Expected 400, got {response.status}"
            error_body = await response.json()

            # Verify ValidationErrorDict structure per OpenAPI spec
            assert "type" in error_body, "Missing 'type' field in error response"
            assert "message" in error_body, "Missing 'message' field in error response"
            assert "details" in error_body, "Missing 'details' field in error response"
            assert "extra_info" in error_body, "Missing 'extra_info' field in error response"


@pytest.mark.asyncio
async def test_api_error_reporting_async_prefer_header(frontend_backend_worker_with_rabbitmq):
    """Test error reporting with Prefer: respond-async header"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a valid prompt structure but with invalid checkpoint
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
        prompt["4"]["inputs"]["ckpt_name"] = "nonexistent.safetensors"

        # Post with Prefer: respond-async header
        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)
        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Prefer': 'respond-async'
            }
        ) as response:
            # Should return 400 immediately for validation error
            assert response.status == 400, f"Expected 400 for validation error, got {response.status}"
            error_body = await response.json()
            assert "type" in error_body


@pytest.mark.asyncio
async def test_api_error_reporting_async_accept_mimetype(frontend_backend_worker_with_rabbitmq):
    """Test error reporting with +respond-async in Accept mimetype"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a prompt with validation error
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
        prompt["4"]["inputs"]["ckpt_name"] = "invalid_model.safetensors"

        # Post with +respond-async in Accept header
        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)
        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json+respond-async'
            }
        ) as response:
            # Should return 400 for validation error (happens before queuing)
            assert response.status == 400, f"Expected 400, got {response.status}"
            error_body = await response.json()
            assert "type" in error_body


@pytest.mark.asyncio
async def test_api_get_prompt_status_success(frontend_backend_worker_with_rabbitmq):
    """Test GET /api/v1/prompts/{prompt_id} returns 200 with Outputs on success"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a valid prompt
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)

        # Queue async to get prompt_id
        task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
        assert task_id is not None

        # Poll until done
        status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=1.0)

        # For a valid prompt, should get 200
        assert status_code == 200, f"Expected 200 for successful execution, got {status_code}"
        assert result is not None

        # Verify it returns outputs structure (dict with node IDs)
        assert isinstance(result, dict)
        assert len(result) > 0, "Expected non-empty outputs"


@pytest.mark.asyncio
async def test_api_get_prompt_status_404(frontend_backend_worker_with_rabbitmq):
    """Test GET /api/v1/prompts/{prompt_id} returns 404 for non-existent prompt"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Request a non-existent prompt ID
        fake_prompt_id = str(uuid.uuid4())

        async with await client.get_prompt_status(fake_prompt_id) as response:
            assert response.status == 404, f"Expected 404 for non-existent prompt, got {response.status}"


@pytest.mark.asyncio
async def test_api_get_prompt_status_204_in_progress(frontend_backend_worker_with_rabbitmq):
    """Test GET /api/v1/prompts/{prompt_id} returns 204 while prompt is in progress"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a prompt that takes some time to execute
        prompt = sdxl_workflow_with_refiner("test", inference_steps=10, refiner_steps=10)

        # Queue async
        task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")

        # Immediately check status (should be 204 or 200 if very fast)
        async with await client.get_prompt_status(task_id) as response:
            # Should be either 204 (in progress) or 200 (completed very fast)
            assert response.status in [200, 204], f"Expected 200 or 204, got {response.status}"

            if response.status == 204:
                # No content for in-progress
                content = await response.read()
                assert len(content) == 0 or content == b'', "Expected no content for 204 response"


@pytest.mark.asyncio
async def test_api_async_workflow_both_methods(frontend_backend_worker_with_rabbitmq):
    """Test full async workflow: queue with respond-async, then poll for completion"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a valid prompt
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)

        # Method 1: Prefer header
        task_id_1 = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
        assert task_id_1 is not None

        # Method 2: +respond-async in Accept header
        task_id_2 = await client.queue_and_forget_prompt_api(
            prompt, prefer_header=None, accept_header="application/json+respond-async"
        )
        assert task_id_2 is not None

        # Poll both until done
        status_1, result_1 = await client.poll_prompt_until_done(task_id_1, max_attempts=60, poll_interval=1.0)
        status_2, result_2 = await client.poll_prompt_until_done(task_id_2, max_attempts=60, poll_interval=1.0)

        # Both should succeed
        assert status_1 == 200, f"Task 1 failed with status {status_1}"
        assert status_2 == 200, f"Task 2 failed with status {status_2}"

        # Both should have outputs
        assert result_1 is not None and len(result_1) > 0
        assert result_2 is not None and len(result_2) > 0


@pytest.mark.asyncio
async def test_api_validation_error_structure(frontend_backend_worker_with_rabbitmq):
    """Test that validation errors return proper ValidationErrorDict structure"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create an invalid prompt (invalid checkpoint name)
        prompt = sdxl_workflow_with_refiner("test", "", 1, refiner_steps=1)
        prompt["4"]["inputs"]["ckpt_name"] = "fake.safetensors"

        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)

        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as response:
            assert response.status == 400, f"Expected 400, got {response.status}"

            error_body = await response.json()

            # Verify ValidationErrorDict structure per OpenAPI spec
            assert "type" in error_body, "Missing 'type'"
            assert "message" in error_body, "Missing 'message'"
            assert "details" in error_body, "Missing 'details'"
            assert "extra_info" in error_body, "Missing 'extra_info'"

            assert error_body["type"] == "prompt_outputs_failed_validation", "unexpected type"

            # extra_info should have exception_type and traceback
            assert "exception_type" in error_body["extra_info"], "Missing 'exception_type' in extra_info"
            assert "traceback" in error_body["extra_info"], "Missing 'traceback' in extra_info"
            assert isinstance(error_body["extra_info"]["traceback"], list), "traceback should be a list"

            # extra_info should have node_errors with detailed validation information
            assert "node_errors" in error_body["extra_info"], "Missing 'node_errors' in extra_info"
            node_errors = error_body["extra_info"]["node_errors"]
            assert isinstance(node_errors, dict), "node_errors should be a dict"
            assert len(node_errors) > 0, "node_errors should contain at least one node"

            # Verify node_errors structure for node "4" (CheckpointLoaderSimple with invalid ckpt_name)
            assert "4" in node_errors, "Node '4' should have validation errors"
            node_4_errors = node_errors["4"]
            assert "errors" in node_4_errors, "Node '4' should have 'errors' field"
            assert "class_type" in node_4_errors, "Node '4' should have 'class_type' field"
            assert "dependent_outputs" in node_4_errors, "Node '4' should have 'dependent_outputs' field"

            assert node_4_errors["class_type"] == "CheckpointLoaderSimple", "Node '4' class_type should be CheckpointLoaderSimple"
            assert len(node_4_errors["errors"]) > 0, "Node '4' should have at least one error"

            # Verify the error details include the validation error type and message
            first_error = node_4_errors["errors"][0]
            assert "type" in first_error, "Error should have 'type' field"
            assert "message" in first_error, "Error should have 'message' field"
            assert "details" in first_error, "Error should have 'details' field"
            assert first_error["type"] == "value_not_in_list", f"Expected 'value_not_in_list' error, got {first_error['type']}"
            assert "fake.safetensors" in first_error["details"], "Error details should mention 'fake.safetensors'"


@pytest.mark.asyncio
async def test_api_success_response_contract(frontend_backend_worker_with_rabbitmq):
    """Test that successful execution returns proper response structure"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a valid prompt
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)

        # Queue and wait for blocking response
        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)
        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as response:
            assert response.status == 200, f"Expected 200, got {response.status}"

            result = await response.json()

            # Should have 'outputs' key (and deprecated 'urls' key)
            assert "outputs" in result, "Missing 'outputs' in response"

            # outputs should be a dict with node IDs as keys
            outputs = result["outputs"]
            assert isinstance(outputs, dict), "outputs should be a dict"
            assert len(outputs) > 0, "outputs should not be empty"

            # Each output should follow the Output schema
            for node_id, output in outputs.items():
                assert isinstance(output, dict), f"Output for node {node_id} should be a dict"
                # Should have images or other output types
                if "images" in output:
                    assert isinstance(output["images"], list), f"images for node {node_id} should be a list"
                    for image in output["images"]:
                        assert "filename" in image, f"image missing 'filename' in node {node_id}"
                        assert "subfolder" in image, f"image missing 'subfolder' in node {node_id}"
                        assert "type" in image, f"image missing 'type' in node {node_id}"


@pytest.mark.asyncio
async def test_api_get_prompt_returns_outputs_directly(frontend_backend_worker_with_rabbitmq):
    """Test GET /api/v1/prompts/{prompt_id} returns Outputs directly (not wrapped in history entry)"""
    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create and queue a prompt
        prompt = sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1)
        task_id = await client.queue_and_forget_prompt_api(prompt)

        # Poll until done
        status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=1.0)

        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result is not None, "Result should not be None"

        # Per OpenAPI spec, GET should return Outputs directly, not wrapped
        # result should be a dict with node IDs as keys
        assert isinstance(result, dict), "Result should be a dict (Outputs)"

        # Should NOT have 'prompt', 'outputs', 'status' keys (those are in history entry)
        # Should have node IDs directly
        for key in result.keys():
            # Node IDs are typically numeric strings like "4", "13", etc.
            # Should not be "prompt", "outputs", "status"
            assert key not in ["prompt", "status"], \
                f"GET endpoint should return Outputs directly, not history entry. Found key: {key}"


@pytest.mark.asyncio
async def test_api_execution_error_blocking_mode(frontend_backend_worker_with_rabbitmq):
    """Test that execution errors (not validation) return proper error structure in blocking mode"""
    from comfy_execution.graph_utils import GraphBuilder

    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a prompt that will fail during execution (not validation)
        # Use Regex with a group name that doesn't exist - validation passes but execution fails
        g = GraphBuilder()
        regex_match = g.node("Regex", pattern="hello", string="hello world")
        # Request a non-existent group name - this will pass validation but fail during execution
        match_group = g.node("RegexMatchGroupByName", match=regex_match.out(0), name="nonexistent_group")
        g.node("SaveString", value=match_group.out(0), filename_prefix="test")

        prompt = g.finalize()
        prompt_json = client._AsyncRemoteComfyClient__json_encoder.encode(prompt)

        async with client.session.post(
            f"{frontend_backend_worker_with_rabbitmq}/api/v1/prompts",
            data=prompt_json,
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'}
        ) as response:
            # Execution errors return 500
            assert response.status == 500, f"Expected 500 for execution error, got {response.status}"

            error_body = await response.json()

            # Verify ExecutionStatus structure
            assert "status_str" in error_body, "Missing 'status_str'"
            assert "completed" in error_body, "Missing 'completed'"
            assert "messages" in error_body, "Missing 'messages'"

            assert error_body["status_str"] == "error", f"Expected 'error', got {error_body['status_str']}"
            assert error_body["completed"] == False, "completed should be False for errors"
            assert isinstance(error_body["messages"], list), "messages should be a list"
            assert len(error_body["messages"]) > 0, "messages should contain error details"


@pytest.mark.asyncio
async def test_api_execution_error_async_mode(frontend_backend_worker_with_rabbitmq):
    """Test that execution errors return proper error structure in respond-async mode"""
    from comfy_execution.graph_utils import GraphBuilder

    async with AsyncRemoteComfyClient(server_address=frontend_backend_worker_with_rabbitmq) as client:
        # Create a prompt that will fail during execution (not validation)
        # Use Regex with a group name that doesn't exist - validation passes but execution fails
        g = GraphBuilder()
        regex_match = g.node("Regex", pattern="hello", string="hello world")
        # Request a non-existent group name - this will pass validation but fail during execution
        match_group = g.node("RegexMatchGroupByName", match=regex_match.out(0), name="nonexistent_group")
        g.node("SaveString", value=match_group.out(0), filename_prefix="test")

        prompt = g.finalize()

        # Queue with respond-async
        task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
        assert task_id is not None, "Should get task_id in async mode"

        # Poll for completion
        status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=1.0)

        # In async mode with polling, errors come back as 200 with error in the response body
        # because the prompt was accepted (202) and we're just retrieving the completed result
        assert status_code in (200, 500), f"Expected 200 or 500, got {status_code}"

        if status_code == 500:
            # Error returned directly - should be ExecutionStatus
            assert "status_str" in result, "Missing 'status_str'"
            assert "completed" in result, "Missing 'completed'"
            assert "messages" in result, "Missing 'messages'"
            assert result["status_str"] == "error"
            assert result["completed"] == False
            assert len(result["messages"]) > 0
        else:
            # Error in successful response - result might be ExecutionStatus or empty outputs
            # If it's a dict with status info, verify it
            if "status_str" in result:
                assert result["status_str"] == "error"
                assert result["completed"] == False
                assert len(result["messages"]) > 0

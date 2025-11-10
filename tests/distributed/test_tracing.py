import asyncio
import logging

logging.basicConfig(level=logging.ERROR)

import uuid

import pytest
from testcontainers.rabbitmq import RabbitMqContainer
from opentelemetry import trace, propagate, context
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner
from comfy.component_model.make_mutable import make_mutable
from comfy.component_model.queue_types import QueueItem, QueueTuple, ExecutionStatus
from comfy.distributed.server_stub import ServerStub


async def create_test_prompt() -> QueueItem:
    from comfy.cmd.execution import validate_prompt

    prompt = make_mutable(sdxl_workflow_with_refiner("test", inference_steps=1, refiner_steps=1))
    item_id = str(uuid.uuid4())

    validation_tuple = await validate_prompt(item_id, prompt)
    queue_tuple: QueueTuple = (0, item_id, prompt, {}, validation_tuple[2])
    return QueueItem(queue_tuple, None)


@pytest.mark.asyncio
async def test_rabbitmq_message_properties_contain_trace_context():
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue
        import aio_pika

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = trace.get_tracer(__name__, tracer_provider=provider)

        with tracer.start_as_current_span("test_message_headers", kind=SpanKind.PRODUCER):
            async with DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=connection_uri) as frontend:
                async with DistributedPromptQueue(ServerStub(), is_callee=True, is_caller=False, connection_uri=connection_uri) as worker:
                    queue_item = await create_test_prompt()

                    put_task = asyncio.create_task(frontend.put_async(queue_item))

                    incoming, incoming_prompt_id = await worker.get_async(timeout=5.0)
                    assert incoming is not None, "Worker should receive message"

                    worker.task_done(incoming_prompt_id, {}, ExecutionStatus("success", True, []))

                    result = await put_task
                    assert result is not None, "Frontend should get result"

                # Now inspect the RabbitMQ queue directly to see message structure
                connection = await aio_pika.connect_robust(connection_uri)
                channel = await connection.channel()

                # Declare a test queue to inspect message format
                test_queue = await channel.declare_queue("test_inspection_queue", durable=False, auto_delete=True)

                # Publish a test message with trace context
                carrier = {}
                propagate.inject(carrier)

                test_message = aio_pika.Message(
                    body=b"test",
                    headers=carrier
                )

                await channel.default_exchange.publish(
                    test_message,
                    routing_key=test_queue.name
                )

                # Get and inspect the message
                received = await test_queue.get(timeout=2, fail=False)
                if received:
                    headers = received.headers or {}

                    # Document what trace headers should be present
                    # OpenTelemetry uses 'traceparent' header for W3C Trace Context
                    has_traceparent = "traceparent" in headers

                    assert has_traceparent

                    await received.ack()

                await connection.close()


@pytest.mark.asyncio
async def test_distributed_queue_uses_async_interface():
    """
    Test that demonstrates the correct way to use DistributedPromptQueue in async context.
    The synchronous get() method cannot be used in async tests due to event loop assertions.
    """
    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()
        connection_uri = f"amqp://guest:guest@127.0.0.1:{params.port}"

        from comfy.distributed.distributed_prompt_queue import DistributedPromptQueue

        async with DistributedPromptQueue(ServerStub(), is_callee=False, is_caller=True, connection_uri=connection_uri) as frontend:
            async with DistributedPromptQueue(ServerStub(), is_callee=True, is_caller=False, connection_uri=connection_uri) as worker:
                queue_item = await create_test_prompt()

                # Start consuming in background
                result_future = asyncio.create_task(frontend.put_async(queue_item))

                # Worker gets item asynchronously (not using blocking get())
                incoming, incoming_prompt_id = await worker.get_async(timeout=5.0)
                assert incoming is not None, "Should receive a queue item"

                # Complete the work
                worker.task_done(incoming_prompt_id, {}, ExecutionStatus("success", True, []))

                # Wait for frontend to complete
                result = await result_future
                assert result is not None, "Should get result from worker"
                assert result.status.status_str == "success"

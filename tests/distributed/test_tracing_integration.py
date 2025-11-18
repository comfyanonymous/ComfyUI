"""
Integration tests for distributed tracing across RabbitMQ and services.

These tests validate that trace context propagates correctly from frontend
to backend workers through RabbitMQ, and that Jaeger can reconstruct the
full distributed trace.
"""
import asyncio
import logging
import os
import subprocess
import tempfile
import time
import uuid

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.attributes import service_attributes
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs
from testcontainers.nginx import NginxContainer
from testcontainers.rabbitmq import RabbitMqContainer

from comfy.client.sdxl_with_refiner_workflow import sdxl_workflow_with_refiner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JaegerContainer(DockerContainer):
    """Testcontainer for Jaeger all-in-one with OTLP support."""

    def __init__(self, image: str = "jaegertracing/all-in-one:latest"):
        super().__init__(image)
        self.with_exposed_ports(16686, 4318, 14268)  # UI, OTLP HTTP, Jaeger HTTP
        self.with_env("COLLECTOR_OTLP_ENABLED", "true")

    def get_query_url(self) -> str:
        """Get Jaeger Query API URL."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(16686)
        return f"http://{host}:{port}"

    def get_otlp_endpoint(self) -> str:
        """Get OTLP HTTP endpoint for sending traces."""
        host = self.get_container_host_ip()
        port = self.get_exposed_port(4318)
        return f"http://{host}:{port}"

    def start(self):
        super().start()
        wait_for_logs(self, ".*Starting GRPC server.*", timeout=30)
        return self


@pytest.fixture(scope="function")
def nginx_proxy(frontend_backend_worker_with_rabbitmq):
    """
    Provide an nginx proxy in front of the ComfyUI frontend.
    This tests if nginx is blocking W3C trace context propagation.
    """
    import socket
    import subprocess

    # Extract host and port from frontend address
    frontend_url = frontend_backend_worker_with_rabbitmq
    # frontend_url is like "http://127.0.0.1:19001"
    import re
    match = re.match(r'http://([^:]+):(\d+)', frontend_url)
    if not match:
        raise ValueError(f"Could not parse frontend URL: {frontend_url}")

    frontend_host = match.group(1)
    frontend_port = match.group(2)
    nginx_port = 8085

    # Get the Docker bridge gateway IP (this is how containers reach the host on Linux)
    # Try to get the default Docker bridge gateway
    try:
        result = subprocess.run(
            ["docker", "network", "inspect", "bridge", "-f", "{{range .IPAM.Config}}{{.Gateway}}{{end}}"],
            capture_output=True,
            text=True,
            check=True
        )
        docker_gateway = result.stdout.strip()
        logger.info(f"Using Docker gateway IP: {docker_gateway}")
    except Exception as e:
        # Fallback: try common gateway IPs
        docker_gateway = "172.17.0.1"  # Default Docker bridge gateway on Linux
        logger.warning(f"Could not detect Docker gateway, using default: {docker_gateway}")

    # Create nginx config that proxies to the frontend and passes trace headers
    nginx_conf = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream backend {{
        server {docker_gateway}:{frontend_port};
    }}

    server {{
        listen {nginx_port};

        location / {{
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }}
    }}
}}
"""

    # Write config to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conf', delete=False) as f:
        f.write(nginx_conf)
        nginx_conf_path = f.name

    try:
        # Start nginx container with the config
        nginx = NginxContainer(port=nginx_port)
        nginx.with_volume_mapping(nginx_conf_path, "/etc/nginx/nginx.conf")
        nginx.start()

        # Get the nginx URL
        host = nginx.get_container_host_ip()
        port = nginx.get_exposed_port(nginx_port)
        nginx_url = f"http://{host}:{port}"

        logger.info(f"Nginx proxy started at {nginx_url} -> {frontend_url}")

        # Wait for nginx to be ready
        for _ in range(30):
            try:
                response = requests.get(nginx_url, timeout=1)
                if response.status_code:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        yield nginx_url
    finally:
        nginx.stop()
        os.unlink(nginx_conf_path)


@pytest.fixture(scope="module")
def jaeger_container():
    """
    Provide a Jaeger container for collecting traces.

    This fixture automatically sets OTEL_EXPORTER_OTLP_ENDPOINT to point to the
    Jaeger container, and cleans it up when the container stops.
    """
    container = JaegerContainer()
    container.start()

    # Wait for Jaeger to be fully ready
    query_url = container.get_query_url()
    otlp_endpoint = container.get_otlp_endpoint()

    for _ in range(30):
        try:
            response = requests.get(f"{query_url}/api/services")
            if response.status_code == 200:
                logger.info(f"Jaeger ready at {query_url}")
                logger.info(f"OTLP endpoint: {otlp_endpoint}")
                break
        except Exception:
            pass
        time.sleep(1)

    # Set OTEL_EXPORTER_OTLP_ENDPOINT for the duration of the test
    old_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    logger.info(f"Set OTEL_EXPORTER_OTLP_ENDPOINT={otlp_endpoint}")

    try:
        yield container
    finally:
        # Restore original OTEL_EXPORTER_OTLP_ENDPOINT
        if old_endpoint is not None:
            os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = old_endpoint
            logger.info(f"Restored OTEL_EXPORTER_OTLP_ENDPOINT={old_endpoint}")
        else:
            os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
            logger.info("Removed OTEL_EXPORTER_OTLP_ENDPOINT")

        container.stop()


def query_jaeger_traces(jaeger_url: str, service: str, operation: str = None,
                       lookback: str = "1h", limit: int = 100) -> dict:
    """
    Query Jaeger for traces.

    Args:
        jaeger_url: Base URL of Jaeger query service
        service: Service name to query
        operation: Optional operation name filter
        lookback: Lookback period (e.g., "1h", "30m")
        limit: Maximum number of traces to return

    Returns:
        JSON response from Jaeger API
    """
    params = {
        "service": service,
        "lookback": lookback,
        "limit": limit
    }
    if operation:
        params["operation"] = operation

    response = requests.get(f"{jaeger_url}/api/traces", params=params)
    response.raise_for_status()
    return response.json()


def find_trace_by_operation(traces_response: dict, operation_name: str) -> dict:
    """Find a specific trace by operation name."""
    for trace in traces_response.get("data", []):
        for span in trace.get("spans", []):
            if span.get("operationName") == operation_name:
                return trace
    return None


def verify_trace_continuity(trace: dict, expected_services: list[str]) -> bool:
    """
    Verify that a trace spans multiple services and maintains parent-child relationships.

    Args:
        trace: Jaeger trace object
        expected_services: List of service names expected in the trace

    Returns:
        True if trace shows proper distributed tracing across services
    """
    if not trace:
        return False

    spans = trace.get("spans", [])
    if not spans:
        return False

    # Check that all expected services are present
    trace_services = set()
    for span in spans:
        process_id = span.get("processID")
        if process_id:
            process = trace.get("processes", {}).get(process_id, {})
            service_name = process.get("serviceName")
            if service_name:
                trace_services.add(service_name)

    logger.info(f"Trace contains services: {trace_services}")
    logger.info(f"Expected services: {set(expected_services)}")

    # Verify all expected services are present
    for service in expected_services:
        if service not in trace_services:
            logger.warning(f"Expected service '{service}' not found in trace")
            return False

    # Verify all spans share the same trace ID
    trace_ids = set(span.get("traceID") for span in spans)
    if len(trace_ids) != 1:
        logger.warning(f"Multiple trace IDs found: {trace_ids}")
        return False

    # Verify parent-child relationships exist
    span_ids = {span.get("spanID") for span in spans}
    has_parent_refs = False

    for span in spans:
        references = span.get("references", [])
        for ref in references:
            if ref.get("refType") == "CHILD_OF":
                parent_span_id = ref.get("spanID")
                if parent_span_id in span_ids:
                    has_parent_refs = True
                    logger.info(f"Found parent-child relationship: {parent_span_id} -> {span.get('spanID')}")

    if not has_parent_refs:
        logger.warning("No parent-child relationships found in trace")
        return False

    return True


# order matters, execute jaeger_container first
@pytest.mark.asyncio
async def test_tracing_integration(jaeger_container, nginx_proxy):
    """
    Integration test for distributed tracing across services with nginx proxy.

    This test:
    1. Starts ComfyUI frontend and worker with RabbitMQ
    2. Starts nginx proxy in front of the frontend to test trace context propagation through nginx
    3. Configures OTLP export to Jaeger testcontainer
    4. Submits a workflow through the nginx proxy
    5. Queries Jaeger to verify trace propagation
    6. Validates that the trace spans multiple services with proper relationships

    This specifically tests if nginx is blocking W3C trace context (traceparent/tracestate headers).
    """
    server_address = nginx_proxy
    jaeger_url = jaeger_container.get_query_url()
    otlp_endpoint = jaeger_container.get_otlp_endpoint()

    logger.info(f"Frontend server: {server_address}")
    logger.info(f"Jaeger UI: {jaeger_url}")
    logger.info(f"OTLP endpoint: {otlp_endpoint}")

    # Set up tracing for the async HTTP client
    resource = Resource.create({
        service_attributes.SERVICE_NAME: "comfyui-client",
    })
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    from opentelemetry import trace

    trace.set_tracer_provider(provider)

    # Instrument aiohttp client
    AioHttpClientInstrumentor().instrument()

    # we have to call this very late, so that the instrumentation isn't initialized too early
    from comfy.client.aio_client import AsyncRemoteComfyClient

    # Note: In a real integration test, you'd need to configure the ComfyUI
    # services to export traces to this Jaeger instance. For now, this test
    # documents the expected behavior.

    # Create a unique prompt to identify our trace
    test_id = str(uuid.uuid4())[:8]
    prompt = sdxl_workflow_with_refiner(f"test_trace_{test_id}", inference_steps=1, refiner_steps=1)

    # Get the tracer for the client
    client_tracer = trace.get_tracer("test_tracing_integration")

    # Submit the workflow - wrap in a span to capture the trace ID
    with client_tracer.start_as_current_span("submit_workflow") as workflow_span:
        trace_id = format(workflow_span.get_span_context().trace_id, '032x')
        logger.info(f"Started trace with trace_id: {trace_id}")

        async with AsyncRemoteComfyClient(server_address=server_address) as client:
            logger.info(f"Submitting workflow with test_id: {test_id}")

            # Queue the prompt with async response
            task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
            assert task_id is not None, "Failed to get task ID"

            logger.info(f"Queued task: {task_id}")

            # Poll for completion
            status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=1.0)
            assert status_code == 200, f"Task failed with status {status_code}"
            logger.info("Task completed successfully")

    # Give Jaeger time to receive and process spans
    await asyncio.sleep(5)

    # Query Jaeger for traces
    # Note: The actual service names depend on how your services are configured
    # Common service names might be: "slack-bot", "comfyui-frontend", "comfyui-worker"

    expected_services = ["comfyui", "comfyui-client"]  # Adjust based on actual service names

    logger.info(f"Querying Jaeger for traces with trace_id: {trace_id}...")

    # First, try to find our specific trace by trace_id from the client service
    our_trace = None
    for service in expected_services:
        try:
            traces_response = query_jaeger_traces(jaeger_url, service, lookback="5m")
            if traces_response.get("data"):
                logger.info(f"Found {len(traces_response['data'])} traces for service '{service}'")
                for trace in traces_response["data"]:
                    if trace.get("traceID") == trace_id:
                        our_trace = trace
                        logger.info(f"Found our trace in service '{service}'")
                        break
            if our_trace:
                break
        except Exception as e:
            logger.warning(f"Could not query traces for service '{service}': {e}")

    # Assert we can find the trace we just created
    assert our_trace is not None, (
        f"Could not find trace with trace_id {trace_id} in Jaeger. "
        f"This indicates that spans from comfyui-client are not being exported correctly."
    )

    logger.info(f"Successfully found trace with trace_id {trace_id}")

    # Extract services from the trace
    trace_services = set()
    for span in our_trace.get("spans", []):
        process_id = span.get("processID")
        if process_id:
            process = our_trace.get("processes", {}).get(process_id, {})
            service_name = process.get("serviceName")
            if service_name:
                trace_services.add(service_name)

    logger.info(f"Services found in trace: {trace_services}")

    # Assert that comfyui-client service is present (since we instrumented it)
    assert "comfyui-client" in trace_services, (
        f"Expected 'comfyui-client' service in trace, but found only: {trace_services}. "
        f"This indicates the client instrumentation is not working."
    )

    # Validate trace structure
    logger.info(f"Analyzing trace with {len(our_trace.get('spans', []))} spans")

    # Log all spans for debugging
    for span in our_trace.get("spans", []):
        process_id = span.get("processID")
        process = our_trace.get("processes", {}).get(process_id, {})
        service_name = process.get("serviceName", "unknown")
        operation = span.get("operationName", "unknown")
        logger.info(f"  Span: {service_name}.{operation}")

    # Verify trace continuity - only if both services are present
    assert "comfyui" in trace_services
    is_continuous = verify_trace_continuity(our_trace, expected_services)

    # This assertion documents what SHOULD happen when distributed tracing works
    assert is_continuous, (
        "Trace does not show proper distributed tracing. "
        "Expected to see spans from multiple services with parent-child relationships. "
        "This indicates that trace context is not being propagated correctly through RabbitMQ."
    )

@pytest.mark.asyncio
async def test_trace_context_in_http_headers(frontend_backend_worker_with_rabbitmq):
    """
    Test that HTTP requests include traceparent headers.

    This validates that the HTTP layer is properly instrumented for tracing.
    """
    server_address = frontend_backend_worker_with_rabbitmq

    # Make a simple HTTP request and check for trace headers
    # Note: We're checking the server's response headers to see if it's trace-aware
    response = requests.get(f"{server_address}/system_stats")

    logger.info(f"Response headers: {dict(response.headers)}")

    # The server should be instrumented and may include trace context in responses
    # or at minimum, should accept traceparent headers in requests

    # Test sending a traceparent header
    test_traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    response_with_trace = requests.get(
        f"{server_address}/system_stats",
        headers={"traceparent": test_traceparent}
    )

    # Should not error when traceparent is provided
    assert response_with_trace.status_code == 200, "Server should accept traceparent header"

    logger.info("✓ Server accepts traceparent headers in HTTP requests")


@pytest.mark.asyncio
async def test_multiple_requests_different_traces(frontend_backend_worker_with_rabbitmq, jaeger_container):
    """
    Test that multiple independent requests create separate traces.

    This validates that trace context is properly scoped per request.
    """
    server_address = frontend_backend_worker_with_rabbitmq

    # Submit multiple workflows
    task_ids = []

    from comfy.client.aio_client import AsyncRemoteComfyClient
    async with AsyncRemoteComfyClient(server_address=server_address) as client:
        for i in range(3):
            prompt = sdxl_workflow_with_refiner(f"test_{i}", inference_steps=1, refiner_steps=1)
            task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
            task_ids.append(task_id)
            logger.info(f"Queued task {i}: {task_id}")

        # Wait for all to complete
        for i, task_id in enumerate(task_ids):
            status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=1.0)
            assert status_code == 200, f"Task {i} failed"
            logger.info(f"Task {i} completed")

    # Give Jaeger time to receive spans
    await asyncio.sleep(5)

    # Query Jaeger and verify we have multiple distinct traces
    jaeger_url = jaeger_container.get_query_url()

    traces_response = query_jaeger_traces(jaeger_url, "comfyui", lookback="5m", limit=10)
    traces = traces_response.get("data", [])

    assert len(traces) >= 2
    # Get trace IDs
    trace_ids = [trace.get("traceID") for trace in traces]
    unique_trace_ids = set(trace_ids)

    logger.info(f"Found {len(unique_trace_ids)} unique traces")

    # Verify we have multiple distinct traces
    assert len(unique_trace_ids) >= 2, (
        f"Expected at least 2 distinct traces, found {len(unique_trace_ids)}. "
        "Each request should create its own trace."
    )

    logger.info("✓ Multiple requests created distinct traces")


@pytest.mark.asyncio
async def test_trace_contains_rabbitmq_operations(frontend_backend_worker_with_rabbitmq, jaeger_container):
    """
    Test that traces include RabbitMQ publish/consume operations.

    This is critical for distributed tracing - the RabbitMQ operations
    are what link the frontend and backend spans together.
    """
    server_address = frontend_backend_worker_with_rabbitmq
    jaeger_url = jaeger_container.get_query_url()

    # Submit a workflow
    from comfy.client.aio_client import AsyncRemoteComfyClient
    async with AsyncRemoteComfyClient(server_address=server_address) as client:
        prompt = sdxl_workflow_with_refiner("test_rmq", inference_steps=1, refiner_steps=1)
        task_id = await client.queue_and_forget_prompt_api(prompt)
        status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60)
        assert status_code == 200

    await asyncio.sleep(5)

    traces_response = query_jaeger_traces(jaeger_url, "comfyui", lookback="5m")
    traces = traces_response.get("data", [])

    # Look for RabbitMQ-related operations in any trace
    rabbitmq_operations = [
        "publish", "consume", "amq_queue_publish", "amq_queue_consume",
        "amq.basic.publish", "amq.basic.consume", "send", "receive"
    ]

    found_rabbitmq_ops = []
    for trace in traces:
        for span in trace.get("spans", []):
            op_name = span.get("operationName", "").lower()
            for rmq_op in rabbitmq_operations:
                if rmq_op in op_name:
                    found_rabbitmq_ops.append(op_name)

    assert found_rabbitmq_ops, "No RabbitMQ-related operations found in traces"


@pytest.mark.asyncio
async def test_aiohttp_and_aio_pika_spans_with_docker_frontend(jaeger_container):
    """
    Test that both aiohttp and aio_pika instrumentation work in the Docker image.

    This test helps diagnose if there's a dependency issue in the Docker image preventing
    instrumentation from working correctly by:
    1. Starting the ComfyUI frontend in a Docker container
    2. Starting a local worker process
    3. Submitting a workflow
    4. Querying Jaeger to verify both aiohttp and aio_pika spans are present

    Set COMFYUI_IMAGE env var to override default image, e.g.:
    COMFYUI_IMAGE=ghcr.io/hiddenswitch/comfyui:latest
    """
    docker_image = os.environ.get("COMFYUI_IMAGE", "ghcr.io/hiddenswitch/comfyui:latest")

    jaeger_url = jaeger_container.get_query_url()
    otlp_endpoint = jaeger_container.get_otlp_endpoint()
    otlp_port = jaeger_container.get_exposed_port(4318)

    with RabbitMqContainer("rabbitmq:latest") as rabbitmq:
        params = rabbitmq.get_connection_params()

        # Get Docker bridge gateway for container-to-host communication
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", "bridge", "-f", "{{(index .IPAM.Config 0).Gateway}}"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            docker_host = result.stdout.strip()
            if not docker_host:
                docker_host = "host.docker.internal"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            docker_host = "host.docker.internal"

        connection_uri_container = f"amqp://guest:guest@{docker_host}:{params.port}"
        connection_uri_local = f"amqp://guest:guest@127.0.0.1:{params.port}"

        # Start frontend in Docker container
        frontend_container = DockerContainer(docker_image)
        frontend_container.with_exposed_ports(8188)

        otlp_endpoint_container = f"http://{docker_host}:{otlp_port}"
        env_vars = {
            "OTEL_SERVICE_NAME": "comfyui-docker-frontend",
            "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint_container,
        }

        for key, value in env_vars.items():
            frontend_container.with_env(key, value)

        frontend_container.with_command(
            f"python -m comfy.cmd.main --listen 0.0.0.0 --port 8188 "
            f"--cpu --distributed-queue-frontend "
            f"--distributed-queue-connection-uri={connection_uri_container}"
        )

        frontend_container.start()

        try:
            frontend_host = frontend_container.get_container_host_ip()
            frontend_port = frontend_container.get_exposed_port(8188)
            frontend_url = f"http://{frontend_host}:{frontend_port}"

            # Wait for frontend to be ready
            connected = False
            for _ in range(15):
                try:
                    response = requests.get(frontend_url, timeout=1)
                    if response.status_code == 200:
                        connected = True
                        break
                except Exception:
                    pass
                time.sleep(1)

            assert connected, f"Could not connect to Docker frontend at {frontend_url}"

            # Start local worker
            worker_env = os.environ.copy()
            worker_env["OTEL_SERVICE_NAME"] = "comfyui-worker"
            worker_env["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint

            worker_process = subprocess.Popen(
                [
                    "comfyui-worker",
                    "--port=19099",
                    f"--distributed-queue-connection-uri={connection_uri_local}",
                    "--executor-factory=ThreadPoolExecutor"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=worker_env,
                text=True,
                bufsize=1
            )

            try:
                time.sleep(5)

                from comfy.client.aio_client import AsyncRemoteComfyClient

                test_id = str(uuid.uuid4())[:8]
                prompt = sdxl_workflow_with_refiner(f"docker_test_{test_id}", inference_steps=1, refiner_steps=1)

                async with AsyncRemoteComfyClient(server_address=frontend_url) as client:
                    task_id = await client.queue_and_forget_prompt_api(prompt, prefer_header="respond-async")
                    assert task_id is not None, "Failed to get task ID"

                    status_code, result = await client.poll_prompt_until_done(task_id, max_attempts=60, poll_interval=2.0)

                    if status_code != 200:
                        # Capture worker logs
                        worker_output = ""
                        if worker_process.stdout:
                            worker_output = worker_process.stdout.read()

                        # Get frontend container logs
                        frontend_logs = frontend_container.get_logs()

                        logger.error("=" * 80)
                        logger.error("TASK FAILED - Diagnostic Information:")
                        logger.error("=" * 80)
                        logger.error(f"Task ID: {task_id}")
                        logger.error(f"Status Code: {status_code}")
                        logger.error(f"Result: {result}")
                        logger.error("\n--- Frontend Container Logs (last 100 lines) ---")
                        frontend_log_lines = frontend_logs.decode('utf-8').split('\n')
                        for line in frontend_log_lines[-100:]:
                            logger.error(line)
                        logger.error("\n--- Worker Process Output ---")
                        for line in worker_output.split('\n')[-100:]:
                            logger.error(line)
                        logger.error("=" * 80)

                    assert status_code == 200, f"Task failed with status {status_code}. Check logs above for details."

                await asyncio.sleep(5)

                # Query Jaeger for traces from both services
                frontend_traces = query_jaeger_traces(jaeger_url, "comfyui-docker-frontend", lookback="5m").get("data", [])
                worker_traces = query_jaeger_traces(jaeger_url, "comfyui-worker", lookback="5m").get("data", [])

                assert frontend_traces, (
                    f"No traces found in Jaeger for service 'comfyui-docker-frontend'. "
                    f"Check that OTEL export is working from Docker container. Jaeger UI: {jaeger_url}"
                )

                assert worker_traces, (
                    f"No traces found in Jaeger for service 'comfyui-worker'. "
                    f"Check that OTEL export is working from worker. Jaeger UI: {jaeger_url}"
                )

                # Analyze span types from both services
                aiohttp_spans = []
                aio_pika_frontend_spans = []
                aio_pika_worker_spans = []

                for trace_item in frontend_traces:
                    for span in trace_item.get("spans", []):
                        operation_name = span.get("operationName", "")
                        if any(http_op in operation_name.upper() for http_op in ["GET", "POST", "PUT", "DELETE", "PATCH"]):
                            aiohttp_spans.append(operation_name)
                        elif "publish" in operation_name.lower() or "send" in operation_name.lower():
                            aio_pika_frontend_spans.append(operation_name)

                for trace_item in worker_traces:
                    for span in trace_item.get("spans", []):
                        operation_name = span.get("operationName", "")
                        if "consume" in operation_name.lower() or "receive" in operation_name.lower() or "publish" in operation_name.lower():
                            aio_pika_worker_spans.append(operation_name)

                assert aiohttp_spans, (
                    f"No aiohttp server spans found in traces from Docker frontend. "
                    f"This indicates aiohttp server instrumentation is not working in the Docker image. "
                    f"Image: {docker_image}. Jaeger UI: {jaeger_url}"
                )

                total_aio_pika_spans = len(aio_pika_frontend_spans) + len(aio_pika_worker_spans)
                assert total_aio_pika_spans > 0, (
                    f"No aio_pika spans found in traces. "
                    f"Frontend aio_pika spans: {len(aio_pika_frontend_spans)}, Worker aio_pika spans: {len(aio_pika_worker_spans)}. "
                    f"Expected messaging spans for distributed queue operations. "
                    f"This indicates aio_pika instrumentation is not working. Jaeger UI: {jaeger_url}"
                )

            finally:
                worker_process.terminate()
                worker_process.wait(timeout=10)

        finally:
            frontend_container.stop()
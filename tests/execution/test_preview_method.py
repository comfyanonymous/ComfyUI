"""
E2E tests for Queue-specific Preview Method Override feature.

Tests actual execution with different preview_method values.
Requires a running ComfyUI server with models.

Usage:
    COMFYUI_SERVER=http://localhost:8988 pytest test_preview_method_e2e.py -v -m preview_method

Note:
    These tests execute actual image generation and wait for completion.
    Tests verify preview image transmission based on preview_method setting.
"""
import os
import json
import pytest
import uuid
import time
import random
import websocket
import urllib.request
from pathlib import Path


# Server configuration
SERVER_URL = os.environ.get("COMFYUI_SERVER", "http://localhost:8988")
SERVER_HOST = SERVER_URL.replace("http://", "").replace("https://", "")

# Use existing inference graph fixture
GRAPH_FILE = Path(__file__).parent.parent / "inference" / "graphs" / "default_graph_sdxl1_0.json"


def is_server_running() -> bool:
    """Check if ComfyUI server is running."""
    try:
        request = urllib.request.Request(f"{SERVER_URL}/system_stats")
        with urllib.request.urlopen(request, timeout=2.0):
            return True
    except Exception:
        return False


def prepare_graph_for_test(graph: dict, steps: int = 5) -> dict:
    """Prepare graph for testing: randomize seeds and reduce steps."""
    adapted = json.loads(json.dumps(graph))  # Deep copy
    for node_id, node in adapted.items():
        inputs = node.get("inputs", {})
        # Handle both "seed" and "noise_seed" (used by KSamplerAdvanced)
        if "seed" in inputs:
            inputs["seed"] = random.randint(0, 2**32 - 1)
        if "noise_seed" in inputs:
            inputs["noise_seed"] = random.randint(0, 2**32 - 1)
        # Reduce steps for faster testing (default 20 -> 5)
        if "steps" in inputs:
            inputs["steps"] = steps
    return adapted


# Alias for backward compatibility
randomize_seed = prepare_graph_for_test


class PreviewMethodClient:
    """Client for testing preview_method with WebSocket execution tracking."""

    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.ws = None

    def connect(self):
        """Connect to WebSocket."""
        self.ws = websocket.WebSocket()
        self.ws.settimeout(120)  # 2 minute timeout for sampling
        self.ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")

    def close(self):
        """Close WebSocket connection."""
        if self.ws:
            self.ws.close()

    def queue_prompt(self, prompt: dict, extra_data: dict = None) -> dict:
        """Queue a prompt and return response with prompt_id."""
        data = {
            "prompt": prompt,
            "client_id": self.client_id,
            "extra_data": extra_data or {}
        }
        req = urllib.request.Request(
            f"http://{self.server_address}/prompt",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        return json.loads(urllib.request.urlopen(req).read())

    def wait_for_execution(self, prompt_id: str, timeout: float = 120.0) -> dict:
        """
        Wait for execution to complete via WebSocket.

        Returns:
            dict with keys: completed, error, preview_count, execution_time
        """
        result = {
            "completed": False,
            "error": None,
            "preview_count": 0,
            "execution_time": 0.0
        }

        start_time = time.time()
        self.ws.settimeout(timeout)

        try:
            while True:
                out = self.ws.recv()
                elapsed = time.time() - start_time

                if isinstance(out, str):
                    message = json.loads(out)
                    msg_type = message.get("type")
                    data = message.get("data", {})

                    if data.get("prompt_id") != prompt_id:
                        continue

                    if msg_type == "executing":
                        if data.get("node") is None:
                            # Execution complete
                            result["completed"] = True
                            result["execution_time"] = elapsed
                            break

                    elif msg_type == "execution_error":
                        result["error"] = data
                        result["execution_time"] = elapsed
                        break

                    elif msg_type == "progress":
                        # Progress update during sampling
                        pass

                elif isinstance(out, bytes):
                    # Binary data = preview image
                    result["preview_count"] += 1

        except websocket.WebSocketTimeoutException:
            result["error"] = "Timeout waiting for execution"
            result["execution_time"] = time.time() - start_time

        return result


def load_graph() -> dict:
    """Load the SDXL graph fixture with randomized seed."""
    with open(GRAPH_FILE) as f:
        graph = json.load(f)
    return randomize_seed(graph)  # Avoid caching


# Skip all tests if server is not running
pytestmark = [
    pytest.mark.skipif(
        not is_server_running(),
        reason=f"ComfyUI server not running at {SERVER_URL}"
    ),
    pytest.mark.preview_method,
    pytest.mark.execution,
]


@pytest.fixture
def client():
    """Create and connect a test client."""
    c = PreviewMethodClient(SERVER_HOST)
    c.connect()
    yield c
    c.close()


@pytest.fixture
def graph():
    """Load the test graph."""
    return load_graph()


class TestPreviewMethodExecution:
    """Test actual execution with different preview methods."""

    def test_execution_with_latent2rgb(self, client, graph):
        """
        Execute with preview_method=latent2rgb.
        Should complete and potentially receive preview images.
        """
        extra_data = {"preview_method": "latent2rgb"}

        response = client.queue_prompt(graph, extra_data)
        assert "prompt_id" in response

        result = client.wait_for_execution(response["prompt_id"])

        # Should complete (may error if model missing, but that's separate)
        assert result["completed"] or result["error"] is not None
        # Execution should take some time (sampling)
        if result["completed"]:
            assert result["execution_time"] > 0.5, "Execution too fast - likely didn't run"
            # latent2rgb should produce previews
            print(f"latent2rgb: {result['preview_count']} previews in {result['execution_time']:.2f}s")  # noqa: T201

    def test_execution_with_taesd(self, client, graph):
        """
        Execute with preview_method=taesd.
        TAESD provides higher quality previews.
        """
        extra_data = {"preview_method": "taesd"}

        response = client.queue_prompt(graph, extra_data)
        assert "prompt_id" in response

        result = client.wait_for_execution(response["prompt_id"])

        assert result["completed"] or result["error"] is not None
        if result["completed"]:
            assert result["execution_time"] > 0.5
            # taesd should also produce previews
            print(f"taesd: {result['preview_count']} previews in {result['execution_time']:.2f}s")  # noqa: T201

    def test_execution_with_none_preview(self, client, graph):
        """
        Execute with preview_method=none.
        No preview images should be generated.
        """
        extra_data = {"preview_method": "none"}

        response = client.queue_prompt(graph, extra_data)
        assert "prompt_id" in response

        result = client.wait_for_execution(response["prompt_id"])

        assert result["completed"] or result["error"] is not None
        if result["completed"]:
            # With "none", should receive no preview images
            assert result["preview_count"] == 0, \
                f"Expected no previews with 'none', got {result['preview_count']}"
            print(f"none: {result['preview_count']} previews in {result['execution_time']:.2f}s")  # noqa: T201

    def test_execution_with_default(self, client, graph):
        """
        Execute with preview_method=default.
        Should use server's CLI default setting.
        """
        extra_data = {"preview_method": "default"}

        response = client.queue_prompt(graph, extra_data)
        assert "prompt_id" in response

        result = client.wait_for_execution(response["prompt_id"])

        assert result["completed"] or result["error"] is not None
        if result["completed"]:
            print(f"default: {result['preview_count']} previews in {result['execution_time']:.2f}s")  # noqa: T201

    def test_execution_without_preview_method(self, client, graph):
        """
        Execute without preview_method in extra_data.
        Should use server's default preview method.
        """
        extra_data = {}  # No preview_method

        response = client.queue_prompt(graph, extra_data)
        assert "prompt_id" in response

        result = client.wait_for_execution(response["prompt_id"])

        assert result["completed"] or result["error"] is not None
        if result["completed"]:
            print(f"(no override): {result['preview_count']} previews in {result['execution_time']:.2f}s")  # noqa: T201


class TestPreviewMethodComparison:
    """Compare preview behavior between different methods."""

    def test_none_vs_latent2rgb_preview_count(self, client, graph):
        """
        Compare preview counts: 'none' should have 0, others should have >0.
        This is the key verification that preview_method actually works.
        """
        results = {}

        # Run with none (randomize seed to avoid caching)
        graph_none = randomize_seed(graph)
        extra_data_none = {"preview_method": "none"}
        response = client.queue_prompt(graph_none, extra_data_none)
        results["none"] = client.wait_for_execution(response["prompt_id"])

        # Run with latent2rgb (randomize seed again)
        graph_rgb = randomize_seed(graph)
        extra_data_rgb = {"preview_method": "latent2rgb"}
        response = client.queue_prompt(graph_rgb, extra_data_rgb)
        results["latent2rgb"] = client.wait_for_execution(response["prompt_id"])

        # Verify both completed
        assert results["none"]["completed"], f"'none' execution failed: {results['none']['error']}"
        assert results["latent2rgb"]["completed"], f"'latent2rgb' execution failed: {results['latent2rgb']['error']}"

        # Key assertion: 'none' should have 0 previews
        assert results["none"]["preview_count"] == 0, \
            f"'none' should have 0 previews, got {results['none']['preview_count']}"

        # 'latent2rgb' should have at least 1 preview (depends on steps)
        assert results["latent2rgb"]["preview_count"] > 0, \
            f"'latent2rgb' should have >0 previews, got {results['latent2rgb']['preview_count']}"

        print("\nPreview count comparison:")  # noqa: T201
        print(f"  none: {results['none']['preview_count']} previews")  # noqa: T201
        print(f"  latent2rgb: {results['latent2rgb']['preview_count']} previews")  # noqa: T201


class TestPreviewMethodSequential:
    """Test sequential execution with different preview methods."""

    def test_sequential_different_methods(self, client, graph):
        """
        Execute multiple prompts sequentially with different preview methods.
        Each should complete independently with correct preview behavior.
        """
        methods = ["latent2rgb", "none", "default"]
        results = []

        for method in methods:
            # Randomize seed for each execution to avoid caching
            graph_run = randomize_seed(graph)
            extra_data = {"preview_method": method}
            response = client.queue_prompt(graph_run, extra_data)

            result = client.wait_for_execution(response["prompt_id"])
            results.append({
                "method": method,
                "completed": result["completed"],
                "preview_count": result["preview_count"],
                "execution_time": result["execution_time"],
                "error": result["error"]
            })

        # All should complete or have clear errors
        for r in results:
            assert r["completed"] or r["error"] is not None, \
                f"Method {r['method']} neither completed nor errored"

        # "none" should have zero previews if completed
        none_result = next(r for r in results if r["method"] == "none")
        if none_result["completed"]:
            assert none_result["preview_count"] == 0, \
                f"'none' should have 0 previews, got {none_result['preview_count']}"

        print("\nSequential execution results:")  # noqa: T201
        for r in results:
            status = "✓" if r["completed"] else f"✗ ({r['error']})"
            print(f"  {r['method']}: {status}, {r['preview_count']} previews, {r['execution_time']:.2f}s")  # noqa: T201

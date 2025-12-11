"""
Tests for public ComfyAPI and ComfyAPISync functions.

These tests verify that the public API methods work correctly in both sync and async contexts,
ensuring that the sync wrapper generation (via get_type_hints() in async_to_sync.py) correctly
handles string annotations from 'from __future__ import annotations'.
"""

import pytest
from pytest import fixture

from comfy_execution.graph_utils import GraphBuilder
from tests.execution.common import ComfyClient, client_fixture


@pytest.mark.execution
class TestPublicAPI:
    # Initialize server and client
    client = fixture(client_fixture, scope="class", autouse=True)

    @fixture
    def builder(self, request):
        """Create GraphBuilder for each test."""
        yield GraphBuilder(prefix=request.node.name)

    async def test_sync_progress_update_executes(self, client: ComfyClient, builder: GraphBuilder):
        """Test that TestSyncProgressUpdate executes without errors.

        This test validates that api_sync.execution.set_progress() works correctly,
        which is the primary code path fixed by adding get_type_hints() to async_to_sync.py.
        """
        g = builder
        image = g.node("StubImage", content="BLACK", height=256, width=256, batch_size=1)

        # Use TestSyncProgressUpdate with short sleep
        progress_node = g.node("TestSyncProgressUpdate",
                               value=image.out(0),
                               sleep_seconds=0.5)
        output = g.node("SaveImage", images=progress_node.out(0))

        # Execute workflow
        result = await client.run(g)

        # Verify execution
        assert result.did_run(progress_node), "Progress node should have executed"
        assert result.did_run(output), "Output node should have executed"

        # Verify output
        images = result.get_images(output)
        assert len(images) == 1, "Should have produced 1 image"

    async def test_async_progress_update_executes(self, client: ComfyClient, builder: GraphBuilder):
        """Test that TestAsyncProgressUpdate executes without errors.

        This test validates that await api.execution.set_progress() works correctly
        in async contexts.
        """
        g = builder
        image = g.node("StubImage", content="WHITE", height=256, width=256, batch_size=1)

        # Use TestAsyncProgressUpdate with short sleep
        progress_node = g.node("TestAsyncProgressUpdate",
                               value=image.out(0),
                               sleep_seconds=0.5)
        output = g.node("SaveImage", images=progress_node.out(0))

        # Execute workflow
        result = await client.run(g)

        # Verify execution
        assert result.did_run(progress_node), "Async progress node should have executed"
        assert result.did_run(output), "Output node should have executed"

        # Verify output
        images = result.get_images(output)
        assert len(images) == 1, "Should have produced 1 image"

    async def test_sync_and_async_progress_together(self, client: ComfyClient, builder: GraphBuilder):
        """Test both sync and async progress updates in same workflow.

        This test ensures that both ComfyAPISync and ComfyAPI can coexist and work
        correctly in the same workflow execution.
        """
        g = builder
        image1 = g.node("StubImage", content="BLACK", height=256, width=256, batch_size=1)
        image2 = g.node("StubImage", content="WHITE", height=256, width=256, batch_size=1)

        # Use both types of progress nodes
        sync_progress = g.node("TestSyncProgressUpdate",
                               value=image1.out(0),
                               sleep_seconds=0.3)
        async_progress = g.node("TestAsyncProgressUpdate",
                                value=image2.out(0),
                                sleep_seconds=0.3)

        # Create outputs
        output1 = g.node("SaveImage", images=sync_progress.out(0))
        output2 = g.node("SaveImage", images=async_progress.out(0))

        # Execute workflow
        result = await client.run(g)

        # Both should execute successfully
        assert result.did_run(sync_progress), "Sync progress node should have executed"
        assert result.did_run(async_progress), "Async progress node should have executed"
        assert result.did_run(output1), "First output node should have executed"
        assert result.did_run(output2), "Second output node should have executed"

        # Verify outputs
        images1 = result.get_images(output1)
        images2 = result.get_images(output2)
        assert len(images1) == 1, "Should have produced 1 image from sync node"
        assert len(images2) == 1, "Should have produced 1 image from async node"

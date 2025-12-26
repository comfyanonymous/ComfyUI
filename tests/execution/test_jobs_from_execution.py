# Jobs API tests
import dataclasses

import pytest

from comfy.client.aio_client import AsyncRemoteComfyClient
from comfy_execution.graph_utils import GraphBuilder

@dataclasses.dataclass
class Result:
    res:dict

    def get_prompt_id(self):
        return self.res["prompt_id"]


class TestJobs:
    @pytest.fixture
    def builder(self, request):
        yield GraphBuilder(prefix=request.node.name)

    @pytest.fixture
    async def client(self, comfy_background_server_from_config):
        async with AsyncRemoteComfyClient(f"http://localhost:{comfy_background_server_from_config[0].port}") as obj:
            yield obj

    async def _create_history_item(self, client, builder):
        g = GraphBuilder(prefix="offset_test")
        input_node = g.node(
            "StubImage", content="BLACK", height=32, width=32, batch_size=1
        )
        g.node("SaveImage", images=input_node.out(0))
        await client.queue_prompt_api(g.finalize())

    async def test_jobs_api_job_structure(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test that job objects have required fields"""
        await self._create_history_item(client, builder)

        jobs_response = await client.get_jobs(status="completed", limit=1)
        assert len(jobs_response["jobs"]) > 0, "Should have at least one job"

        job = jobs_response["jobs"][0]
        assert "id" in job, "Job should have id"
        assert "status" in job, "Job should have status"
        assert "create_time" in job, "Job should have create_time"
        assert "outputs_count" in job, "Job should have outputs_count"
        assert "preview_output" in job, "Job should have preview_output"

    async def test_jobs_api_preview_output_structure(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test that preview_output has correct structure"""
        await self._create_history_item(client, builder)

        jobs_response = await client.get_jobs(status="completed", limit=1)
        job = jobs_response["jobs"][0]

        if job["preview_output"] is not None:
            preview = job["preview_output"]
            assert "filename" in preview, "Preview should have filename"
            assert "nodeId" in preview, "Preview should have nodeId"
            assert "mediaType" in preview, "Preview should have mediaType"

    async def test_jobs_api_pagination(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test jobs API pagination"""
        for _ in range(5):
            await self._create_history_item(client, builder)

        first_page = await client.get_jobs(limit=2, offset=0)
        second_page = await client.get_jobs(limit=2, offset=2)

        assert len(first_page["jobs"]) <= 2, "First page should have at most 2 jobs"
        assert len(second_page["jobs"]) <= 2, "Second page should have at most 2 jobs"

        first_ids = {j["id"] for j in first_page["jobs"]}
        second_ids = {j["id"] for j in second_page["jobs"]}
        assert first_ids.isdisjoint(second_ids), "Pages should have different jobs"

    async def test_jobs_api_sorting(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test jobs API sorting"""
        for _ in range(3):
            await self._create_history_item(client, builder)

        desc_jobs = await client.get_jobs(sort_order="desc")
        asc_jobs = await client.get_jobs(sort_order="asc")

        if len(desc_jobs["jobs"]) >= 2:
            desc_times = [j["create_time"] for j in desc_jobs["jobs"] if j["create_time"]]
            asc_times = [j["create_time"] for j in asc_jobs["jobs"] if j["create_time"]]
            if len(desc_times) >= 2:
                assert desc_times == sorted(desc_times, reverse=True), "Desc should be newest first"
            if len(asc_times) >= 2:
                assert asc_times == sorted(asc_times), "Asc should be oldest first"

    async def test_jobs_api_status_filter(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test jobs API status filtering"""
        await self._create_history_item(client, builder)

        completed_jobs = await client.get_jobs(status="completed")
        assert len(completed_jobs["jobs"]) > 0, "Should have completed jobs from history"

        for job in completed_jobs["jobs"]:
            assert job["status"] == "completed", "Should only return completed jobs"

        # Pending jobs are transient - just verify filter doesn't error
        pending_jobs = await client.get_jobs(status="pending")
        for job in pending_jobs["jobs"]:
            assert job["status"] == "pending", "Should only return pending jobs"

    async def test_get_job_by_id(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test getting a single job by ID"""
        result = await self._create_history_item(client, builder)
        prompt_id = result.get_prompt_id()

        job = await client.get_job(prompt_id)
        assert job is not None, "Should find the job"
        assert job["id"] == prompt_id, "Job ID should match"
        assert "outputs" in job, "Single job should include outputs"

    async def test_get_job_not_found(
            self, client: AsyncRemoteComfyClient, builder: GraphBuilder
    ):
        """Test getting a non-existent job returns 404"""
        job = await client.get_job("nonexistent-job-id")
        assert job is None, "Non-existent job should return None"

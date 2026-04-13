"""Automation routes for job scheduling."""
import asyncio
from functools import partial
from aiohttp import web
from custom_nodes.research.automation.schedule import scheduler, Job


class AutomationRoutes:
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: web.Application = None

    def get_app(self) -> web.Application:
        if self._app is None:
            self._app = web.Application()
            self._app.add_routes(self.routes)
        return self._app

    def setup_routes(self):
        # List all jobs
        @self.routes.get("/research/automation/jobs/")
        async def list_jobs(request):
            loop = asyncio.get_running_loop()
            jobs = await loop.run_in_executor(None, scheduler.list_jobs)
            return web.json_response(jobs)

        # Get a specific job
        @self.routes.get("/research/automation/jobs/{job_id}")
        async def get_job(request):
            job_id = request.match_info["job_id"]
            loop = asyncio.get_running_loop()
            job = await loop.run_in_executor(None, lambda: scheduler.get_job(job_id))
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({
                "id": job.id,
                "name": job.name,
                "schedule": {
                    "interval_hours": job.schedule.interval_hours,
                    "cron_hour": job.schedule.cron_hour,
                    "cron_minute": job.schedule.cron_minute,
                    "enabled": job.schedule.enabled,
                },
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_result": str(job.last_result) if job.last_result else None,
                "last_error": job.error,
                "run_count": job.run_count,
            })

        # Trigger a job manually
        @self.routes.post("/research/automation/jobs/{job_id}/run")
        async def trigger_job(request):
            job_id = request.match_info["job_id"]
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, lambda: scheduler.trigger_now(job_id))
            if result["status"] == "error" and "not found" in result.get("message", ""):
                return web.json_response(result, status=404)
            return web.json_response(result)

        # Enable/disable a job
        @self.routes.patch("/research/automation/jobs/{job_id}")
        async def update_job(request):
            job_id = request.match_info["job_id"]
            loop = asyncio.get_running_loop()
            job = await loop.run_in_executor(None, lambda: scheduler.get_job(job_id))
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)

            data = await request.json()

            if "enabled" in data:
                job.schedule.enabled = data["enabled"]
            if "interval_hours" in data:
                job.schedule.interval_hours = float(data["interval_hours"])
            if "cron_hour" in data:
                job.schedule.cron_hour = data["cron_hour"]
            if "cron_minute" in data:
                job.schedule.cron_minute = data["cron_minute"]

            return web.json_response({"status": "ok", "job_id": job_id})

        # Start/stop scheduler
        @self.routes.post("/research/automation/scheduler/start")
        async def start_scheduler(request):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, scheduler.start)
            return web.json_response({"status": "started"})

        @self.routes.post("/research/automation/scheduler/stop")
        async def stop_scheduler(request):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, scheduler.stop)
            return web.json_response({"status": "stopped"})

        @self.routes.get("/research/automation/scheduler/status")
        async def scheduler_status(request):
            loop = asyncio.get_running_loop()
            jobs = await loop.run_in_executor(None, scheduler.list_jobs)
            return web.json_response({
                "running": True,
                "jobs_count": len(jobs),
                "jobs": jobs,
            })

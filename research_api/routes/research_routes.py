"""Research API routes using aiohttp."""
import os
from aiohttp import web

from research_app.config import resolve_external_asset_path
from research_app.integrations import (
    ensure_default_jobs,
    get_default_keywords,
    get_feed_categories,
    get_scheduler,
)
from research_api.routes._db_helpers import (
    asyncio_get_projects,
    asyncio_create_project,
    asyncio_get_project,
    asyncio_update_project,
    asyncio_list_intents,
    asyncio_create_intent,
    asyncio_update_intent,
    asyncio_list_papers,
    asyncio_create_paper,
    asyncio_get_paper,
    asyncio_update_paper,
    asyncio_list_claims,
    asyncio_create_claim,
    asyncio_update_claim,
    asyncio_list_sources,
    asyncio_create_source,
    asyncio_update_source,
    asyncio_get_today_feed,
    asyncio_list_feed,
    asyncio_create_feed_item,
    asyncio_update_feed_item,
    asyncio_list_styles,
    asyncio_create_style,
    asyncio_get_style,
    asyncio_update_style,
    asyncio_delete_style,
    asyncio_run_feed_discovery,
)

# In-memory store for latest evidence graph (per session)
_latest_evidence_graph = {}


class ResearchRoutes:
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: web.Application = None
        self.scheduler = get_scheduler()
        self._jobs_registered = False

    def get_app(self) -> web.Application:
        if self._app is None:
            self._app = web.Application()
            self._app.add_routes(self.routes)
        return self._app

    def setup_routes(self):
        # Projects
        @self.routes.get("/research/projects/")
        async def list_projects(request):
            projects = await asyncio_get_projects()
            return web.json_response(projects)

        @self.routes.post("/research/projects/")
        async def create_project(request):
            data = await request.json()
            project = await asyncio_create_project(data)
            return web.json_response(project)

        @self.routes.get("/research/projects/{project_id}")
        async def get_project(request):
            project_id = request.match_info["project_id"]
            project = await asyncio_get_project(project_id)
            if not project:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(project)

        @self.routes.patch("/research/projects/{project_id}")
        async def update_project(request):
            project_id = request.match_info["project_id"]
            data = await request.json()
            project = await asyncio_update_project(project_id, data)
            if not project:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(project)

        @self.routes.get("/research/projects/{project_id}/intents")
        async def list_intents(request):
            project_id = request.match_info["project_id"]
            intents = await asyncio_list_intents(project_id)
            return web.json_response(intents)

        @self.routes.post("/research/projects/{project_id}/intents")
        async def create_intent(request):
            project_id = request.match_info["project_id"]
            data = await request.json()
            data["project_id"] = project_id
            intent = await asyncio_create_intent(data)
            return web.json_response(intent)

        @self.routes.patch("/research/projects/intents/{intent_id}")
        async def update_intent(request):
            intent_id = request.match_info["intent_id"]
            data = await request.json()
            intent = await asyncio_update_intent(intent_id, data)
            if not intent:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(intent)

        # Papers
        @self.routes.get("/research/papers/")
        async def list_papers(request):
            library_status = request.query.get("library_status")
            read_status = request.query.get("read_status")
            papers = await asyncio_list_papers(library_status, read_status)
            return web.json_response(papers)

        @self.routes.post("/research/papers/")
        async def create_paper(request):
            data = await request.json()
            paper = await asyncio_create_paper(data)
            return web.json_response(paper)

        @self.routes.get("/research/papers/{paper_id}")
        async def get_paper(request):
            paper_id = request.match_info["paper_id"]
            paper = await asyncio_get_paper(paper_id)
            if not paper:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(paper)

        @self.routes.patch("/research/papers/{paper_id}")
        async def update_paper(request):
            paper_id = request.match_info["paper_id"]
            data = await request.json()
            paper = await asyncio_update_paper(paper_id, data)
            if not paper:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(paper)

        @self.routes.get("/research/papers/{paper_id}/content")
        async def get_paper_content(request):
            paper_id = request.match_info["paper_id"]
            paper = await asyncio_get_paper(paper_id)
            if not paper:
                return web.json_response({"error": "Not found"}, status=404)

            content = {"type": paper.get("asset_type", "paper")}

            if paper.get("asset_type") == "paper" and paper.get("local_path"):
                import os
                local_path = paper.get("local_path")

                # Extract folder name from local_path for image URLs
                # local_path is like C:/r-assests/papers/FolderName or /c/r-assests/papers/FolderName
                folder_name = os.path.basename(local_path.rstrip('/\\'))
                content["folder_name"] = folder_name

                # Read meta.json if exists
                meta_path = os.path.join(local_path, "meta.json")
                if os.path.exists(meta_path):
                    import json
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        content["meta"] = json.load(f)

                # Read paper.md if exists
                md_path = os.path.join(local_path, "paper.md")
                if os.path.exists(md_path):
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content["markdown"] = f.read()

                # List images
                img_path = os.path.join(local_path, "images")
                if os.path.exists(img_path):
                    content["images"] = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.gif', '.jpeg'))]

            elif paper.get("asset_type") in ("figure", "table") and paper.get("local_path"):
                import os
                content["local_path"] = paper.get("local_path")
                if os.path.exists(paper.get("local_path")):
                    content["filename"] = os.path.basename(paper.get("local_path"))

            return web.json_response(content)

        @self.routes.get("/research/assets/{type}/{asset_id}")
        async def get_asset(request):
            """Get a specific asset by type and ID."""
            asset_type = request.match_info["type"]
            asset_id = request.match_info["asset_id"]

            if asset_type == "paper":
                paper = await asyncio_get_paper(asset_id)
                if not paper:
                    return web.json_response({"error": "Not found"}, status=404)
                return web.json_response({"type": "paper", **paper})
            elif asset_type == "claim":
                # asyncio_get_claim not yet implemented in _db_helpers
                return web.json_response({"error": "Not implemented"}, status=501)
            elif asset_type == "style":
                style = await asyncio_get_style(asset_id)
                if not style:
                    return web.json_response({"error": "Not found"}, status=404)
                return web.json_response({"type": "style", **style})
            else:
                return web.json_response({"error": "Unknown asset type"}, status=400)

        @self.routes.get("/research/assets/{path:.*}")
        async def get_asset_file(request):
            """Serve asset files from local paths."""
            file_path = request.match_info["path"]
            full_path = resolve_external_asset_path(file_path)
            if full_path is None:
                return web.json_response({"error": "Forbidden"}, status=403)

            if not full_path.exists():
                return web.json_response({"error": "Not found"}, status=404)

            if full_path.is_dir():
                # Return directory listing
                files = os.listdir(full_path)
                return web.json_response({"type": "directory", "files": files})

            # Return file
            from aiohttp.web_fileresponse import FileResponse
            return FileResponse(full_path)

        # Claims
        @self.routes.get("/research/claims/")
        async def list_claims(request):
            project_id = request.query.get("project_id")
            support_level = request.query.get("support_level")
            claims = await asyncio_list_claims(project_id, support_level)
            return web.json_response(claims)

        @self.routes.post("/research/claims/")
        async def create_claim(request):
            data = await request.json()
            claim = await asyncio_create_claim(data)
            return web.json_response(claim)

        @self.routes.patch("/research/claims/{claim_id}")
        async def update_claim(request):
            claim_id = request.match_info["claim_id"]
            data = await request.json()
            claim = await asyncio_update_claim(claim_id, data)
            if not claim:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(claim)

        # Sources
        @self.routes.get("/research/sources/")
        async def list_sources(request):
            sources = await asyncio_list_sources()
            return web.json_response(sources)

        @self.routes.post("/research/sources/")
        async def create_source(request):
            data = await request.json()
            source = await asyncio_create_source(data)
            return web.json_response(source)

        @self.routes.patch("/research/sources/{source_id}")
        async def update_source(request):
            source_id = request.match_info["source_id"]
            data = await request.json()
            source = await asyncio_update_source(source_id, data)
            if not source:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(source)

        # Feed
        @self.routes.get("/research/feed/today")
        async def get_today_feed(request):
            limit = int(request.query.get("limit", 50))
            offset = int(request.query.get("offset", 0))
            items = await asyncio_get_today_feed(limit=limit, offset=offset)
            return web.json_response(items)

        @self.routes.get("/research/feed/")
        async def list_feed(request):
            source_id = request.query.get("source_id")
            status = request.query.get("status")
            limit = int(request.query.get("limit", 50))
            offset = int(request.query.get("offset", 0))
            items = await asyncio_list_feed(source_id, status, limit=limit, offset=offset)
            return web.json_response(items)

        @self.routes.post("/research/feed/")
        async def create_feed_item(request):
            data = await request.json()
            item = await asyncio_create_feed_item(data)
            return web.json_response(item)

        @self.routes.patch("/research/feed/{item_id}")
        async def update_feed_item(request):
            item_id = request.match_info["item_id"]
            data = await request.json()
            item = await asyncio_update_feed_item(item_id, data)
            if not item:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(item)

        @self.routes.post("/research/feed/discover")
        async def discover_feed(request):
            """Discover new papers from configured journal sources using academic APIs."""
            data = await request.json() if request.can_read_body else {}
            categories = data.get("categories", get_feed_categories())
            keywords = data.get("keywords", get_default_keywords())
            limit_per_keyword = data.get("limit_per_keyword", 3)
            result = await asyncio_run_feed_discovery(categories, keywords, limit_per_keyword)
            return web.json_response(result)

        # Styles
        @self.routes.get("/research/assets/styles/")
        async def list_styles(request):
            styles = await asyncio_list_styles()
            return web.json_response(styles)

        @self.routes.post("/research/assets/styles/")
        async def create_style(request):
            data = await request.json()
            style = await asyncio_create_style(data)
            return web.json_response(style, status=201)

        @self.routes.get("/research/assets/styles/{style_id}")
        async def get_style(request):
            style_id = request.match_info["style_id"]
            style = await asyncio_get_style(style_id)
            if not style:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(style)

        @self.routes.patch("/research/assets/styles/{style_id}")
        async def update_style(request):
            style_id = request.match_info["style_id"]
            data = await request.json()
            style = await asyncio_update_style(style_id, data)
            if not style:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response(style)

        @self.routes.delete("/research/assets/styles/{style_id}")
        async def delete_style(request):
            style_id = request.match_info["style_id"]
            result = await asyncio_delete_style(style_id)
            if not result:
                return web.json_response({"error": "Not found"}, status=404)
            return web.json_response({"status": "deleted"})

        # Unified asset listing by type
        @self.routes.get("/research/assets/")
        async def list_assets(request):
            """List assets filtered by type: ?type=paper|claim|style"""
            asset_type = request.query.get("type")

            if asset_type == "paper" or not asset_type:
                papers = await asyncio_list_papers(
                    library_status=request.query.get("library_status"),
                    read_status=request.query.get("read_status")
                )
                return web.json_response([{"type": "paper", **p} for p in papers])
            elif asset_type == "claim":
                claims = await asyncio_list_claims(
                    project_id=request.query.get("project_id"),
                    support_level=request.query.get("support_level")
                )
                return web.json_response([{"type": "claim", **c} for c in claims])
            elif asset_type == "style":
                styles = await asyncio_list_styles()
                return web.json_response([{"type": "style", **s} for s in styles])
            else:
                return web.json_response({"error": "Unknown asset type"}, status=400)

        # Evidence Graph API
        @self.routes.get("/research/api/graph/latest")
        async def get_latest_graph(request):
            """Get the latest evidence graph from node execution."""
            return web.json_response(_latest_evidence_graph)

        @self.routes.post("/research/api/graph/latest")
        async def set_latest_graph(request):
            """Set the latest evidence graph from node execution."""
            global _latest_evidence_graph
            data = await request.json()
            _latest_evidence_graph = data
            return web.json_response({"status": "ok"})

        # Automation routes
        @self.routes.get("/research/automation/jobs/")
        async def list_jobs(request):
            jobs = self.scheduler.list_jobs()
            return web.json_response(jobs)

        @self.routes.get("/research/automation/jobs/{job_id}")
        async def get_job(request):
            job_id = request.match_info["job_id"]
            job = self.scheduler.get_job(job_id)
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

        @self.routes.post("/research/automation/jobs/{job_id}/run")
        async def trigger_job(request):
            job_id = request.match_info["job_id"]
            result = self.scheduler.trigger_now(job_id)
            if result["status"] == "error" and "not found" in result.get("message", ""):
                return web.json_response(result, status=404)
            return web.json_response(result)

        @self.routes.patch("/research/automation/jobs/{job_id}")
        async def update_job(request):
            job_id = request.match_info["job_id"]
            job = self.scheduler.get_job(job_id)
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

        @self.routes.post("/research/automation/scheduler/start")
        async def start_scheduler(request):
            self.scheduler.start()
            return web.json_response({"status": "started"})

        @self.routes.post("/research/automation/scheduler/stop")
        async def stop_scheduler(request):
            self.scheduler.stop()
            return web.json_response({"status": "stopped"})

        @self.routes.get("/research/automation/scheduler/status")
        async def scheduler_status(request):
            jobs = self.scheduler.list_jobs()
            return web.json_response({
                "running": True,
                "jobs_count": len(jobs),
                "jobs": jobs,
            })

        if not self._jobs_registered:
            ensure_default_jobs()
            self._jobs_registered = True

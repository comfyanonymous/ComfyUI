"""Research API routes using aiohttp."""
from aiohttp import web
from research_api.routes._db_helpers import (
    asyncio_get_projects,
    asyncio_create_project,
    asyncio_get_project,
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
)


class ResearchRoutes:
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: web.Application = None

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
            items = await asyncio_get_today_feed()
            return web.json_response(items)

        @self.routes.get("/research/feed/")
        async def list_feed(request):
            source_id = request.query.get("source_id")
            status = request.query.get("status")
            items = await asyncio_list_feed(source_id, status)
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

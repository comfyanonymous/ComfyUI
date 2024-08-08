from aiohttp import web

class InternalRoutes:
    def __init__(self):
        self.routes = web.RouteTableDef()
        self._app = None

    def setup_routes(self):
        @self.routes.get('/files/')
        async def internal_test(request):
            return web.json_response({"message": "Internal route is working"})

    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app
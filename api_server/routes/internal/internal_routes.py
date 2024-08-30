from aiohttp import web
from typing import Optional
from folder_paths import models_dir, user_directory, output_directory
from api_server.services.file_service import FileService
import app.logger

class InternalRoutes:
    '''
    The top level web router for internal routes: /internal/*
    The endpoints here should NOT be depended upon. It is for ComfyUI frontend use only.
    Check README.md for more information.
    
    '''
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.file_service = FileService({
            "models": models_dir,
            "user": user_directory,
            "output": output_directory
        })

    def setup_routes(self):
        @self.routes.get('/files')
        async def list_files(request):
            directory_key = request.query.get('directory', '')
            try:
                file_list = self.file_service.list_files(directory_key)
                return web.json_response({"files": file_list})
            except ValueError as e:
                return web.json_response({"error": str(e)}, status=400)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @self.routes.get('/logs')
        async def get_logs(request):
            return web.json_response(app.logger.get_logs())

    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app

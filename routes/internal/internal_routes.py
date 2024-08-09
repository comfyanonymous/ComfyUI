from aiohttp import web
from typing import Optional, Dict, List
from folder_paths import models_dir, user_directory, output_directory
import os 

class InternalRoutes:
    def __init__(self):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.allowed_directories: Dict[str, str] = {
            "models": models_dir,
            "user": user_directory,
            "output": output_directory
        }

    def setup_routes(self):
        @self.routes.get('/files')
        async def list_files(request):
            directory_key = request.query.get('directory', '')
            if directory_key not in self.allowed_directories:
                return web.json_response({"error": "Invalid directory key"}, status=400)

            directory_path = self.allowed_directories[directory_key]

            try:
                file_list = self._get_file_list(directory_path)
                return web.json_response({"files": file_list})
            except FileNotFoundError:
                return web.json_response({"error": "Directory not found"}, status=404)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

    def _get_file_list(self, directory: str) -> List[Dict[str, Optional[str]]]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        file_list = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)
                relative_path = os.path.relpath(file_path, directory)
                file_list.append({
                    "name": name,
                    "path": relative_path,
                    "type": "file",
                    "size": os.path.getsize(file_path)
                })
            for name in dirs:
                dir_path = os.path.join(root, name)
                relative_path = os.path.relpath(dir_path, directory)
                file_list.append({
                    "name": name,
                    "path": relative_path,
                    "type": "directory",
                    "size": None
                })
        return file_list


    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app

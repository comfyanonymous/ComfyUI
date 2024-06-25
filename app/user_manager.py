import json
import os
import re
import uuid
import glob
import shutil
from aiohttp import web
from comfy.cli_args import args
from folder_paths import user_directory
from .app_settings import AppSettings

default_user = "default"
users_file = os.path.join(user_directory, "users.json")


class UserManager():
    def __init__(self):
        global user_directory

        self.settings = AppSettings(self)
        if not os.path.exists(user_directory):
            os.mkdir(user_directory)
            if not args.multi_user:
                print("****** User settings have been changed to be stored on the server instead of browser storage. ******")
                print("****** For multi-user setups add the --multi-user CLI argument to enable multiple user profiles. ******")

        if args.multi_user:
            if os.path.isfile(users_file):
                with open(users_file) as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        else:
            self.users = {"default": "default"}

    def get_request_user_id(self, request):
        user = "default"
        if args.multi_user and "comfy-user" in request.headers:
            user = request.headers["comfy-user"]

        if user not in self.users:
            raise KeyError("Unknown user: " + user)

        return user

    def get_request_user_filepath(self, request, file, type="userdata", create_dir=True):
        global user_directory

        if type == "userdata":
            root_dir = user_directory
        else:
            raise KeyError("Unknown filepath type:" + type)

        user = self.get_request_user_id(request)
        path = user_root = os.path.abspath(os.path.join(root_dir, user))

        # prevent leaving /{type}
        if os.path.commonpath((root_dir, user_root)) != root_dir:
            return None

        if file is not None:
            # prevent leaving /{type}/{user}
            path = os.path.abspath(os.path.join(user_root, file))
            if os.path.commonpath((user_root, path)) != user_root:
                return None

        parent = os.path.split(path)[0]

        if create_dir and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        return path

    def add_user(self, name):
        name = name.strip()
        if not name:
            raise ValueError("username not provided")
        user_id = re.sub("[^a-zA-Z0-9-_]+", '-', name)
        user_id = user_id + "_" + str(uuid.uuid4())

        self.users[user_id] = name

        global users_file
        with open(users_file, "w") as f:
            json.dump(self.users, f)

        return user_id

    def add_routes(self, routes):
        self.settings.add_routes(routes)

        @routes.get("/users")
        async def get_users(request):
            if args.multi_user:
                return web.json_response({"storage": "server", "users": self.users})
            else:
                user_dir = self.get_request_user_filepath(request, None, create_dir=False)
                return web.json_response({
                    "storage": "server",
                    "migrated": os.path.exists(user_dir)
                })

        @routes.post("/users")
        async def post_users(request):
            body = await request.json()
            username = body["username"]
            if username in self.users.values():
                return web.json_response({"error": "Duplicate username."}, status=400)

            user_id = self.add_user(username)
            return web.json_response(user_id)

        @routes.get("/userdata")
        async def listuserdata(request):
            directory = request.rel_url.query.get('dir', '')
            if not directory:
                return web.Response(status=400)
                
            path = self.get_request_user_filepath(request, directory)
            if not path:
                return web.Response(status=403)
            
            if not os.path.exists(path):
                return web.Response(status=404)
            
            recurse = request.rel_url.query.get('recurse', '').lower() == "true"
            results = glob.glob(os.path.join(
                glob.escape(path), '**/*'), recursive=recurse)
            results = [os.path.relpath(x, path) for x in results if os.path.isfile(x)]
            
            split_path = request.rel_url.query.get('split', '').lower() == "true"
            if split_path:
                results = [[x] + x.split(os.sep) for x in results]

            return web.json_response(results)

        def get_user_data_path(request, check_exists = False, param = "file"):
            file = request.match_info.get(param, None)
            if not file:
                return web.Response(status=400)
                
            path = self.get_request_user_filepath(request, file)
            if not path:
                return web.Response(status=403)
            
            if check_exists and not os.path.exists(path):
                return web.Response(status=404)
            
            return path

        @routes.get("/userdata/{file}")
        async def getuserdata(request):
            path = get_user_data_path(request, check_exists=True)
            if not isinstance(path, str):
                return path
            
            return web.FileResponse(path)

        @routes.post("/userdata/{file}")
        async def post_userdata(request):
            path = get_user_data_path(request)
            if not isinstance(path, str):
                return path
            
            overwrite = request.query["overwrite"] != "false"
            if not overwrite and os.path.exists(path):
                return web.Response(status=409)

            body = await request.read()

            with open(path, "wb") as f:
                f.write(body)
                
            resp = os.path.relpath(path, self.get_request_user_filepath(request, None))
            return web.json_response(resp)

        @routes.delete("/userdata/{file}")
        async def delete_userdata(request):
            path = get_user_data_path(request, check_exists=True)
            if not isinstance(path, str):
                return path

            os.remove(path)
                
            return web.Response(status=204)

        @routes.post("/userdata/{file}/move/{dest}")
        async def move_userdata(request):
            source = get_user_data_path(request, check_exists=True)
            if not isinstance(source, str):
                return source
            
            dest = get_user_data_path(request, check_exists=False, param="dest")
            if not isinstance(source, str):
                return dest
            
            overwrite = request.query["overwrite"] != "false"
            if not overwrite and os.path.exists(dest):
                return web.Response(status=409)

            print(f"moving '{source}' -> '{dest}'")
            shutil.move(source, dest)
                
            resp = os.path.relpath(dest, self.get_request_user_filepath(request, None))
            return web.json_response(resp)

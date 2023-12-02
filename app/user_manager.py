import json
import os
import re
import uuid
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

        if args.multi_user:
            if os.path.isfile(users_file):
                with open(users_file) as f:
                    self.users = json.load(f)
            else:
                self.users = {}
        else:
            self.users = {"default": "default"}

    def get_request_user_id(self, request, default_user="default"):
        user = default_user
        if args.multi_user and "comfy-user" in request.headers:
            user = request.headers["comfy-user"]

        if user not in self.users:
            raise KeyError("Unknown user: " + user)

        return user

    def get_request_user_filepath(self, request, file, type="userdata", default_user="default"):
        global user_directory

        if type == "userdata":
            root_dir = user_directory
        else:
            raise KeyError("Unknown filepath type:" + type)

        user = self.get_request_user_id(request, default_user)
        user_root = os.path.abspath(os.path.join(root_dir, user))

        # prevent leaving /{type}
        if os.path.commonpath((root_dir, user_root)) != root_dir:
            return None

        # prevent leaving /{type}/{user}
        path = os.path.abspath(os.path.join(user_root, file))
        if os.path.commonpath((user_root, path)) != user_root:
            return None

        if not os.path.exists(user_root):
            os.mkdir(user_root)

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
                return web.json_response(self.users)
            else:
                return web.json_response(None)

        @routes.post("/users")
        async def post_users(request):
            body = await request.json()
            user_id = self.add_user(body["username"])
            return web.json_response(user_id)

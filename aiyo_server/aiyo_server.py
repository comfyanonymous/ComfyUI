import os
import sys


import nodes

import urllib

import server

from config.config import CONFIG
from framework.app_log import AppLog
from framework.model import object_storage
from framework.model import tb_data
from aiyo_server.server_client_communicate import ServerClientCommunicator
from aiyo_server import server_task_queue

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    AppLog.info("Module 'aiohttp' not installed. Please install it via:")
    AppLog.info("pip install aiohttp")
    AppLog.info("or")
    AppLog.info("pip install -r requirements.txt")
    sys.exit()

import mimetypes
from comfy.cli_args import args



@web.middleware
async def cache_control(request: web.Request, handler):
    response: web.Response = await handler(request)
    if request.path.endswith('.js') or request.path.endswith('.css'):
        response.headers.setdefault('Cache-Control', 'no-cache')
    return response

def create_cors_middleware(allowed_origin: str):
    @web.middleware
    async def cors_middleware(request: web.Request, handler):
        if request.method == "OPTIONS":
            # Pre-flight request. Reply successfully:
            response = web.Response()
        else:
            response = await handler(request)

        response.headers['Access-Control-Allow-Origin'] = allowed_origin
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    return cors_middleware




class AIYoServer():
    def __init__(self, loop):
        server.PromptServer.instance = self         # hack code for extension supports
        AIYoServer.instance = self

        mimetypes.init()
        mimetypes.types_map['.js'] = 'application/javascript; charset=utf-8'
        
        # init task queue, resource manager
        que = None
        if CONFIG["deploy"]:
            que = server_task_queue.TaskQueueKafka(self)
            que.init_producer()
            tb_data.default_connect()
            
            self.resource_mgr = object_storage.ResourceMgrRemote()
        else:
            self.resource_mgr = object_storage.ResourceMgrLocal()
            que = server_task_queue.TaskQueueLocal(self)
        self.prompt_queue = que

        self.supports = ["custom_nodes_from_web"]
        self.loop = loop
        self.number = 0

        middlewares = [cache_control]
        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app = web.Application(client_max_size=max_upload_size, middlewares=middlewares)
        self.sockets = dict()
        self.web_root = os.path.join(os.getcwd(), "web")
        routes = web.RouteTableDef()
        self.routes = routes
        self.last_node_id = None
        self.client_id = None

        self.on_prompt_handlers = []
        
        # server client communicator
        self.server_client_communicator = ServerClientCommunicator(self)

        
        import aiyo_server.editor_route
        import aiyo_server.open_api_route
            
        
    def add_routes(self):
        self.app.add_routes(self.routes)

        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([
                web.static('/extensions/' + urllib.parse.quote(name), dir, follow_symlinks=True),
            ])

        self.app.add_routes([
            web.static('/', self.web_root, follow_symlinks=True),
        ])
        

    async def publish_loop(self):
        while True:
            await self.server_client_communicator.process_one()


    async def start(self, address, port, verbose=True, call_on_start=None):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, address, port)
        await site.start()

        if address == '':
            address = '0.0.0.0'
        if verbose:
            AppLog.info("Starting server\n")
            AppLog.info("To see the GUI go to: http://{}:{}".format(address, port))
        if call_on_start is not None:
            call_on_start(address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception as e:
                AppLog.info(f"[ERROR] An error occurred during the on_prompt_handler processing")

        return json_data

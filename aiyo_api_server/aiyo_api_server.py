import os
import sys


from framework.app_log import AppLog
from framework.model import object_storage
from framework.model import tb_data
from aiyo_api_server import server_task_queue


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




class AIYoApiServer():
    def __init__(self):
        # server.PromptServer.instance = self         # hack code for extension supports
        AIYoApiServer.instance = self

        mimetypes.init()
        mimetypes.types_map['.js'] = 'application/javascript; charset=utf-8'
        
        # init task queue, resource manager
        que = server_task_queue.TaskQueueKafka(self)
        que.init_producer()
        tb_data.default_connect()
        
        self.resource_mgr = object_storage.ResourceMgrRemote()

        self.prompt_queue = que
        self.supports = ["custom_nodes_from_web"]
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
        self.server_client_communicator = None #ServerClientCommunicator(self)
        
        import aiyo_api_server.open_api_route
        


    def start(self, address, port, verbose=True, call_on_start=None):
        # runner = web.AppRunner(self.app, access_log=None)
        # await runner.setup()
        # site = web.TCPSite(runner, address, port)
        # await site.start()

        if address == '':
            address = '0.0.0.0'
        AppLog.info("Starting server\n")
        AppLog.info("To see the GUI go to: http://{}:{}".format(address, port))
        if call_on_start is not None:
            call_on_start(address, port)
        self.app.add_routes(self.routes)
        web.run_app(self.app, host=address, port=port)

    # async def publish_loop(self):
    #     import time
    #     while True:
    #         #await self.server_client_communicator.process_one()
    #         time.sleep(1)            
    
    

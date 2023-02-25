import os
import sys
import asyncio
import nodes
import main
import uuid
import json

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("Module 'aiohttp' not installed. Please install it via:")
    print("pip install aiohttp")
    print("or")
    print("pip install -r requirements.txt")
    sys.exit()

class PromptServer():
    def __init__(self, loop):
        self.prompt_queue = None
        self.loop = loop
        self.messages = asyncio.Queue()
        self.number = 0
        self.app = web.Application()
        self.sockets = dict()
        self.web_root = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "webshit")
        routes = web.RouteTableDef()

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = uuid.uuid4().hex
            self.sockets[sid] = ws
            try:
                # Send initial state to the new client
                await self.send("status", { "status": self.get_queue_info(), 'sid': sid }, sid)
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        print('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid)
            return ws

        @routes.get("/")
        async def get_root(request):
            return web.FileResponse(os.path.join(self.web_root, "index.html"))
        
        @routes.get("/view/{file}")
        async def view_image(request):
            if "file" in request.match_info:
                output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
                file = request.match_info["file"]
                file = os.path.splitext(os.path.basename(file))[0] + ".png"
                file = os.path.join(output_dir, file)
                if os.path.isfile(file):
                    return web.FileResponse(file)
                
            return web.Response(status=404)
        
        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())
        
        @routes.get("/object_info")
        async def get_object_info(request):
            out = {}
            for x in nodes.NODE_CLASS_MAPPINGS:
                obj_class = nodes.NODE_CLASS_MAPPINGS[x]
                info = {}
                info['input'] = obj_class.INPUT_TYPES()
                info['output'] = obj_class.RETURN_TYPES
                info['name'] = x #TODO
                info['description'] = ''
                info['category'] = 'sd'
                if hasattr(obj_class, 'CATEGORY'):
                    info['category'] = obj_class.CATEGORY
                out[x] = info
            return web.json_response(out)
        
        @routes.get("/history")
        async def get_history(request):
            return web.json_response(self.prompt_queue.history)
        
        @routes.get("/queue")
        async def get_queue(request):
            queue_info = {}
            current_queue = self.prompt_queue.get_current_queue()
            queue_info['queue_running'] = current_queue[0]
            queue_info['queue_pending'] = current_queue[1]
            return web.json_response(queue_info)
        
        @routes.post("/prompt")
        async def post_prompt(request):
            print("got prompt")
            resp_code = 200
            out_string = ""
            json_data =  await request.json()

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                valid = main.validate_prompt(prompt)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if valid[0]:
                    self.prompt_queue.put((number, id(prompt), prompt, extra_data))
                else:
                    resp_code = 400
                    out_string = valid[1]
                    print("invalid prompt:", valid[1])

            return web.Response(body=out_string, status=resp_code)
        
        @routes.post("/queue")
        async def post_queue(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == int(id_to_delete)
                    self.prompt_queue.delete_queue_item(delete_func)
                    
            return web.Response(status=200)
        
        @routes.post("/history")
        async def post_history(request):
            json_data =  await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.history = {}
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.history.pop(id_to_delete, None)
                    
            return web.Response(status=200)
    
        self.app.add_routes(routes)
        self.app.add_routes([
            web.static('/', self.web_root),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        message = {"type": event, "data": data}
       
        if isinstance(message, str) == False:
            message = json.dumps(message)

        if sid is None:
            for ws in self.sockets.values():
                await ws.send_str(message)
        elif sid in self.sockets:
            await self.sockets[sid].send_str(message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))
    
    def queue_updated(self):
        self.send_sync("status", { "status": self.get_queue_info() })

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, address, port)
        await site.start()
    
        if address == '':
            address = '0.0.0.0'
        print("Starting server\n")
        print("To see the GUI go to: http://{}:{}".format(address, port))   
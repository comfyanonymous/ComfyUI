from ai_api import (queue_prompt,
                    get_history,
                    get_queue_state,
                    get_parsed_input_nodes,
                    parse_workflow_prompt,
                    parse_outputs,
                    save_binary_file)
import asyncio
import aiohttp
import logging
from aiohttp import web
import json, os

class BridgeServer():
    
    def __init__(self, loop, server_address:list) -> None:
        self.sockets_res = {}
        self.sockets_req = {}
        self.sid_server_map = {}
        self.ws_connection_status = {}
        self.loop = loop
        self.server_address = server_address
        
    async def send_socket_catch_exception(self, sid, message):
        try:
            await self.sockets_res[sid].send_json(message)
            self.ws_connection_status[sid] = message.get("status", None)
        except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
            logging.warning(f"send error: {err} / {message}")

    async def init_app(self):
        app = web.Application()
        app.add_routes([
            web.post('/generate-based-workflow', self.generate_based_workflow),
            web.get('/ws', self.websocket_connection),
            web.get("/", self.main_page),
            web.get("/history", self.get_history),
            web.post("/upload/image", self.upload_image),
            web.post("/workflow-info", self.workflow_info)
        ])
        return app
    
    async def track_progress(self, sid):
        ws_req = self.sockets_req[sid]
        finished_nodes = []
        
        while True:
            out = await ws_req.receive()
            out = out.data
            
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'progress':
                    data = message['data']
                    current_step = data['value']
                    progress_message = {
                        'status': 'progress_update',
                        'details': f'In K-Sampler -> Step: {current_step} of {data["max"]}'
                    }
                    await self.send_socket_catch_exception(sid, progress_message)

                if message['type'] == 'execution_cached':
                    data = message['data']
                    for itm in data['nodes']:
                        if itm not in finished_nodes:
                            finished_nodes.append(itm)
                            progress_message = {
                                'status': 'progress_update',
                                'details': f'Progress: {len(finished_nodes)} / {len(data['nodes'])} Tasks done'
                            }
                            await  self.send_socket_catch_exception(sid, progress_message)

                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] not in finished_nodes:
                        finished_nodes.append(data['node'])
                        progress_message = {
                            'status': 'progress_update',
                            'details': f'Progress: {len(finished_nodes)} / ) Tasks done'
                        }
                        await  self.send_socket_catch_exception(sid, progress_message)

                    if data['node'] is None and data['prompt_id'] == sid:
                        completion_message = {
                            'status': 'closed',
                            'details': 'Execution is done'
                        }
                        await  self.send_socket_catch_exception(sid, completion_message)
                        break  # Execution is done
            elif out == None:
                completion_message = {
                    'status': 'closed',
                    'details': 'comfy ui out is None'
                }
                await  self.send_socket_catch_exception(sid, completion_message)
                break
            else:
                continue
        return


    async def websocket_connection(self, request):
        sid = request.rel_url.query.get('clientId', '')

        async with aiohttp.ClientSession() as session:
            server_address = await self.get_not_busy_server_address()
            self.sid_server_map[sid] = server_address

            ws_res = web.WebSocketResponse()
            ws_req = await session.ws_connect(f"ws://{server_address}/ws?clientId={sid}")

            self.sockets_res[sid] = ws_res
            self.sockets_req[sid] = ws_req

            try:
                await self.sockets_res[sid].prepare(request)

                await self.send_socket_catch_exception(sid, {"status":"connected", "details":"web socket connected"})
                self.ws_connection_status[sid] = "connected"
                
                task = asyncio.create_task(self.track_progress(sid))

                while True:
                    out = await ws_res.receive()
                    out = out.data

                    if isinstance(out, str):
                        message = json.loads(out)
                        if message['status'] == 'error':
                            break
                    if self.ws_connection_status[sid] == "closed":
                        break
                
                await task
                
            except aiohttp.ServerDisconnectedError as e:
                await self.send_socket_catch_exception(sid, {"status":"error", "details":"server disconnected"})
            
            except Exception as e:
                await self.send_socket_catch_exception(sid, {"status":"error", "details":str(e)})

            finally:
                await self.send_socket_catch_exception(sid, {"status":"closed", "details":"connection will be closed"})
                await self.sockets_req[sid].close()
                await self.sockets_res[sid].close()
                self.sockets_req.pop(sid, None)
                self.sockets_res.pop(sid, None)
                self.ws_connection_status.pop(sid, None)

    async def get_not_busy_server_address(self):
        queue_lenghs = []
        for server_address in self.server_address:
            try:
                queue_state = get_queue_state(server_address)
            except Exception as e:
                print(f"{server_address} in error {e}")
                
            queue_length = sum([len(cur) for cur in queue_state.values()])
            queue_lenghs.append(queue_length)
        
        target_server_address = self.server_address[queue_lenghs.index(min(queue_lenghs))]

        return target_server_address
    
    async def generate_based_workflow(self, request):
        data = await request.json()
        sid = request.rel_url.query.get('clientId', '')

        workflow = data.pop("workflow", None)
        kwargs = data
    
        prompt = parse_workflow_prompt(workflow, **kwargs)
        prompt = queue_prompt(prompt, sid, self.sid_server_map[sid])

        return web.Response(status=200)
    
    async def upload_image(self, request):
        reader = await request.multipart()

        fns={}
        async for part in reader:
            file_name = part.headers.get('Content-Disposition', '').split('filename=')[1].strip('"')
            file_data = await part.read()
        
            fn = save_binary_file(file_data, file_name)
            fns[part.headers.get("ori_file_id", None)] = fn

        return web.Response(
            status=200,
            body=json.dumps(fns),
            headers={"Content-Type": "application/json"}
        )
        
    async def get_history(self, request):
        sid = request.rel_url.query.get('clientId', '')

        history = get_history(sid, self.sid_server_map[sid])
        history = history.get(sid, None)
        self.sid_server_map.pop(sid, None)

        if history is not None:
            output = history["outputs"],
            if isinstance(output, tuple):
                output = (output)
            
            writer = aiohttp.MultipartWriter("form-data")
            
            file_paths, mime_types, file_contents = parse_outputs(output[0])
            for file_path, mime_type, file_content in zip(file_paths, mime_types, file_contents):

                headers = {'Content-Type': mime_type, 'Content-Disposition': f'attachment; filename="{file_path.split("/")[-1]}"'}
                writer.append(file_content, headers)

            headers = {
                'Content-Type': writer.content_type,
            }
            
            return web.Response(
                status=200,
                body=writer,
                headers=headers
            )
        else:
            return web.Response(
                status=204,
                text="No contents with that client id."
            )
    
    async def workflow_info(self, request):
        data = await request.json()
        
        workflow = data.pop("workflow", None)
        node_info = get_parsed_input_nodes(workflow)

        return web.Response(status=200, body=json.dumps(node_info), content_type="application/json")

    async def main_page(self, request):
        return web.Response(text="Hello, this is Favorfit Bridge Server!")

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    servers_str = os.getenv('COMFYUI_SERVERS')
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    server_list = servers_str.split(',') if servers_str else []

    loop = asyncio.get_event_loop()
    server = BridgeServer(loop, server_list)
    app = loop.run_until_complete(server.init_app())
    web.run_app(app, host=host, port=int(port))
    

if __name__ == '__main__':
    main()

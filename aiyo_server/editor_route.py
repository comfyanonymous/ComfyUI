import uuid, os, sys, json
import traceback
from io import BytesIO
import glob
import urllib

import aiohttp
from aiohttp import web
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo


import nodes
import folder_paths
from framework.app_log import AppLog
from aiyo_server.aiyo_server import AIYoServer
from framework.resource_loader import Resourceloader

from comfy.cli_args import args
import comfy.utils
import comfy.model_management

@AIYoServer.instance.routes.post("/editor/{flow_id}/save_workflow")
async def save_workflow(request):
    post = await request.post()
    workflow = post["workflow"]
    prompt = post["prompt"]




@AIYoServer.instance.routes.get('/ws')
async def websocket_handler(request):
    """
    Accept websocket requests
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    sid = request.rel_url.query.get('clientId', '')
    if sid:
        # Reusing existing session, remove old
        AIYoServer.instance.sockets.pop(sid, None)
    else:
        sid = uuid.uuid4().hex

    AIYoServer.instance.sockets[sid] = ws

    try:
        # Send initial state to the new client
        await AIYoServer.instance.server_client_communicator.send("status", { "status": AIYoServer.instance.prompt_queue.get_queue_info(), 'sid': sid }, sid)
        # On reconnect if we are the currently executing client send the current node
        if AIYoServer.instance.client_id == sid and AIYoServer.instance.last_node_id is not None:
            await AIYoServer.instance.server_client_communicator.send("executing", { "node": AIYoServer.instance.last_node_id }, sid)
            
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.ERROR:
                AppLog.info('ws connection closed with exception %s' % ws.exception())
    finally:
        AIYoServer.instance.sockets.pop(sid, None)
    return ws

@AIYoServer.instance.routes.get("/")
async def get_root(request):
    """
    Get root web page.
    """
    return web.FileResponse(os.path.join(AIYoServer.instance.web_root, "index.html"))

@AIYoServer.instance.routes.get("/embeddings")
def get_embeddings(request):
    """
    Get all embeddings
    RETURN:
        list of embedding file names.
    """
    embeddings = folder_paths.get_filename_list("embeddings")
    return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

@AIYoServer.instance.routes.get("/extensions")
async def get_extensions(request):
    """
    Get get all extensions .js scripts.
    """
    files = glob.glob(os.path.join(
        glob.escape(AIYoServer.instance.web_root), 'extensions/**/*.js'), recursive=True)
    
    extensions = list(map(lambda f: "/" + os.path.relpath(f, AIYoServer.instance.web_root).replace("\\", "/"), files))
    
    for name, dir in nodes.EXTENSION_WEB_DIRS.items():
        files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
        extensions.extend(list(map(lambda f: "/extensions/" + urllib.parse.quote(
            name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

    return web.json_response(extensions)


@AIYoServer.instance.routes.post("/upload/image")
async def upload_image(request):
    post = await request.post()
    succ, data = Resourceloader.image_upload(post)
    if succ:
        return web.json_response(data)
    else:
        return web.Response(status=400)

@AIYoServer.instance.routes.post("/upload/mask")
async def upload_mask(request):
    post = await request.post()

    succ, data = Resourceloader.mask_upload(post)
    if succ:
        return web.json_response(data)
    else:
        return web.Response(status=400)

@AIYoServer.instance.routes.get("/view")
async def view_image(request):
    if "filename" in request.rel_url.query:
        filename = request.rel_url.query["filename"]
        filename,output_dir = folder_paths.annotated_filepath(filename)

        # validation for security: prevent accessing arbitrary path
        if filename[0] == '/' or '..' in filename:
            return web.Response(status=400)

        if output_dir is None:
            type = request.rel_url.query.get("type", "output")
            output_dir = folder_paths.get_directory_by_type(type)

        if output_dir is None:
            return web.Response(status=400)

        if "subfolder" in request.rel_url.query:
            full_output_dir = os.path.join(output_dir, request.rel_url.query["subfolder"])
            if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                return web.Response(status=403)
            output_dir = full_output_dir

        filename = os.path.basename(filename)
        file = os.path.join(output_dir, filename)

        if os.path.isfile(file):
            if 'preview' in request.rel_url.query:
                with Image.open(file) as img:
                    preview_info = request.rel_url.query['preview'].split(';')
                    image_format = preview_info[0]
                    if image_format not in ['webp', 'jpeg'] or 'a' in request.rel_url.query.get('channel', ''):
                        image_format = 'webp'

                    quality = 90
                    if preview_info[-1].isdigit():
                        quality = int(preview_info[-1])

                    buffer = BytesIO()
                    if image_format in ['jpeg'] or request.rel_url.query.get('channel', '') == 'rgb':
                        img = img.convert("RGB")
                    img.save(buffer, format=image_format, quality=quality)
                    buffer.seek(0)

                    return web.Response(body=buffer.read(), content_type=f'image/{image_format}',
                                        headers={"Content-Disposition": f"filename=\"{filename}\""})

            if 'channel' not in request.rel_url.query:
                channel = 'rgba'
            else:
                channel = request.rel_url.query["channel"]

            if channel == 'rgb':
                with Image.open(file) as img:
                    if img.mode == "RGBA":
                        r, g, b, a = img.split()
                        new_img = Image.merge('RGB', (r, g, b))
                    else:
                        new_img = img.convert("RGB")

                    buffer = BytesIO()
                    new_img.save(buffer, format='PNG')
                    buffer.seek(0)

                    return web.Response(body=buffer.read(), content_type='image/png',
                                        headers={"Content-Disposition": f"filename=\"{filename}\""})

            elif channel == 'a':
                with Image.open(file) as img:
                    if img.mode == "RGBA":
                        _, _, _, a = img.split()
                    else:
                        a = Image.new('L', img.size, 255)

                    # alpha img
                    alpha_img = Image.new('RGBA', img.size)
                    alpha_img.putalpha(a)
                    alpha_buffer = BytesIO()
                    alpha_img.save(alpha_buffer, format='PNG')
                    alpha_buffer.seek(0)

                    return web.Response(body=alpha_buffer.read(), content_type='image/png',
                                        headers={"Content-Disposition": f"filename=\"{filename}\""})
            else:
                return web.FileResponse(file, headers={"Content-Disposition": f"filename=\"{filename}\""})

    return web.Response(status=404)

@AIYoServer.instance.routes.get("/view_metadata/{folder_name}")
async def view_metadata(request):
    AppLog.info(f"/view_metadata/{folder_name}")
    folder_name = request.match_info.get("folder_name", None)
    if folder_name is None:
        return web.Response(status=404)
    if not "filename" in request.rel_url.query:
        return web.Response(status=404)

    filename = request.rel_url.query["filename"]
    if not filename.endswith(".safetensors"):
        return web.Response(status=404)

    safetensors_path = folder_paths.get_full_path(folder_name, filename)
    if safetensors_path is None:
        return web.Response(status=404)
    out = comfy.utils.safetensors_header(safetensors_path, max_size=1024*1024)
    if out is None:
        return web.Response(status=404)
    dt = json.loads(out)
    if not "__metadata__" in dt:
        return web.Response(status=404)
    return web.json_response(dt["__metadata__"])

@AIYoServer.instance.routes.get("/system_stats")
async def get_queue(request):
    device = comfy.model_management.get_torch_device()
    device_name = comfy.model_management.get_torch_device_name(device)
    vram_total, torch_vram_total = comfy.model_management.get_total_memory(device, torch_total_too=True)
    vram_free, torch_vram_free = comfy.model_management.get_free_memory(device, torch_free_too=True)
    system_stats = {
        "system": {
            "os": os.name,
            "python_version": sys.version,
            "embedded_python": os.path.split(os.path.split(sys.executable)[0])[1] == "python_embeded"
        },
        "devices": [
            {
                "name": device_name,
                "type": device.type,
                "index": device.index,
                "vram_total": vram_total,
                "vram_free": vram_free,
                "torch_vram_total": torch_vram_total,
                "torch_vram_free": torch_vram_free,
            }
        ]
    }
    return web.json_response(system_stats)

@AIYoServer.instance.routes.get("/prompt")
async def get_prompt(request):
    return web.json_response(AIYoServer.instance.get_queue_info())

def node_info(node_class):
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = {}
    info['input'] = obj_class.INPUT_TYPES()
    info['output'] = obj_class.RETURN_TYPES
    info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
    info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
    info['name'] = node_class
    info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
    info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
    info['category'] = 'sd'
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
        info['output_node'] = True
    else:
        info['output_node'] = False

    if hasattr(obj_class, 'CATEGORY'):
        info['category'] = obj_class.CATEGORY
        
    # flow inputs
    if hasattr(obj_class, 'FLOW_INPUTS'):
        info['flow_inputs'] = obj_class.FLOW_INPUTS
    else:
        info['flow_inputs'] = [("FROM", "FLOW")]            # by default, every node has one flow-input
    # flow outputs
    if hasattr(obj_class, 'FLOW_OUTPUTS'):
        info['flow_outputs'] = obj_class.FLOW_OUTPUTS
    else:
        info['flow_outputs'] = [("TO", "FLOW")]            # by default, every node has one flow-output
        
    return info

@AIYoServer.instance.routes.get("/object_info")
async def get_object_info(request):
    AppLog.info(f"get object_info")
    out = {}
    for x in nodes.NODE_CLASS_MAPPINGS:
        try:
            out[x] = node_info(x)
        except Exception as e:
            AppLog.info(f"[ERROR] An error occurred while retrieving information for the '{x}' node.", file=sys.stderr)
            AppLog.info(traceback.print_exc())
    return web.json_response(out)

@AIYoServer.instance.routes.get("/object_info/{node_class}")
async def get_object_info_node(request):
    AppLog.info(f"get node info: {node_class}")
    
    node_class = request.match_info.get("node_class", None)
    out = {}
    if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
        out[node_class] = node_info(node_class)
    return web.json_response(out)

@AIYoServer.instance.routes.get("/history")
async def get_history(request):
    max_items = request.rel_url.query.get("max_items", None)
    if max_items is not None:
        max_items = int(max_items)
    return web.json_response(AIYoServer.instance.prompt_queue.get_history(max_items=max_items))

@AIYoServer.instance.routes.get("/history/{prompt_id}")
async def get_history(request):
    prompt_id = request.match_info.get("prompt_id", None)
    return web.json_response(AIYoServer.instance.prompt_queue.get_history(prompt_id=prompt_id))

@AIYoServer.instance.routes.get("/queue")
async def get_queue(request):
    queue_info = {}
    current_queue = AIYoServer.instance.prompt_queue.get_current_queue()
    queue_info['queue_running'] = current_queue[0]
    queue_info['queue_pending'] = current_queue[1]
    AppLog.info(f"Get Queue: {queue_info}")
    return web.json_response(queue_info)

@AIYoServer.instance.routes.post("/prompt")
async def post_prompt(request):
    AppLog.info("got prompt")
    resp_code = 200
    out_string = ""
    json_data =  await request.json()
    json_data = AIYoServer.instance.trigger_on_prompt(json_data)
    
    succ, data = AIYoServer.instance.prompt_queue.put(json_data)
    if succ:
        return web.json_response(data)
    else:
        return web.json_response(data, status=400)


@AIYoServer.instance.routes.post("/queue")
async def post_queue(request):
    json_data =  await request.json()
    if "clear" in json_data:
        if json_data["clear"]:
            AIYoServer.instance.prompt_queue.wipe_queue()
    if "delete" in json_data:
        to_delete = json_data['delete']
        for id_to_delete in to_delete:
            delete_func = lambda a: a[1] == id_to_delete
            AIYoServer.instance.prompt_queue.delete_queue_item(delete_func)

    return web.Response(status=200)

@AIYoServer.instance.routes.post("/interrupt")
async def post_interrupt(request):
    nodes.interrupt_processing()
    return web.Response(status=200)

@AIYoServer.instance.routes.post("/history")
async def post_history(request):
    json_data =  await request.json()
    if "clear" in json_data:
        if json_data["clear"]:
            AIYoServer.instance.prompt_queue.wipe_history()
    if "delete" in json_data:
        to_delete = json_data['delete']
        for id_to_delete in to_delete:
            AIYoServer.instance.prompt_queue.delete_history_item(id_to_delete)

    return web.Response(status=200)


@AIYoServer.instance.routes.post("/task_exe/send_task_msg")
async def send_task_msg(request):
    """
    INPUTS:
    event_type: event type,
    data: message data
    sid: client id
    """
    json_data = await request.json()
    AIYoServer.instance.server_client_communicator.send_sync(json_data["event_type"], json_data["data"], json_data["sid"])
    return web.Response(status=200)
    
@AIYoServer.instance.routes.get("/task_exe/get_task")
async def get_task(request):
    next_task = AIYoServer.instance.prompt_queue.get()
    if next_task is not None:
        next_task = next_task[0]
    else:
        return web.json_response({
            "prompt_id": None,
            "prompt": None,
            "extra_data": None,
            "outputs_to_execute": None,
            "flows": None
        })
    
    return web.json_response({
        "prompt_id": next_task[1],
        "prompt": next_task[2],
        "extra_data": next_task[3],
        "outputs_to_execute": next_task[4],
        "flows": next_task[5]
    })
    
    
@AIYoServer.instance.routes.post("/task_exe/task_done")
async def task_done(request):
    json_data = await request.json()
    prompt_id = json_data["prompt_id"]
    output_data = json_data["output"]
    AIYoServer.instance.prompt_queue.task_done(prompt_id, output_data)
    return web.Response(status=200)
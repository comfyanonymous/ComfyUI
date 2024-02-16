from __future__ import annotations
import asyncio
import traceback
import glob
import struct
import sys
import shutil
from urllib.parse import quote, urljoin
from pkg_resources import resource_filename

from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from io import BytesIO

import json
import os
import uuid
from asyncio import Future, AbstractEventLoop
from typing import List, Optional

import aiofiles
import aiohttp
from aiohttp import web

from ..component_model.queue_types import QueueItem, HistoryEntry, BinaryEventTypes, TaskInvocation
from ..cmd import execution
from ..cmd import folder_paths
import mimetypes

from ..digest import digest
from ..cli_args import args
from .. import utils
from .. import model_management
from ..component_model.executor_types import ExecutorToClientProgress
from ..component_model.file_output_path import file_output_path
from ..nodes.package import import_all_nodes_in_workspace
from ..vendor.appdirs import user_data_dir

nodes = import_all_nodes_in_workspace()

from ..app.user_manager import UserManager


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
        print("send error:", err)


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


class PromptServer(ExecutorToClientProgress):
    instance: 'PromptServer'

    def __init__(self, loop):
        PromptServer.instance = self

        mimetypes.init()
        mimetypes.types_map['.js'] = 'application/javascript; charset=utf-8'

        self.address: str = "0.0.0.0"
        self.user_manager = UserManager()
        # todo: this is probably read by custom nodes elsewhere
        self.supports: List[str] = ["custom_nodes_from_web"]
        self.prompt_queue: execution.AbstractPromptQueue | None = None
        self.loop: AbstractEventLoop = loop
        self.messages: asyncio.Queue = asyncio.Queue()
        self.number: int = 0
        self.port: int = 8188
        self._external_address: Optional[str] = None

        middlewares = [cache_control]
        if args.enable_cors_header:
            middlewares.append(create_cors_middleware(args.enable_cors_header))

        max_upload_size = round(args.max_upload_size * 1024 * 1024)
        self.app: web.Application = web.Application(client_max_size=max_upload_size,
                                                    handler_args={'max_field_size': 16380},
                                                    middlewares=middlewares)
        self.sockets = dict()
        web_root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../web")
        if not os.path.exists(web_root_path):
            web_root_path = resource_filename('comfy', 'web/')
        self.web_root = web_root_path
        routes = web.RouteTableDef()
        self.routes: web.RouteTableDef = routes
        self.last_node_id = None
        self.last_prompt_id = None
        self.client_id = None

        self.on_prompt_handlers = []

        @routes.get('/ws')
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            sid = request.rel_url.query.get('clientId', '')
            if sid:
                # Reusing existing session, remove old
                self.sockets.pop(sid, None)
            else:
                sid = uuid.uuid4().hex

            self.sockets[sid] = ws

            try:
                # Send initial state to the new client
                await self.send("status", {"status": self.get_queue_info(), 'sid': sid}, sid)
                # On reconnect if we are the currently executing client send the current node
                if self.client_id == sid and self.last_node_id is not None:
                    await self.send("executing", {"node": self.last_node_id}, sid)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.ERROR:
                        print('ws connection closed with exception %s' % ws.exception())
            finally:
                self.sockets.pop(sid, None)
            return ws

        @routes.get("/")
        async def get_root(request):
            return web.FileResponse(os.path.join(self.web_root, "index.html"))

        @routes.get("/embeddings")
        def get_embeddings(self):
            embeddings = folder_paths.get_filename_list("embeddings")
            return web.json_response(list(map(lambda a: os.path.splitext(a)[0], embeddings)))

        @routes.get("/extensions")
        async def get_extensions(request):
            files = glob.glob(os.path.join(
                glob.escape(self.web_root), 'extensions/**/*.js'), recursive=True)
            extensions = list(map(lambda f: "/" + os.path.relpath(f, self.web_root).replace("\\", "/"), files))

            for name, dir in nodes.EXTENSION_WEB_DIRS.items():
                files = glob.glob(os.path.join(glob.escape(dir), '**/*.js'), recursive=True)
                extensions.extend(list(map(lambda f: "/extensions/" + quote(
                    name) + "/" + os.path.relpath(f, dir).replace("\\", "/"), files)))

            return web.json_response(extensions)

        def get_dir_by_type(dir_type=None):
            type_dir = ""
            if dir_type is None:
                dir_type = "input"

            if dir_type == "input":
                type_dir = folder_paths.get_input_directory()
            elif dir_type == "temp":
                type_dir = folder_paths.get_temp_directory()
            elif dir_type == "output":
                type_dir = folder_paths.get_output_directory()

            return type_dir, dir_type

        async def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")

            image_upload_type = post.get("type")
            upload_dir, image_upload_type = get_dir_by_type(image_upload_type)

            if image and image.file:
                filename = image.filename
                if not filename:
                    return web.Response(status=400)

                subfolder = post.get("subfolder", "")
                full_output_folder = os.path.join(upload_dir, os.path.normpath(subfolder))
                filepath = os.path.abspath(os.path.join(full_output_folder, filename))

                if os.path.commonpath((upload_dir, filepath)) != upload_dir:
                    return web.Response(status=400)

                if not os.path.exists(full_output_folder):
                    os.makedirs(full_output_folder)

                split = os.path.splitext(filename)

                if overwrite is not None and (overwrite == "true" or overwrite == "1"):
                    pass
                else:
                    i = 1
                    while os.path.exists(filepath):
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if image_save_function is not None:
                    image_save_function(image, post, filepath)
                else:
                    async with aiofiles.open(filepath, mode='wb') as file:
                        await file.write(image.file.read())

                return web.json_response({"name": filename, "subfolder": subfolder, "type": image_upload_type})
            else:
                return web.Response(status=400)

        @routes.post("/upload/image")
        async def upload_image(request):
            post = await request.post()
            return await image_upload(post)

        @routes.post("/upload/mask")
        async def upload_mask(request):
            post = await request.post()

            def image_save_function(image, post, filepath):
                original_ref = json.loads(post.get("original_ref"))
                filename, output_dir = folder_paths.annotated_filepath(original_ref['filename'])

                # validation for security: prevent accessing arbitrary path
                if filename[0] == '/' or '..' in filename:
                    return web.Response(status=400)

                if output_dir is None:
                    type = original_ref.get("type", "output")
                    output_dir = folder_paths.get_directory_by_type(type)

                if output_dir is None:
                    return web.Response(status=400)

                if original_ref.get("subfolder", "") != "":
                    full_output_dir = os.path.join(output_dir, original_ref["subfolder"])
                    if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
                        return web.Response(status=403)
                    output_dir = full_output_dir

                file = os.path.join(output_dir, filename)

                if os.path.isfile(file):
                    with Image.open(file) as original_pil:
                        metadata = PngInfo()
                        if hasattr(original_pil, 'text'):
                            for key in original_pil.text:
                                metadata.add_text(key, original_pil.text[key])
                        original_pil = original_pil.convert('RGBA')
                        mask_pil = Image.open(image.file).convert('RGBA')

                        # alpha copy
                        new_alpha = mask_pil.getchannel('A')
                        original_pil.putalpha(new_alpha)
                        original_pil.save(filepath, compress_level=4, pnginfo=metadata)

            return image_upload(post, image_save_function)

        @routes.get("/view")
        async def view_image(request):
            if "filename" in request.rel_url.query:
                filename = request.rel_url.query["filename"]
                type = request.rel_url.query.get("type", "output")
                subfolder = request.rel_url.query["subfolder"] if "subfolder" in request.rel_url.query else None

                try:
                    file = file_output_path(filename, type=type, subfolder=subfolder)
                except PermissionError:
                    return web.Response(status=403)
                except ValueError:
                    return web.Response(status=400)

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

        @routes.get("/view_metadata/{folder_name}")
        async def view_metadata(request):
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
            out = utils.safetensors_header(safetensors_path, max_size=1024 * 1024)
            if out is None:
                return web.Response(status=404)
            dt = json.loads(out)
            if not "__metadata__" in dt:
                return web.Response(status=404)
            return web.json_response(dt["__metadata__"])

        @routes.get("/system_stats")
        async def get_queue(request):
            device = model_management.get_torch_device()
            device_name = model_management.get_torch_device_name(device)
            vram_total, torch_vram_total = model_management.get_total_memory(device, torch_total_too=True)
            vram_free, torch_vram_free = model_management.get_free_memory(device, torch_free_too=True)
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

        @routes.get("/prompt")
        async def get_prompt(request):
            return web.json_response(self.get_queue_info())

        def node_info(node_class):
            obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [
                                                                                                               False] * len(
                obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = nodes.NODE_DISPLAY_NAME_MAPPINGS[
                node_class] if node_class in nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
            info['description'] = obj_class.DESCRIPTION if hasattr(obj_class, 'DESCRIPTION') else ''
            info['category'] = 'sd'
            if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
                info['output_node'] = True
            else:
                info['output_node'] = False

            if hasattr(obj_class, 'CATEGORY'):
                info['category'] = obj_class.CATEGORY
            return info

        @routes.get("/object_info")
        async def get_object_info(request):
            out = {}
            for x in nodes.NODE_CLASS_MAPPINGS:
                try:
                    out[x] = node_info(x)
                except Exception as e:
                    print(f"[ERROR] An error occurred while retrieving information for the '{x}' node.",
                          file=sys.stderr)
                    traceback.print_exc()
            return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history(request):
            prompt_id = request.match_info.get("prompt_id", None)
            return web.json_response(self.prompt_queue.get_history(prompt_id=prompt_id))

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
            json_data = await request.json()
            json_data = self.trigger_on_prompt(json_data)

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
                valid = execution.validate_prompt(prompt)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                if valid[0]:
                    prompt_id = str(uuid.uuid4())
                    outputs_to_execute = valid[2]
                    self.prompt_queue.put(
                        QueueItem(queue_tuple=(number, prompt_id, prompt, extra_data, outputs_to_execute),
                                  completed=None))
                    response = {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
                    return web.json_response(response)
                else:
                    print("invalid prompt:", valid[1])
                    return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)
            else:
                return web.json_response({"error": "no prompt", "node_errors": []}, status=400)

        @routes.post("/queue")
        async def post_queue(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_queue()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    delete_func = lambda a: a[1] == id_to_delete
                    self.prompt_queue.delete_queue_item(delete_func)

            return web.Response(status=200)

        @routes.post("/interrupt")
        async def post_interrupt(request):
            model_management.interrupt_current_processing()
            return web.Response(status=200)

        @routes.post("/free")
        async def post_free(request):
            json_data = await request.json()
            unload_models = json_data.get("unload_models", False)
            free_memory = json_data.get("free_memory", False)
            if unload_models:
                self.prompt_queue.set_flag("unload_models", unload_models)
            if free_memory:
                self.prompt_queue.set_flag("free_memory", free_memory)
            return web.Response(status=200)

        @routes.post("/history")
        async def post_history(request):
            json_data = await request.json()
            if "clear" in json_data:
                if json_data["clear"]:
                    self.prompt_queue.wipe_history()
            if "delete" in json_data:
                to_delete = json_data['delete']
                for id_to_delete in to_delete:
                    self.prompt_queue.delete_history_item(id_to_delete)

            return web.Response(status=200)

        @routes.get("/api/v1/images/{content_digest}")
        async def get_image(request: web.Request) -> web.FileResponse:
            digest_ = request.match_info['content_digest']
            path = str(os.path.join(user_data_dir("comfyui", "comfyanonymous", roaming=False), digest_))
            return web.FileResponse(path,
                                    headers={"Content-Disposition": f"filename=\"{digest_}.png\""})

        @routes.post("/api/v1/prompts")
        async def post_prompt(request: web.Request) -> web.Response | web.FileResponse:
            # check if the queue is too long
            accept = request.headers.get("accept", "application/json")
            queue_size = self.prompt_queue.size()
            queue_too_busy_size = PromptServer.get_too_busy_queue_size()
            if queue_size > queue_too_busy_size:
                return web.Response(status=429,
                                    reason=f"the queue has {queue_size} elements and {queue_too_busy_size} is the limit for this worker")
            # read the request
            prompt_dict: dict = {}
            if request.headers[aiohttp.hdrs.CONTENT_TYPE] == 'application/json':
                prompt_dict = await request.json()
            elif request.headers[aiohttp.hdrs.CONTENT_TYPE] == 'multipart/form-data':
                try:
                    reader = await request.multipart()
                    async for part in reader:
                        if part is None:
                            break
                        if part.headers[aiohttp.hdrs.CONTENT_TYPE] == 'application/json':
                            prompt_dict = await part.json()
                            if 'prompt' in prompt_dict:
                                prompt_dict = prompt_dict['prompt']
                        elif part.filename:
                            file_data = await part.read(decode=True)
                            # overwrite existing files
                            upload_dir = PromptServer.get_upload_dir()
                            async with aiofiles.open(os.path.join(upload_dir, part.filename), mode='wb') as file:
                                await file.write(file_data)
                except IOError as ioError:
                    return web.Response(status=507, reason=str(ioError))
                except MemoryError as memoryError:
                    return web.Response(status=507, reason=str(memoryError))
                except Exception as ex:
                    return web.Response(status=400, reason=str(ex))

            if len(prompt_dict) == 0:
                return web.Response(status=400, reason="no prompt was specified")

            content_digest = digest(prompt_dict)

            valid = execution.validate_prompt(prompt_dict)
            if not valid[0]:
                return web.Response(status=400, body=valid[1])

            cache_path = os.path.join(user_data_dir("comfyui", "comfyanonymous", roaming=False), content_digest)
            cache_url = f"/api/v1/images/{content_digest}"

            if os.path.exists(cache_path):
                filename__ = os.path.basename(cache_path)
                digest_headers_ = {
                    "Digest": f"SHA-256={content_digest}",
                    "Location": f"/api/v1/images/{content_digest}"
                }
                if accept == "application/json":

                    return web.Response(status=200,
                                        headers=digest_headers_,
                                        content_type="application/json",
                                        body=json.dumps({'urls': [cache_url]}))
                elif accept == "image/png":
                    return web.FileResponse(cache_path,
                                            headers={"Content-Disposition": f"filename=\"{filename__}\"",
                                                     **digest_headers_})
                else:
                    return web.json_response(status=400, reason=f"invalid accept header {accept}")

            # todo: check that the files specified in the InputFile nodes exist

            # convert a valid prompt to the queue tuple this expects
            completed: Future[TaskInvocation | dict] = self.loop.create_future()
            number = self.number
            self.number += 1
            self.prompt_queue.put(
                QueueItem(queue_tuple=(number, str(uuid.uuid4()), prompt_dict, {}, valid[2]),
                          completed=completed))

            try:
                await completed
            except Exception as ex:
                return web.Response(body=str(ex), status=503)
                # expect a single image
            result: TaskInvocation | dict = completed.result()
            outputs_dict: dict = result.outputs if isinstance(result, TaskInvocation) else result
            # find images and read them

            output_images: List[str] = []
            for node_id, node in outputs_dict.items():
                images: List[dict] = []
                if 'images' in node:
                    images = node['images']
                elif (isinstance(node, dict)
                      and 'ui' in node and isinstance(node['ui'], dict)
                      and 'images' in node['ui']):
                    images = node['ui']['images']
                for image_tuple in images:
                    filename_ = image_tuple['abs_path']
                    output_images.append(filename_)

            if len(output_images) > 0:
                image_ = output_images[-1]
                if not os.path.exists(os.path.dirname(cache_path)):
                    try:
                        os.makedirs(os.path.dirname(cache_path))
                    except:
                        pass
                # shutil.copy(image_, cache_path)
                filename = os.path.basename(image_)
                digest_headers_ = {
                    "Digest": f"SHA-256={content_digest}",
                }
                urls_ = [cache_url]
                if len(output_images) == 1:
                    digest_headers_.update({
                        "Location": f"/api/v1/images/{content_digest}",
                        "Content-Disposition": f"filename=\"{filename}\""
                    })

                for image_indv_ in output_images:
                    image_indv_filename_ = os.path.basename(image_indv_)
                    urls_ += [
                        f"http://{self.address}:{self.port}/view?filename={image_indv_filename_}&type=output",
                        urljoin(self.external_address, f"/view?filename={image_indv_filename_}&type=output")
                    ]

                if accept == "application/json":
                    return web.Response(status=200,
                                        content_type="application/json",
                                        headers=digest_headers_,
                                        body=json.dumps({'urls': urls_}))
                elif accept == "image/png":
                    return web.FileResponse(image_,
                                            headers=digest_headers_)
            else:
                return web.Response(status=204)

        @routes.get("/api/v1/prompts")
        async def get_prompt(_: web.Request) -> web.Response:
            history = self.prompt_queue.get_history()
            history_items = list(history.values())
            if len(history_items) == 0:
                return web.Response(status=404)

            last_history_item: HistoryEntry = history_items[-1]
            prompt = last_history_item['prompt'][2]
            return web.json_response(prompt, status=200)

    @property
    def external_address(self):
        return self._external_address if self._external_address is not None else f"http://{'localhost' if self.address == '0.0.0.0' else self.address}:{self.port}"

    @external_address.setter
    def external_address(self, value):
        self._external_address = value

    def add_routes(self):
        self.user_manager.add_routes(self.routes)
        self.app.add_routes(self.routes)

        for name, dir in nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([
                web.static('/extensions/' + quote(name), dir, follow_symlinks=True),
            ])

        self.app.add_routes([
            web.static('/', self.web_root, follow_symlinks=True),
        ])

    def get_queue_info(self):
        prompt_info = {}
        exec_info = {}
        exec_info['queue_remaining'] = self.prompt_queue.get_tasks_remaining()
        prompt_info['exec_info'] = exec_info
        return prompt_info

    async def send(self, event, data, sid=None):
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event, data):
        if not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        if max_size is not None:
            if hasattr(Image, 'Resampling'):
                resampling = Image.Resampling.BILINEAR
            else:
                resampling = Image.ANTIALIAS

            image = ImageOps.contain(image, (max_size, max_size), resampling)
        type_num = 1
        if image_type == "JPEG":
            type_num = 1
        elif image_type == "PNG":
            type_num = 2

        bytesIO = BytesIO()
        header = struct.pack(">I", type_num)
        bytesIO.write(header)
        image.save(bytesIO, format=image_type, quality=95, compress_level=1)
        preview_bytes = bytesIO.getvalue()
        await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, preview_bytes, sid=sid)

    async def send_bytes(self, event, data, sid=None):
        message = self.encode_bytes(event, data)

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_bytes, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_bytes, message)

    async def send_json(self, event, data, sid=None):
        message = {"type": event, "data": data}

        if sid is None:
            sockets = list(self.sockets.values())
            for ws in sockets:
                await send_socket_catch_exception(ws.send_json, message)
        elif sid in self.sockets:
            await send_socket_catch_exception(self.sockets[sid].send_json, message)

    def send_sync(self, event, data, sid=None):
        self.loop.call_soon_threadsafe(
            self.messages.put_nowait, (event, data, sid))

    def queue_updated(self):
        self.send_sync("status", {"status": self.get_queue_info()})

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address, port, verbose=True, call_on_start=None):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, address, port)
        await site.start()

        if verbose:
            print("Starting server\n")
            print("To see the GUI go to: http://{}:{}".format("localhost" if address == "0.0.0.0" else address, port))
        if call_on_start is not None:
            call_on_start(address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception as e:
                print(f"[ERROR] An error occurred during the on_prompt_handler processing")
                traceback.print_exc()

        return json_data

    @classmethod
    def get_output_path(cls, subfolder: str | None = None, filename: str | None = None):
        paths = [path for path in ["output", subfolder, filename] if path is not None and path != ""]
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), *paths)

    @classmethod
    def get_upload_dir(cls) -> str:
        upload_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../input")

        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        return upload_dir

    @classmethod
    def get_too_busy_queue_size(cls):
        # todo: what is too busy of a queue for API clients?
        return 100

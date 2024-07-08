from __future__ import annotations

import asyncio
import glob
import json
import logging
import mimetypes
import os
import struct
import traceback
import uuid
import hashlib
from asyncio import Future, AbstractEventLoop
from enum import Enum
from io import BytesIO
from posixpath import join as urljoin
from typing import List, Optional
from urllib.parse import quote, urlencode

import aiofiles
import aiohttp
import sys
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from aiohttp import web
from can_ada import URL, parse as urlparse  # pylint: disable=no-name-in-module
from typing_extensions import NamedTuple

from .latent_preview_image_encoding import encode_preview_image
from .. import interruption
from .. import model_management
from .. import utils
from ..app.user_manager import UserManager
from ..cli_args import args
from ..client.client_types import FileOutput
from ..cmd import execution
from ..cmd import folder_paths
from ..component_model.abstract_prompt_queue import AbstractPromptQueue, AsyncAbstractPromptQueue
from ..component_model.executor_types import ExecutorToClientProgress, StatusMessage, QueueInfo, ExecInfo
from ..component_model.file_output_path import file_output_path
from ..component_model.files import get_package_as_path
from ..component_model.queue_types import QueueItem, HistoryEntry, BinaryEventTypes, TaskInvocation, ExecutionError, \
    ExecutionStatus
from ..digest import digest
from ..images import open_image
from ..nodes.package_typing import ExportedNodes


class HeuristicPath(NamedTuple):
    filename_heuristic: str
    abs_path: str


async def send_socket_catch_exception(function, message):
    try:
        await function(message)
    except (aiohttp.ClientError, aiohttp.ClientPayloadError, ConnectionResetError) as err:
        logging.warning("send error: {}".format(err))


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
        self.prompt_queue: AbstractPromptQueue | AsyncAbstractPromptQueue | None = None
        self.loop: AbstractEventLoop = loop
        self.messages: asyncio.Queue = asyncio.Queue()
        self.number: int = 0
        self.port: int = 8188
        self._external_address: Optional[str] = None
        self.receive_all_progress_notifications = True

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
            web_root_path = get_package_as_path('comfy', 'web/')
        self.web_root = web_root_path
        routes = web.RouteTableDef()
        self.routes: web.RouteTableDef = routes
        self.last_node_id = None
        self.last_prompt_id = None
        self.client_id = None

        self.on_prompt_handlers = []
        self.nodes: ExportedNodes = ExportedNodes()

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
                        logging.warning('ws connection closed with exception %s' % ws.exception())
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

            for name, dir in self.nodes.EXTENSION_WEB_DIRS.items():
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

        def compare_image_hash(filepath, image):
            # function to compare hashes of two images to see if it already exists, fix to #3465
            if os.path.exists(filepath):
                a = hashlib.sha256()
                b = hashlib.sha256()
                with open(filepath, "rb") as f:
                    a.update(f.read())
                    b.update(image.file.read())
                    image.file.seek(0)
                    f.close()
                return a.hexdigest() == b.hexdigest()
            return False

        async def image_upload(post, image_save_function=None):
            image = post.get("image")
            overwrite = post.get("overwrite")
            image_is_duplicate = False

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
                        if compare_image_hash(filepath, image): #compare hash to prevent saving of duplicates with same name, fix for #3465
                            image_is_duplicate = True
                            break
                        filename = f"{split[0]} ({i}){split[1]}"
                        filepath = os.path.join(full_output_folder, filename)
                        i += 1

                if not image_is_duplicate:
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
                    # todo: any image file we upload that browsers don't support, we should encode a preview
                    # todo: image handling has to be a little bit more standardized, sometimes we want a Pillow Image, sometimes
                    # we want something that will render to the user, sometimes we want tensors
                    if 'preview' in request.rel_url.query or file.endswith(".exr"):
                        with open_image(file) as img:
                            preview_info = request.rel_url.query.get("preview", "jpeg;90").split(';')
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
        async def get_system_stats(request):
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
            obj_class = self.nodes.NODE_CLASS_MAPPINGS[node_class]
            info = {}
            info['input'] = obj_class.INPUT_TYPES()
            info['output'] = obj_class.RETURN_TYPES
            info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
            info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
            info['name'] = node_class
            info['display_name'] = self.nodes.NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in self.nodes.NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
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
            for x in self.nodes.NODE_CLASS_MAPPINGS:
                try:
                    out[x] = node_info(x)
                except Exception as e:
                    logging.error(f"[ERROR] An error occurred while retrieving information for the '{x}' node.")
                    logging.error(traceback.format_exc())
            return web.json_response(out)

        @routes.get("/object_info/{node_class}")
        async def get_object_info_node(request):
            node_class = request.match_info.get("node_class", None)
            out = {}
            if (node_class is not None) and (node_class in self.nodes.NODE_CLASS_MAPPINGS):
                out[node_class] = node_info(node_class)
            return web.json_response(out)

        @routes.get("/history")
        async def get_history(request):
            max_items = request.rel_url.query.get("max_items", None)
            if max_items is not None:
                max_items = int(max_items)
            return web.json_response(self.prompt_queue.get_history(max_items=max_items))

        @routes.get("/history/{prompt_id}")
        async def get_history_prompt(request):
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
            logging.info("got prompt")
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
                    logging.warning("invalid prompt: {}".format(valid[1]))
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
            interruption.interrupt_current_processing()
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

        @routes.post("/api/v1/prompts")
        async def post_api_prompt(request: web.Request) -> web.Response | web.FileResponse:
            # check if the queue is too long
            accept = request.headers.get("accept", "application/json")
            content_type = request.headers.get("content-type", "application/json")
            queue_size = self.prompt_queue.size()
            queue_too_busy_size = PromptServer.get_too_busy_queue_size()
            if queue_size > queue_too_busy_size:
                return web.Response(status=429,
                                    reason=f"the queue has {queue_size} elements and {queue_too_busy_size} is the limit for this worker")
            # read the request
            prompt_dict: dict = {}
            if content_type == 'application/json':
                prompt_dict = await request.json()
            elif content_type == 'multipart/form-data':
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
                return web.Response(status=400, content_type="application/json", body=json.dumps(valid[1]))

            # convert a valid prompt to the queue tuple this expects
            number = self.number
            self.number += 1

            result: TaskInvocation
            completed: Future[TaskInvocation | dict] = self.loop.create_future()
            item = QueueItem(queue_tuple=(number, str(uuid.uuid4()), prompt_dict, {}, valid[2]), completed=completed)

            try:
                if hasattr(self.prompt_queue, "put_async") or isinstance(self.prompt_queue, AsyncAbstractPromptQueue):
                    # this enables span propagation seamlessly
                    result = await self.prompt_queue.put_async(item)
                    if result is None:
                        return web.Response(body="the queue is shutting down", status=503)
                else:
                    self.prompt_queue.put(item)
                    await completed
                    task_invocation_or_dict: TaskInvocation | dict = completed.result()
                    if isinstance(task_invocation_or_dict, dict):
                        result = TaskInvocation(item_id=item.prompt_id, outputs=task_invocation_or_dict, status=ExecutionStatus("success", True, []))
                    else:
                        result = task_invocation_or_dict
            except ExecutionError as exec_exc:
                result = exec_exc.as_task_invocation()
            except Exception as ex:
                return web.Response(body=str(ex), status=500)

            if result.status is not None and result.status.status_str == "error":
                return web.Response(body=json.dumps(result.status._asdict()), status=500, content_type="application/json")
            # find images and read them
            output_images: List[FileOutput] = []
            for node_id, node in result.outputs.items():
                images: List[FileOutput] = []
                if 'images' in node:
                    images = node['images']
                    # todo: does this ever occur?
                elif (isinstance(node, dict)
                      and 'ui' in node and isinstance(node['ui'], dict)
                      and 'images' in node['ui']):
                    images = node['ui']['images']
                for image_tuple in images:
                    output_images.append(image_tuple)

            if len(output_images) > 0:
                main_image = output_images[0]
                filename = main_image["filename"]
                digest_headers_ = {
                    "Digest": f"SHA-256={content_digest}",
                }
                urls_ = []
                if len(output_images) == 1:
                    digest_headers_.update({
                        "Content-Disposition": f"filename=\"{filename}\""
                    })

                for image_indv_ in output_images:
                    local_address = f"http://{self.address}:{self.port}"
                    external_address = self.external_address

                    for base in (local_address, external_address):
                        url: URL = urlparse(urljoin(base, "view"))
                        url_search_dict: FileOutput = dict(image_indv_)
                        del url_search_dict["abs_path"]
                        if "name" in url_search_dict:
                            del url_search_dict["name"]
                        if url_search_dict["subfolder"] == "":
                            del url_search_dict["subfolder"]
                        url.search = f"?{urlencode(url_search_dict)}"
                        urls_.append(str(url))

                if accept == "application/json":
                    return web.Response(status=200,
                                        content_type="application/json",
                                        headers=digest_headers_,
                                        body=json.dumps({
                                            'urls': urls_,
                                            'outputs': result.outputs
                                        }))
                elif accept == "image/png":
                    return web.FileResponse(main_image["abs_path"],
                                            headers=digest_headers_)
            else:
                return web.Response(status=204)

        @routes.get("/api/v1/prompts")
        async def get_api_prompt(_: web.Request) -> web.Response:
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

        # Prefix every route with /api for easier matching for delegation.
        # This is very useful for frontend dev server, which need to forward
        # everything except serving of static files.
        # Currently both the old endpoints without prefix and new endpoints with
        # prefix are supported.
        api_routes = web.RouteTableDef()
        for route in self.routes:
            # Custom nodes might add extra static routes. Only process non-static
            # routes to add /api prefix.
            if isinstance(route, web.RouteDef):
                api_routes.route(route.method, "/api" + route.path)(route.handler, **route.kwargs)
        self.app.add_routes(api_routes)
        self.app.add_routes(self.routes)

        for name, dir in self.nodes.EXTENSION_WEB_DIRS.items():
            self.app.add_routes([
                web.static('/extensions/' + quote(name), dir),
            ])

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
        if event == BinaryEventTypes.UNENCODED_PREVIEW_IMAGE:
            await self.send_image(data, sid=sid)
        elif isinstance(data, (bytes, bytearray)):
            await self.send_bytes(event, data, sid)
        else:
            await self.send_json(event, data, sid)

    def encode_bytes(self, event: int | Enum | str, data):
        # todo: investigate what is propagating these spurious, string-repr'd previews
        if event == repr(BinaryEventTypes.UNENCODED_PREVIEW_IMAGE):
            event = BinaryEventTypes.UNENCODED_PREVIEW_IMAGE.value
        elif event == repr(BinaryEventTypes.PREVIEW_IMAGE):
            event = BinaryEventTypes.PREVIEW_IMAGE.value
        elif isinstance(event, Enum):
            event: int = event.value
        elif not isinstance(event, int):
            raise RuntimeError(f"Binary event types must be integers, got {event}")

        packed = struct.pack(">I", event)
        message = bytearray(packed)
        message.extend(data)
        return message

    async def send_image(self, image_data, sid=None):
        image_type = image_data[0]
        image = image_data[1]
        max_size = image_data[2]
        preview_bytes = encode_preview_image(image, image_type, max_size)
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

    def queue_updated(self, queue_remaining: Optional[int] = None):
        if queue_remaining is None:
            status = {"status": self.get_queue_info()}
        else:
            status = StatusMessage(status=QueueInfo(exec_info=ExecInfo(queue_remaining=queue_remaining)))
        self.send_sync("status", status)

    async def publish_loop(self):
        while True:
            msg = await self.messages.get()
            await self.send(*msg)

    async def start(self, address: str | None, port: int | None, verbose=True, call_on_start=None):
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host=address, port=port)
        await site.start()

        if verbose:
            logging.info("Starting server\n")
            logging.info("To see the GUI go to: http://{}:{}".format("localhost" if address == "0.0.0.0" else address, port))
        if call_on_start is not None:
            call_on_start(address, port)

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def trigger_on_prompt(self, json_data):
        for handler in self.on_prompt_handlers:
            try:
                json_data = handler(json_data)
            except Exception as e:
                logging.warning(f"[ERROR] An error occurred during the on_prompt_handler processing")
                logging.warning(traceback.format_exc())

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
        return args.max_queue_size

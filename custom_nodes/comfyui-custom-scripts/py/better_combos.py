import glob
import os
from nodes import LoraLoader, CheckpointLoaderSimple
import folder_paths
from server import PromptServer
from folder_paths import get_directory_by_type
from aiohttp import web
import shutil


@PromptServer.instance.routes.get("/pysssss/view/{name}")
async def view(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    image_path = folder_paths.get_full_path(
        type, name)
    if not image_path:
        return web.Response(status=404)

    filename = os.path.basename(image_path)
    return web.FileResponse(image_path, headers={"Content-Disposition": f"filename=\"{filename}\""})


@PromptServer.instance.routes.post("/pysssss/save/{name}")
async def save_preview(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    body = await request.json()

    dir = get_directory_by_type(body.get("type", "output"))
    subfolder = body.get("subfolder", "")
    full_output_folder = os.path.join(dir, os.path.normpath(subfolder))

    filepath = os.path.join(full_output_folder, body.get("filename", ""))

    if os.path.commonpath((dir, os.path.abspath(filepath))) != dir:
        return web.Response(status=400)

    image_path = folder_paths.get_full_path(type, name)
    image_path = os.path.splitext(
        image_path)[0] + os.path.splitext(filepath)[1]

    shutil.copyfile(filepath, image_path)

    return web.json_response({
        "image":  type + "/" + os.path.basename(image_path)
    })


@PromptServer.instance.routes.get("/pysssss/examples/{name}")
async def get_examples(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]

    file_path = folder_paths.get_full_path(
        type, name)
    if not file_path:
        return web.Response(status=404)

    file_path_no_ext = os.path.splitext(file_path)[0]
    examples = []

    if os.path.isdir(file_path_no_ext):
        examples += sorted(map(lambda t: os.path.relpath(t, file_path_no_ext),
                               glob.glob(file_path_no_ext + "/*.txt")))

    if os.path.isfile(file_path_no_ext + ".txt"):
        examples += ["notes"]

    return web.json_response(examples)


@PromptServer.instance.routes.post("/pysssss/examples/{name}")
async def save_example(request):
    name = request.match_info["name"]
    pos = name.index("/")
    type = name[0:pos]
    name = name[pos+1:]
    body = await request.json()
    example_name = body["name"]
    example = body["example"]

    file_path = folder_paths.get_full_path(
        type, name)
    if not file_path:
        return web.Response(status=404)

    if not example_name.endswith(".txt"):
        example_name += ".txt"

    file_path_no_ext = os.path.splitext(file_path)[0]
    example_file = os.path.join(file_path_no_ext, example_name)
    if not os.path.exists(file_path_no_ext):
        os.mkdir(file_path_no_ext)
    with open(example_file, 'w', encoding='utf8') as f:
        f.write(example)

    return web.Response(status=201)


@PromptServer.instance.routes.get("/pysssss/images/{type}")
async def get_images(request):
    type = request.match_info["type"]
    names = folder_paths.get_filename_list(type)

    images = {}
    for item_name in names:
        file_name = os.path.splitext(item_name)[0]
        file_path = folder_paths.get_full_path(type, item_name)

        if file_path is None:
            continue

        file_path_no_ext = os.path.splitext(file_path)[0]

        for ext in ["png", "jpg", "jpeg", "preview.png", "preview.jpeg"]:
            if os.path.isfile(file_path_no_ext + "." + ext):
                images[item_name] = f"{type}/{file_name}.{ext}"
                break

    return web.json_response(images)


class LoraLoaderWithImages(LoraLoader):
    RETURN_TYPES = (*LoraLoader.RETURN_TYPES, "STRING",)
    RETURN_NAMES = (*getattr(LoraLoader, "RETURN_NAMES",
                    LoraLoader.RETURN_TYPES), "example")

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["optional"] = {"prompt": ("STRING", {"hidden": True})}
        return types

    def load_lora(self, **kwargs):
        prompt = kwargs.pop("prompt", "")
        return (*super().load_lora(**kwargs), prompt)


class CheckpointLoaderSimpleWithImages(CheckpointLoaderSimple):
    RETURN_TYPES = (*CheckpointLoaderSimple.RETURN_TYPES, "STRING",)
    RETURN_NAMES = (*getattr(CheckpointLoaderSimple, "RETURN_NAMES",
                    CheckpointLoaderSimple.RETURN_TYPES), "example")

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["optional"] = {"prompt": ("STRING", {"hidden": True})}
        return types

    def load_checkpoint(self, **kwargs):
        prompt = kwargs.pop("prompt", "")
        return (*super().load_checkpoint(**kwargs), prompt)


NODE_CLASS_MAPPINGS = {
    "LoraLoader|pysssss": LoraLoaderWithImages,
    "CheckpointLoader|pysssss": CheckpointLoaderSimpleWithImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraLoader|pysssss": "Lora Loader 🐍",
    "CheckpointLoader|pysssss": "Checkpoint Loader 🐍",
}

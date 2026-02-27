from __future__ import annotations
from .dev_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

import os
import shutil
import json
from typing import Union

import server
from aiohttp import web
from aiohttp.web_request import Request
import folder_paths
from folder_paths import models_dir


@server.PromptServer.instance.routes.get("/devtools/fake_model.safetensors")
async def fake_model(request: Request):
    file_path = os.path.join(os.path.dirname(__file__), "fake_model.safetensors")
    return web.FileResponse(file_path)


@server.PromptServer.instance.routes.get("/devtools/cleanup_fake_model")
async def cleanup_fake_model(request: Request):
    model_folder = request.query.get("model_folder", "clip")
    model_path = os.path.join(models_dir, model_folder, "fake_model.safetensors")
    if os.path.exists(model_path):
        os.remove(model_path)
    return web.Response(status=200, text="Fake model cleaned up")


TreeType = dict[str, Union[str, "TreeType"]]


def write_tree_structure(tree: TreeType, base_path: str):
    # Remove existing files and folders in users/workflows
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    # Recreate the base directory
    os.makedirs(base_path, exist_ok=True)

    def write_recursive(current_tree: TreeType, current_path: str):
        for key, value in current_tree.items():
            new_path = os.path.join(current_path, key)
            if isinstance(value, dict):
                # If it's a dictionary, create a new directory and recurse
                os.makedirs(new_path, exist_ok=True)
                write_recursive(value, new_path)
            else:
                # If it's a string, write the content to a file
                with open(new_path, "w") as f:
                    f.write(value)

    write_recursive(tree, base_path)


@server.PromptServer.instance.routes.post("/devtools/setup_folder_structure")
async def setup_folder_structure(request: Request):
    try:
        data = await request.json()
        tree_structure = data.get("tree_structure")
        base_path = os.path.join(
            folder_paths.base_path, data.get("base_path", "users/workflows")
        )

        if not isinstance(tree_structure, dict):
            return web.Response(status=400, text="Invalid tree structure")

        write_tree_structure(tree_structure, base_path)
        return web.Response(status=200, text=f"Folder structure created at {base_path}")
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON data")
    except Exception as e:
        return web.Response(status=500, text=f"Error: {str(e)}")


@server.PromptServer.instance.routes.post("/devtools/set_settings")
async def set_settings(request: Request):
    """Directly set the settings for the user specified via `Comfy.userId`,
    instead of merging with the existing settings."""
    try:
        settings: dict[str, str | bool | int | float] = await request.json()
        user_root = folder_paths.get_user_directory()
        try:
            user_id: str = settings.pop("Comfy.userId")
        except KeyError:
            user_id = "default"
        settings_file_path = os.path.join(user_root, user_id, "comfy.settings.json")

        # Ensure the directory structure exists
        os.makedirs(os.path.dirname(settings_file_path), exist_ok=True)

        with open(settings_file_path, "w") as f:
            f.write(json.dumps(settings, indent=4))
        return web.Response(status=200)
    except Exception as e:
        return web.Response(status=500, text=f"Error: {str(e)}")


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

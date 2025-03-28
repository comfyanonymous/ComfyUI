from __future__ import annotations

import os
import time
import logging
from app.database.db import create_session
import folder_paths
from aiohttp import web
from PIL import Image
from io import BytesIO
from folder_paths import map_legacy, filter_files_extensions, get_full_path
from app.database.models import Tag, Model
from app.model_processor import get_model_previews, model_processor
from utils.web import dumps
from sqlalchemy.orm import joinedload
import sqlalchemy.exc


def bad_request(message: str):
    return web.json_response({"error": message}, status=400)

def missing_field(field: str):
    return bad_request(f"{field} is required")

def not_found(message: str):
    return web.json_response({"error": message + " not found"}, status=404)

class ModelFileManager:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[list[dict], dict[str, float], float]] = {}

    def get_cache(self, key: str, default=None) -> tuple[list[dict], dict[str, float], float] | None:
        return self.cache.get(key, default)

    def set_cache(self, key: str, value: tuple[list[dict], dict[str, float], float]):
        self.cache[key] = value

    def clear_cache(self):
        self.cache.clear()

    def add_routes(self, routes):
        # NOTE: This is an experiment to replace `/models`
        @routes.get("/experiment/models")
        async def get_model_folders(request):
            model_types = list(folder_paths.folder_names_and_paths.keys())
            folder_black_list = ["configs", "custom_nodes"]
            output_folders: list[dict] = []
            for folder in model_types:
                if folder in folder_black_list:
                    continue
                output_folders.append({"name": folder, "folders": folder_paths.get_folder_paths(folder)})
            return web.json_response(output_folders)

        # NOTE: This is an experiment to replace `/models/{folder}`
        @routes.get("/experiment/models/{folder}")
        async def get_all_models(request):
            folder = request.match_info.get("folder", None)
            if not folder in folder_paths.folder_names_and_paths:
                return web.Response(status=404)
            files = self.get_model_file_list(folder)
            return web.json_response(files)

        @routes.get("/experiment/models/preview/{folder}/{path_index}/{filename:.*}")
        async def get_model_preview(request):
            folder_name = request.match_info.get("folder", None)
            path_index = int(request.match_info.get("path_index", None))
            filename = request.match_info.get("filename", None)

            if not folder_name in folder_paths.folder_names_and_paths:
                return web.Response(status=404)

            folders = folder_paths.folder_names_and_paths[folder_name]
            folder = folders[0][path_index]
            full_filename = os.path.join(folder, filename)

            previews = get_model_previews(full_filename)
            default_preview = previews[0] if len(previews) > 0 else None
            if default_preview is None or (isinstance(default_preview, str) and not os.path.isfile(default_preview)):
                return web.Response(status=404)

            try:
                with Image.open(default_preview) as img:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format="WEBP")
                    img_bytes.seek(0)
                    return web.Response(body=img_bytes.getvalue(), content_type="image/webp")
            except:
                return web.Response(status=404)

        @routes.get("/v2/models")
        async def get_models(request):
            with create_session() as session:
                model_path = request.query.get("path", None)
                model_type = request.query.get("type", None)
                query = session.query(Model).options(joinedload(Model.tags))
                if model_path:
                    query = query.filter(Model.path == model_path)
                if model_type:
                    query = query.filter(Model.type == model_type)
                models = query.all()
                if model_path and model_type:
                    if len(models) == 0:
                        return not_found("Model")
                    return web.json_response(models[0].to_dict(), dumps=dumps)
                
                return web.json_response([model.to_dict() for model in models], dumps=dumps)

        @routes.post("/v2/models")
        async def add_model(request):
            with create_session() as session:
                data = await request.json()
                model_type = data.get("type", None)
                model_path = data.get("path", None)

                if not model_type:
                    return missing_field("type")
                if not model_path:
                    return missing_field("path")

                tags = data.pop("tags", [])
                fields = Model.metadata.tables["model"].columns.keys()

                # Validate keys are valid model fields
                for key in data.keys():
                    if key not in fields:
                        return bad_request(f"Invalid field: {key}")

                # Validate file exists
                if not get_full_path(model_type, model_path):
                    return not_found(f"File '{model_type}/{model_path}'")

                model = Model()
                for field in fields:
                    if field in data:
                        setattr(model, field, data[field])

                model.tags = session.query(Tag).filter(Tag.id.in_(tags)).all()
                for tag in tags:
                    if tag not in [t.id for t in model.tags]:
                        return not_found(f"Tag '{tag}'")

                try:
                    session.add(model)
                    session.commit()
                except sqlalchemy.exc.IntegrityError as e:
                    session.rollback()
                    return bad_request(e.orig.args[0])

                model_processor.run()

                return web.json_response(model.to_dict(), dumps=dumps)
            
        @routes.delete("/v2/models")
        async def delete_model(request):
            with create_session() as session:
                model_path = request.query.get("path", None)
                model_type = request.query.get("type", None)
                if not model_path:
                    return missing_field("path")
                if not model_type:
                    return missing_field("type")
                
                full_path = get_full_path(model_type, model_path)
                if full_path:
                    return bad_request("Model file exists, please delete the file before deleting the model record.")

                model = session.query(Model).filter(Model.path == model_path, Model.type == model_type).first()
                if not model:
                    return not_found("Model")
                session.delete(model)
                session.commit()
                return web.Response(status=204)

        @routes.get("/v2/tags")
        async def get_tags(request):
            with create_session() as session:
                tags = session.query(Tag).all()
                return web.json_response(
                    [{"id": tag.id, "name": tag.name} for tag in tags]
                )

        @routes.post("/v2/tags")
        async def create_tag(request):
            with create_session() as session:
                data = await request.json()
                name = data.get("name", None)
                if not name:
                    return missing_field("name")
                tag = Tag(name=name)
                session.add(tag)
                session.commit()
                return web.json_response({"id": tag.id, "name": tag.name})
            
        @routes.delete("/v2/tags")
        async def delete_tag(request):
            with create_session() as session:
                tag_id = request.query.get("id", None)
                if not tag_id:
                    return missing_field("id")
                tag = session.query(Tag).filter(Tag.id == tag_id).first()
                if not tag:
                    return not_found("Tag")
                session.delete(tag)
                session.commit()
                return web.Response(status=204)

        @routes.post("/v2/models/tags")
        async def add_model_tag(request):
            with create_session() as session:
                data = await request.json()
                tag_id = data.get("tag", None)
                model_path = data.get("path", None)
                model_type = data.get("type", None)

                if tag_id is None:
                    return missing_field("tag")
                if model_path is None:
                    return missing_field("path")
                if model_type is None:
                    return missing_field("type")

                try:
                    tag_id = int(tag_id)
                except ValueError:
                    return bad_request("Invalid tag id")

                tag = session.query(Tag).filter(Tag.id == tag_id).first()
                model = session.query(Model).filter(Model.path == model_path, Model.type == model_type).first()
                if not model:
                    return not_found("Model")
                model.tags.append(tag)
                session.commit()
                return web.json_response(model.to_dict(), dumps=dumps)

        @routes.delete("/v2/models/tags")
        async def delete_model_tag(request):
            with create_session() as session:
                tag_id = request.query.get("tag", None)
                model_path = request.query.get("path", None)
                model_type = request.query.get("type", None)

                if tag_id is None:
                    return missing_field("tag")
                if model_path is None:
                    return missing_field("path")
                if model_type is None:
                    return missing_field("type")
                
                try:
                    tag_id = int(tag_id)
                except ValueError:
                    return bad_request("Invalid tag id")

                model = session.query(Model).filter(Model.path == model_path, Model.type == model_type).first()
                if not model:
                    return not_found("Model")
                model.tags = [tag for tag in model.tags if tag.id != tag_id]
                session.commit()
                return web.Response(status=204)
        
            

        @routes.get("/v2/models/missing")
        async def get_missing_models(request):
            return web.json_response(model_processor.missing_models)

    def get_model_file_list(self, folder_name: str):
        folder_name = map_legacy(folder_name)
        folders = folder_paths.folder_names_and_paths[folder_name]
        output_list: list[dict] = []

        for index, folder in enumerate(folders[0]):
            if not os.path.isdir(folder):
                continue
            out = self.cache_model_file_list_(folder)
            if out is None:
                out = self.recursive_search_models_(folder, index)
                self.set_cache(folder, out)
            output_list.extend(out[0])

        return output_list

    def cache_model_file_list_(self, folder: str):
        model_file_list_cache = self.get_cache(folder)

        if model_file_list_cache is None:
            return None
        if not os.path.isdir(folder):
            return None
        if os.path.getmtime(folder) != model_file_list_cache[1]:
            return None
        for x in model_file_list_cache[1]:
            time_modified = model_file_list_cache[1][x]
            folder = x
            if os.path.getmtime(folder) != time_modified:
                return None

        return model_file_list_cache

    def recursive_search_models_(self, directory: str, pathIndex: int) -> tuple[list[str], dict[str, float], float]:
        if not os.path.isdir(directory):
            return [], {}, time.perf_counter()

        excluded_dir_names = [".git"]
        # TODO use settings
        include_hidden_files = False

        result: list[str] = []
        dirs: dict[str, float] = {}

        for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
            subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
            if not include_hidden_files:
                subdirs[:] = [d for d in subdirs if not d.startswith(".")]
                filenames = [f for f in filenames if not f.startswith(".")]

            filenames = filter_files_extensions(filenames, folder_paths.supported_pt_extensions)

            for file_name in filenames:
                try:
                    relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
                    result.append(relative_path)
                except:
                    logging.warning(f"Warning: Unable to access {file_name}. Skipping this file.")
                    continue

            for d in subdirs:
                path: str = os.path.join(dirpath, d)
                try:
                    dirs[path] = os.path.getmtime(path)
                except FileNotFoundError:
                    logging.warning(f"Warning: Unable to access {path}. Skipping this path.")
                    continue

        return [{"name": f, "pathIndex": pathIndex} for f in result], dirs, time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_cache()

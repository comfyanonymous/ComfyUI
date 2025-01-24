from __future__ import annotations

import os
import logging
import folder_paths
import mimetypes
import shutil
import traceback
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from typing import Literal


class OutputManager:
    def __init__(self) -> None:
        self.cache: dict[str, tuple[list, float]] = {}
        self.output_uri = folder_paths.get_output_directory()

    def get_cache(self, key: str):
        return self.cache.get(key, ([], 0))

    def set_cache(self, key: str, value: tuple[list, float]):
        self.cache[key] = value

    def rm_cache(self, key: str):
        if key in self.cache:
            del self.cache[key]

    def add_routes(self, routes) -> None:
        @routes.get("/output{pathname:.*}")
        async def get_output_file_or_files(request):
            pathname = request.match_info.get("pathname", None)
            try:
                filepath = self.get_output_filepath(pathname)

                if os.path.isfile(filepath):

                    preview_type = request.query.get("preview_type", None)
                    if not preview_type:
                        return web.FileResponse(filepath)

                    # get image preview
                    if self.assert_file_type(filepath, ["image"]):
                        image_data = self.get_image_preview_data(filepath)
                        return web.Response(body=image_data.getvalue(), content_type="image/webp")

                    # TODO get video cover preview

                elif os.path.isdir(filepath):
                    files = self.get_folder_items(filepath)
                    return web.json_response(files)

                return web.Response(status=404)
            except Exception:
                logging.error(f"File '{pathname}' retrieval failed")
                logging.error(traceback.format_exc())
                return web.Response(status=500)

        @routes.delete("/output{pathname:.*}")
        async def delete_output_file_or_files(request):
            pathname = request.match_info.get("pathname", None)
            try:
                filepath = self.get_output_filepath(pathname)

                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                    self.rm_cache(filepath)
                return web.Response(status=200)
            except Exception:
                logging.error(f"File '{pathname}' deletion failed")
                logging.error(traceback.format_exc())
                return web.Response(status=500)

    def get_output_filepath(self, pathname: str):
        return f"{self.output_uri}/{pathname}"

    def get_folder_items(self, folder: str):
        result, m_time = self.get_cache(folder)
        folder_m_time = os.path.getmtime(folder)

        if folder_m_time == m_time:
            return result

        result = []

        def get_file_info(entry: os.DirEntry[str]):
            filepath = entry.path
            is_dir = entry.is_dir()

            if not is_dir and not self.assert_file_type(filepath, ["image", "video", "audio"]):
                return None

            stat = entry.stat()
            return {
                "name": entry.name,
                "type": "folder" if entry.is_dir() else self.get_file_content_type(filepath),
                "size": 0 if is_dir else stat.st_size,
                "createTime": round(stat.st_ctime_ns / 1000000),
                "modifyTime": round(stat.st_mtime_ns / 1000000),
            }

        with os.scandir(folder) as it, ThreadPoolExecutor() as executor:
            future_to_entry = {executor.submit(get_file_info, entry): entry for entry in it}
            for future in as_completed(future_to_entry):
                file_info = future.result()
                if file_info is None:
                    continue
                result.append(file_info)

        self.set_cache(folder, (result, os.path.getmtime(folder)))
        return result

    def assert_file_type(self, filename: str, content_types: Literal["image", "video", "audio"]):
        content_type = self.get_file_content_type(filename)
        if not content_type:
            return False
        return content_type in content_types

    def get_file_content_type(self, filename: str):
        extension_mimetypes_cache = folder_paths.extension_mimetypes_cache

        extension = filename.split(".")[-1]
        content_type = None
        if extension not in extension_mimetypes_cache:
            mime_type, _ = mimetypes.guess_type(filename, strict=False)
            if mime_type:
                content_type = mime_type.split("/")[0]
                extension_mimetypes_cache[extension] = content_type
        else:
            content_type = extension_mimetypes_cache[extension]

        return content_type

    def get_image_preview_data(self, filename: str):
        with Image.open(filename) as img:
            max_size = 128

            old_width, old_height = img.size
            scale = min(max_size / old_width, max_size / old_height)

            if scale >= 1:
                new_width, new_height = old_width, old_height
            else:
                new_width = int(old_width * scale)
                new_height = int(old_height * scale)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="WEBP")
            img_byte_arr.seek(0)
            return img_byte_arr

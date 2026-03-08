import mimetypes
import os
import posixpath
import urllib.parse
import io
import json
import uuid
import time
import threading
import hashlib
import re
import traceback
import logging
import gc
import folder_paths
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence
import execution
import nodes

# ... (lines 1-420 skipped, focusing on view_image)

async def view_image(request):
    filename = request.query.get("filename", None)
    type = request.query.get("type", "output")
    subfolder = request.query.get("subfolder", "")

    if filename is None:
        return web.Response(status=404)

    if type == "output":
        directory = folder_paths.get_output_directory()
    elif type == "temp":
        directory = folder_paths.get_temp_directory()
    elif type == "input":
        directory = folder_paths.get_input_directory()
    else:
        return web.Response(status=400)

    image_path = os.path.abspath(os.path.join(directory, subfolder, filename))
    if os.path.commonpath([image_path, directory]) != directory:
        return web.Response(status=403)

    if not os.path.exists(image_path):
        return web.Response(status=404)

    # Use attachment to follow RFC 2183 for file downloads
    filename = os.path.basename(image_path)
    content_disposition = f'attachment; filename="{filename}"'
    return web.FileResponse(image_path, headers={"Content-Disposition": content_disposition})

# ... (rest of file remains unchanged)

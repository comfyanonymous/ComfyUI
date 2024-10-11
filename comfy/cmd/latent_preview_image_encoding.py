from __future__ import annotations

import struct
from io import BytesIO
from typing import Literal

import PIL.Image
from PIL import Image, ImageOps


def encode_preview_image(image: PIL.Image.Image, image_type: Literal["JPEG", "PNG"], max_size: int, node_id: str = "", task_id: str = "") -> bytes:
    if max_size is not None:
        if hasattr(Image, 'Resampling'):
            resampling = Image.Resampling.BILINEAR
        else:
            resampling = Image.Resampling.LANCZOS

        image = ImageOps.contain(image, (max_size, max_size), resampling)

    has_ids = (node_id is not None and len(node_id) > 0) or (task_id is not None and len(task_id) > 0)

    if image_type == "JPEG":
        type_num = 3 if has_ids else 1
    elif image_type == "PNG":
        type_num = 4 if has_ids else 2
    else:
        raise ValueError(f"Unsupported image type: {image_type}")

    bytesIO = BytesIO()

    if has_ids:
        # Pack the header with type_num, node_id length, task_id length
        node_id = node_id or ""
        task_id = task_id or ""
        header = struct.pack(">III", type_num, len(node_id), len(task_id))
        bytesIO.write(header)
        bytesIO.write(node_id.encode('utf-8'))
        bytesIO.write(task_id.encode('utf-8'))
    else:
        # Pack only the type_num for types 1 and 2
        header = struct.pack(">I", type_num)
        bytesIO.write(header)

    image.save(bytesIO, format=image_type, quality=95, compress_level=1)
    preview_bytes = bytesIO.getvalue()
    return preview_bytes

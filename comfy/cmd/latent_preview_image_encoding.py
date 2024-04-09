from __future__ import annotations

import struct
from io import BytesIO
from typing import Literal

import PIL.Image
from PIL import Image, ImageOps


def encode_preview_image(image: PIL.Image.Image, image_type: Literal["JPEG", "PNG"], max_size: int):
    if max_size is not None:
        if hasattr(Image, 'Resampling'):
            resampling = Image.Resampling.BILINEAR
        else:
            resampling = Image.Resampling.LANCZOS

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
    return preview_bytes

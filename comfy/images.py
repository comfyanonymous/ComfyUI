import os.path
from contextlib import contextmanager

import cv2
from PIL import Image

from . import node_helpers


def _open_exr(exr_path) -> Image.Image:
    return Image.fromarray(cv2.imread(exr_path, cv2.IMREAD_COLOR))


@contextmanager
def open_image(file_path: str) -> Image.Image:
    _, ext = os.path.splitext(file_path)
    if ext == ".exr":
        yield _open_exr(file_path)
    else:
        with node_helpers.open_image(file_path) as image:
            yield image

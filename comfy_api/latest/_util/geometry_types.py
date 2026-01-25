import shutil
from io import BytesIO
from pathlib import Path
from typing import IO

import torch


class VOXEL:
    def __init__(self, data: torch.Tensor):
        self.data = data


class MESH:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        self.vertices = vertices
        self.faces = faces


class File3D:
    """Class representing a 3D file from a file path or binary stream.

    Supports both disk-backed (file path) and memory-backed (BytesIO) storage.
    """

    def __init__(self, path: str | IO[bytes], file_format: str = ""):
        self._path = path
        self._format = file_format or self._infer_format()

    def _infer_format(self) -> str:
        if isinstance(self._path, str):
            return Path(self._path).suffix.lstrip(".").lower()
        return ""

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, value: str) -> None:
        self._format = value.lstrip(".").lower() if value else ""

    @property
    def is_disk_backed(self) -> bool:
        return isinstance(self._path, str)

    def get_source(self) -> str | IO[bytes]:
        if isinstance(self._path, str):
            return self._path
        if hasattr(self._path, "seek"):
            self._path.seek(0)
        return self._path

    @property
    def data(self) -> BytesIO:
        if isinstance(self._path, str):
            with open(self._path, "rb") as f:
                result = BytesIO(f.read())
            return result
        if hasattr(self._path, "seek"):
            self._path.seek(0)
        if isinstance(self._path, BytesIO):
            return self._path
        return BytesIO(self._path.read())

    def save_to(self, path: str) -> str:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self._path, str):
            if Path(self._path).resolve() != dest.resolve():
                shutil.copy2(self._path, dest)
        else:
            if hasattr(self._path, "seek"):
                self._path.seek(0)
            with open(dest, "wb") as f:
                f.write(self._path.read())
        return str(dest)

    def get_bytes(self) -> bytes:
        if isinstance(self._path, str):
            return Path(self._path).read_bytes()
        if hasattr(self._path, "seek"):
            self._path.seek(0)
        return self._path.read()

    def __repr__(self) -> str:
        if isinstance(self._path, str):
            return f"File3D(path={self._path!r}, format={self._format!r})"
        return f"File3D(<stream>, format={self._format!r})"

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

    def __init__(self, source: str | IO[bytes], file_format: str = ""):
        self._source = source
        self._format = file_format or self._infer_format()

    def _infer_format(self) -> str:
        if isinstance(self._source, str):
            return Path(self._source).suffix.lstrip(".").lower()
        return ""

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, value: str) -> None:
        self._format = value.lstrip(".").lower() if value else ""

    @property
    def is_disk_backed(self) -> bool:
        return isinstance(self._source, str)

    def get_source(self) -> str | IO[bytes]:
        if isinstance(self._source, str):
            return self._source
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        return self._source

    def get_data(self) -> BytesIO:
        if isinstance(self._source, str):
            with open(self._source, "rb") as f:
                result = BytesIO(f.read())
            return result
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        if isinstance(self._source, BytesIO):
            return self._source
        return BytesIO(self._source.read())

    def save_to(self, path: str) -> str:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self._source, str):
            if Path(self._source).resolve() != dest.resolve():
                shutil.copy2(self._source, dest)
        else:
            if hasattr(self._source, "seek"):
                self._source.seek(0)
            with open(dest, "wb") as f:
                f.write(self._source.read())
        return str(dest)

    def get_bytes(self) -> bytes:
        if isinstance(self._source, str):
            return Path(self._source).read_bytes()
        if hasattr(self._source, "seek"):
            self._source.seek(0)
        return self._source.read()

    def __repr__(self) -> str:
        if isinstance(self._source, str):
            return f"File3D(source={self._source!r}, format={self._format!r})"
        return f"File3D(<stream>, format={self._format!r})"

from __future__ import annotations

import dataclasses
import os
from typing import List, Set, Any, Iterator, Sequence, Dict, NamedTuple

supported_pt_extensions = frozenset(['.ckpt', '.pt', '.bin', '.pth', '.safetensors', '.pkl', '.sft'])
extension_mimetypes_cache = {
    "webp": "image",
}


@dataclasses.dataclass
class FolderPathsTuple:
    folder_name: str
    paths: List[str] = dataclasses.field(default_factory=list)
    supported_extensions: Set[str] = dataclasses.field(default_factory=lambda: set(supported_pt_extensions))

    def __getitem__(self, item: Any):
        if item == 0:
            return self.paths
        if item == 1:
            return self.supported_extensions
        else:
            raise RuntimeError("unsupported tuple index")

    def __add__(self, other: "FolderPathsTuple"):
        assert self.folder_name == other.folder_name
        # todo: make sure the paths are actually unique, as this method intends
        new_paths = list(frozenset(self.paths + other.paths))
        new_supported_extensions = self.supported_extensions | other.supported_extensions
        return FolderPathsTuple(self.folder_name, new_paths, new_supported_extensions)

    def __iter__(self) -> Iterator[Sequence[str]]:
        yield self.paths
        yield self.supported_extensions


class FolderNames:
    def __init__(self, default_new_folder_path: str):
        self.contents: Dict[str, FolderPathsTuple] = dict()
        self.default_new_folder_path = default_new_folder_path

    def __getitem__(self, item) -> FolderPathsTuple:
        if not isinstance(item, str):
            raise RuntimeError("expected folder path")
        if item not in self.contents:
            default_path = os.path.join(self.default_new_folder_path, item)
            os.makedirs(default_path, exist_ok=True)
            self.contents[item] = FolderPathsTuple(item, paths=[default_path], supported_extensions=set())
        return self.contents[item]

    def __setitem__(self, key: str, value: FolderPathsTuple):
        assert isinstance(key, str)
        if isinstance(value, tuple):
            paths, supported_extensions = value
            value = FolderPathsTuple(key, paths, supported_extensions)
        self.contents[key] = value

    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        return iter(self.contents)

    def __delitem__(self, key):
        del self.contents[key]

    def items(self):
        return self.contents.items()

    def values(self):
        return self.contents.values()

    def keys(self):
        return self.contents.keys()

    def get(self, key, __default=None):
        return self.contents.get(key, __default)


class SaveImagePathResponse(NamedTuple):
    full_output_folder: str
    filename: str
    counter: int
    subfolder: str
    filename_prefix: str

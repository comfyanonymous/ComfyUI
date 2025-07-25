from __future__ import annotations
import comfy.utils
import folder_paths
import logging
from abc import ABC, abstractmethod
from typing import Any
import torch

class ResourceKey(ABC):
    Type = Any
    def __init__(self):
        ...

class TorchDictFolderFilename(ResourceKey):
    '''Key for requesting a torch file via file_name from a folder category.'''
    Type = dict[str, torch.Tensor]
    def __init__(self, folder_name: str, file_name: str):
        self.folder_name = folder_name
        self.file_name = file_name

    def __hash__(self):
        return hash((self.folder_name, self.file_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TorchDictFolderFilename):
            return False
        return self.folder_name == other.folder_name and self.file_name == other.file_name

    def __str__(self):
        return f"{self.folder_name} -> {self.file_name}"

class Resources(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def get(self, key: ResourceKey, default: Any=...) -> Any:
        pass

class ResourcesLocal(Resources):
    def __init__(self):
        super().__init__()
        self.local_resources: dict[ResourceKey, Any] = {}

    def get(self, key: ResourceKey, default: Any=...) -> Any:
        cached = self.local_resources.get(key, None)
        if cached is not None:
            logging.info(f"Using cached resource '{key}'")
            return cached
        logging.info(f"Loading resource '{key}'")
        to_return = None
        if isinstance(key, TorchDictFolderFilename):
            if default is ...:
                to_return = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise(key.folder_name, key.file_name), safe_load=True)
            else:
                full_path = folder_paths.get_full_path(key.folder_name, key.file_name)
                if full_path is not None:
                    to_return = comfy.utils.load_torch_file(full_path, safe_load=True)

        if to_return is not None:
            self.local_resources[key] = to_return
            return to_return
        if default is not ...:
            return default
        raise Exception(f"Unsupported resource key type: {type(key)}")


class _RESOURCES:
    ResourceKey = ResourceKey
    TorchDictFolderFilename = TorchDictFolderFilename
    Resources = Resources
    ResourcesLocal = ResourcesLocal

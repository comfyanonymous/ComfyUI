from abc import ABC, abstractmethod
from typing import Any, TypedDict
import os
import logging
import comfy.utils

import folder_paths

class AssetMetadata(TypedDict):
    device: Any
    return_metadata: bool
    subdir: str
    download_url: str

class AssetInfo:
    def __init__(self, hash: str=None, name: str=None, tags: list[str]=None, metadata: AssetMetadata={}):
        self.hash = hash
        self.name = name
        self.tags = tags
        self.metadata = metadata


class ReturnedAssetABC(ABC):
    def __init__(self, mimetype: str):
        self.mimetype = mimetype


class ModelReturnedAsset(ReturnedAssetABC):
    def __init__(self, model: dict[str, str] | tuple[dict[str, str], dict[str, str]]):
        super().__init__("model")
        self.model = model


class AssetResolverABC(ABC):
    @abstractmethod
    def resolve(self, asset_info: AssetInfo) -> ReturnedAssetABC:
        ...


class LocalAssetResolver(AssetResolverABC):
    def resolve(self, asset_info: AssetInfo) -> ReturnedAssetABC:
        # currently only supports models - make sure models is in the tags
        if "models" not in asset_info.tags:
            return None
        # TODO: if hash exists, call model processor to try to get info about model:
        if asset_info.hash:
            ...
        # if subdir metadata and name exists, use that as the model name going forward
        if "subdir" in asset_info.metadata and asset_info.name:
            relative_path = os.path.join(asset_info.metadata["subdir"], asset_info.name)
            # the good ol' bread and butter - folder_paths's keys as tags
            folder_keys = folder_paths.folder_names_and_paths.keys()
            parent_paths = []
            for tag in asset_info.tags:
                if tag in folder_keys:
                    parent_paths.append(tag)
            if len(parent_paths) == 0:
                return None
            # now we have the parent keys, we can try to get the local path
            chosen_parent = None
            full_path = None
            for parent_path in parent_paths:
                full_path = folder_paths.get_full_path(parent_path, relative_path)
                if full_path:
                    chosen_parent = parent_path
                    break
            logging.info(f"Resolved {asset_info.name} to {full_path} in {chosen_parent}")
            # we know the path, so load the model and return it
            model = comfy.utils.load_torch_file(full_path, safe_load=True, device=asset_info.metadata.get("device", None), return_metadata=asset_info.metadata.get("return_metadata", False))
            return ModelReturnedAsset(model)
        # TODO: if name exists, try to find model by name in all subdirs of parent paths
        if asset_info.name:
            ...
        # TODO: if download_url metadata exists, download the model and load it
        if asset_info.metadata.get("download_url", None):
            ...
        return None


resolvers: list[AssetResolverABC] = []


def resolve(asset_info: AssetInfo) -> Any:
    global resolvers
    for resolver in resolvers:
        try:
            return resolver.resolve(asset_info)
        except Exception as e:
            logging.error(f"Error resolving asset {asset_info.hash}: {e}")
    return None

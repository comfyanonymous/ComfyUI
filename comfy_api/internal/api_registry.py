from typing import Type, List, NamedTuple
from comfy_api.internal.singleton import ProxiedSingleton
from packaging import version as packaging_version


class ComfyAPIBase(ProxiedSingleton):
    def __init__(self):
        pass


class ComfyAPIWithVersion(NamedTuple):
    version: str
    api_class: Type[ComfyAPIBase]


def parse_version(version_str: str) -> packaging_version.Version:
    """
    Parses a version string into a packaging_version.Version object.
    Raises ValueError if the version string is invalid.
    """
    if version_str == "latest":
        return packaging_version.parse("9999999.9999999.9999999")
    return packaging_version.parse(version_str)


registered_versions: List[ComfyAPIWithVersion] = []


def register_versions(versions: List[ComfyAPIWithVersion]):
    versions.sort(key=lambda x: parse_version(x.version))
    global registered_versions
    registered_versions = versions


def get_all_versions() -> List[ComfyAPIWithVersion]:
    """
    Returns a list of all registered ComfyAPI versions.
    """
    return registered_versions

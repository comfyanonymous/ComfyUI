from __future__ import annotations

import json
import os
from importlib import resources as resources
from typing import Optional


def get_path_as_dict(config_dict_or_path: str | dict | None, config_path_inside_package: str, package: str = 'comfy') -> dict:
    """
    Given a package and a filename inside the package, returns it as a JSON dict; or, returns the file pointed to by
    config_dict_or_path, when it is not None and when it exists

    :param config_dict_or_path: a file path or dict pointing to a JSON file. If it exists, it is parsed and returned. Otherwise, when None, falls back to other defaults
    :param config_path_inside_package: a filename inside a package
    :param package: a package containing the file
    :return:
    """
    config: dict | None = None

    if config_dict_or_path is None:
        config_dict_or_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), config_path_inside_package)

    if isinstance(config_dict_or_path, str):
        if config_dict_or_path.startswith("{"):
            config = json.loads(config_dict_or_path)
        else:
            if not os.path.exists(config_dict_or_path):
                with resources.as_file(resources.files(package) / config_path_inside_package) as config_path:
                    with open(config_path, mode="rt", encoding="utf-8") as f:
                        config = json.load(f)
            else:
                with open(config_dict_or_path, mode="rt", encoding="utf-8") as f:
                    config = json.load(f)
    elif isinstance(config_dict_or_path, dict):
        config = config_dict_or_path

    assert config is not None
    return config


def get_package_as_path(package: str, subdir: Optional[str] = None) -> str:
    """
    Gets the path on the file system to a package. This unpacks it completely.
    :param package: the package containing the files
    :param subdir: if specified, a subdirectory containing files (and not python packages), such as a web/ directory inside a package
    :return:
    """
    traversable = resources.files(package)
    if subdir is not None:
        traversable = traversable / subdir
    return os.path.commonpath(list(map(str, traversable.iterdir())))

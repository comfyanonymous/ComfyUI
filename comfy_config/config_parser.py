import os
from typing import Optional

import tomlkit
import tomlkit.exceptions

import logging

from comfy_config.types import (
    ComfyConfig,
    License,
    Model,
    ProjectConfig,
    PyProjectConfig,
    URLs,
)

"""
Original implementation comes from https://github.com/Comfy-Org/comfy-cli/blob/2e36f33dd39ef43b5acf7d1fc5acc5e01be92360/comfy_cli/registry/config_parser.py#L146
 
Extract configuration from a custom node directory's pyproject.toml file.

This function reads and parses the pyproject.toml file in the specified directory
to extract project and ComfyUI-specific configuration information. If no 
pyproject.toml file is found, it creates a minimal configuration using the 
folder name as the project name.

Args:
    path (str): Path to the directory containing the pyproject.toml file.
               If pyproject.toml doesn't exist, the folder name will be used
               as the default project name.

Returns:
    Optional[PyProjectConfig]: A PyProjectConfig object containing:
        - project: Basic project information (name, version, dependencies, etc.)
        - tool_comfy: ComfyUI-specific configuration (publisher_id, models, etc.)
        Returns None if configuration extraction fails.

Notes:
    - If pyproject.toml is missing, creates a default config with folder name

Example:
    >>> from comfy_config import config_parser
    >>> custom_node_dir = os.path.dirname(os.path.realpath(__file__))
    >>> project_config = config_parser.extract_node_configuration(custom_node_dir)
    >>> print(project_config.project.name)  # "my_custom_node" or name from pyproject.toml
    >>> nodes.EXTENSION_WEB_DIRS[project_config.project.name] = js_dir
"""
def extract_node_configuration(
    path,
) -> Optional[PyProjectConfig]:
    folder_name = os.path.basename(path)

    path = os.path.join(path, "pyproject.toml")

    if not os.path.isfile(path):
        logging.warning("No pyproject.toml file found in the current directory, will use custom node folder name as project name as default.")

        project = ProjectConfig(
            name=folder_name,
        )

        return PyProjectConfig(project=project)

    with open(path, "r") as file:
        data = tomlkit.load(file)

    project_data = data.get("project", {})
    urls_data = project_data.get("urls", {})
    comfy_data = data.get("tool", {}).get("comfy", {})

    license_data = project_data.get("license", {})
    if isinstance(license_data, str):
        license = License(text=license_data)
        logging.warning(
            'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
        )
    elif isinstance(license_data, dict):
        if "file" in license_data or "text" in license_data:
            license = License(file=license_data.get("file", ""), text=license_data.get("text", ""))
        else:
            logging.warning(
                'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
            )
            license = License()
    else:
        license = License()
        logging.warning(
            'Warning: License should be in one of these two formats: license = {file = "LICENSE"} OR license = {text = "MIT License"}. Please check the documentation: https://docs.comfy.org/registry/specifications.'
        )

    project = ProjectConfig(
        name=project_data.get("name", ""),
        description=project_data.get("description", ""),
        version=project_data.get("version", ""),
        requires_python=project_data.get("requires-python", ""),
        dependencies=project_data.get("dependencies", []),
        license=license,
        urls=URLs(
            homepage=urls_data.get("Homepage", ""),
            documentation=urls_data.get("Documentation", ""),
            repository=urls_data.get("Repository", ""),
            issues=urls_data.get("Issues", ""),
        ),
    )

    comfy = ComfyConfig(
        publisher_id=comfy_data.get("PublisherId", ""),
        display_name=comfy_data.get("DisplayName", ""),
        icon=comfy_data.get("Icon", ""),
        models=[Model(location=m["location"], model_url=m["model_url"]) for m in comfy_data.get("Models", [])],
        includes=comfy_data.get("includes", []),
    )

    return PyProjectConfig(project=project, tool_comfy=comfy)

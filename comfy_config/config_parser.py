import os
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from pydantic_settings import PydanticBaseSettingsSource, TomlConfigSettingsSource
import logging

from comfy_config.types import (
    ComfyConfig,
    ProjectConfig,
    PyProjectConfig,
    PyProjectSettings
)

"""
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
def extract_node_configuration(path) -> Optional[PyProjectConfig]:
    folder_name = os.path.basename(path)
    toml_path = Path(path) / "pyproject.toml"

    if not toml_path.exists():
        logging.warning("No pyproject.toml file found, using folder name as project name")

        try:
            project = ProjectConfig(name=folder_name)
            comfy = ComfyConfig()
            return PyProjectConfig(project=project, tool_comfy=comfy)
        except ValidationError as e:
            logging.error(f"Failed to create default configuration: {e}")
            return None

    try:
        raw_settings = load_pyproject_settings(toml_path)

        project_data = raw_settings.project

        tool_data = raw_settings.tool
        comfy_data = tool_data.get("comfy", {}) if tool_data else {}

        return PyProjectConfig(project=project_data, tool_comfy=comfy_data)
    except Exception as e:
        logging.error(f"Failed to load configuration from {toml_path}: {e}")
        return None


def load_pyproject_settings(toml_path: Path) -> PyProjectSettings:
    class PyProjectLoader(PyProjectSettings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ):
            return (TomlConfigSettingsSource(settings_cls, toml_path),)

    return PyProjectLoader()

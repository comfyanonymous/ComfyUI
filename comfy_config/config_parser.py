import os
from pathlib import Path
from typing import Optional

from pydantic_settings import PydanticBaseSettingsSource, TomlConfigSettingsSource

from comfy_config.types import (
    ComfyConfig,
    ProjectConfig,
    PyProjectConfig,
    PyProjectSettings
)

"""
Extract configuration from a custom node directory's pyproject.toml file or a Python file.

This function reads and parses the pyproject.toml file in the specified directory
to extract project and ComfyUI-specific configuration information. If no
pyproject.toml file is found, it creates a minimal configuration using the
folder name as the project name. If a Python file is provided, it uses the
file name (without extension) as the project name.

Args:
    path (str): Path to the directory containing the pyproject.toml file, or
               path to a .py file. If pyproject.toml doesn't exist in a directory,
               the folder name will be used as the default project name. If a .py
               file is provided, the filename (without .py extension) will be used
               as the project name.

Returns:
    Optional[PyProjectConfig]: A PyProjectConfig object containing:
        - project: Basic project information (name, version, dependencies, etc.)
        - tool_comfy: ComfyUI-specific configuration (publisher_id, models, etc.)
        Returns None if configuration extraction fails or if the provided file
        is not a Python file.

Notes:
    - If pyproject.toml is missing in a directory, creates a default config with folder name
    - If a .py file is provided, creates a default config with filename (without extension)
    - Returns None for non-Python files

Example:
    >>> from comfy_config import config_parser
    >>> # For directory
    >>> custom_node_dir = os.path.dirname(os.path.realpath(__file__))
    >>> project_config = config_parser.extract_node_configuration(custom_node_dir)
    >>> print(project_config.project.name)  # "my_custom_node" or name from pyproject.toml
    >>>
    >>> # For single-file Python node file
    >>> py_file_path = os.path.realpath(__file__) # "/path/to/my_node.py"
    >>> project_config = config_parser.extract_node_configuration(py_file_path)
    >>> print(project_config.project.name)  # "my_node"
"""
def extract_node_configuration(path) -> Optional[PyProjectConfig]:
    if os.path.isfile(path):
        file_path = Path(path)

        if file_path.suffix.lower() != '.py':
            return None

        project_name = file_path.stem
        project = ProjectConfig(name=project_name)
        comfy = ComfyConfig()
        return PyProjectConfig(project=project, tool_comfy=comfy)

    folder_name = os.path.basename(path)
    toml_path = Path(path) / "pyproject.toml"

    if not toml_path.exists():
        project = ProjectConfig(name=folder_name)
        comfy = ComfyConfig()
        return PyProjectConfig(project=project, tool_comfy=comfy)

    raw_settings = load_pyproject_settings(toml_path)

    project_data = raw_settings.project

    tool_data = raw_settings.tool
    comfy_data = tool_data.get("comfy", {}) if tool_data else {}

    return PyProjectConfig(project=project_data, tool_comfy=comfy_data)


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

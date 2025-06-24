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

def validate_and_extract_os_classifiers(classifiers: list) -> list:
    os_classifiers = [c for c in classifiers if c.startswith("Operating System :: ")]
    if not os_classifiers:
        return []

    os_values = [c[len("Operating System :: ") :] for c in os_classifiers]
    valid_os_prefixes = {"Microsoft", "POSIX", "MacOS", "OS Independent"}

    for os_value in os_values:
        if not any(os_value.startswith(prefix) for prefix in valid_os_prefixes):
            return []

    return os_values


def validate_and_extract_accelerator_classifiers(classifiers: list) -> list:
    accelerator_classifiers = [c for c in classifiers if c.startswith("Environment ::")]
    if not accelerator_classifiers:
        return []

    accelerator_values = [c[len("Environment :: ") :] for c in accelerator_classifiers]

    valid_accelerators = {
        "GPU :: NVIDIA CUDA",
        "GPU :: AMD ROCm",
        "GPU :: Intel Arc",
        "NPU :: Huawei Ascend",
        "GPU :: Apple Metal",
    }

    for accelerator_value in accelerator_values:
        if accelerator_value not in valid_accelerators:
            return []

    return accelerator_values


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

    dependencies = project_data.get("dependencies", [])
    supported_comfyui_frontend_version = ""
    for dep in dependencies:
        if isinstance(dep, str) and dep.startswith("comfyui-frontend-package"):
            supported_comfyui_frontend_version = dep.removeprefix("comfyui-frontend-package")
            break

    supported_comfyui_version = comfy_data.get("requires-comfyui", "")

    classifiers = project_data.get('classifiers', [])
    supported_os = validate_and_extract_os_classifiers(classifiers)
    supported_accelerators = validate_and_extract_accelerator_classifiers(classifiers)

    project_data['supported_os'] = supported_os
    project_data['supported_accelerators'] = supported_accelerators
    project_data['supported_comfyui_frontend_version'] = supported_comfyui_frontend_version
    project_data['supported_comfyui_version'] = supported_comfyui_version

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

from __future__ import annotations

import logging
import mimetypes
import os
import time
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import Optional, List, Literal

from ..cli_args_types import Configuration
from ..component_model.deprecation import _deprecate_method
from ..component_model.files import get_package_as_path
from ..component_model.folder_path_types import FolderNames, SaveImagePathTuple, ModelPaths
from ..component_model.folder_path_types import supported_pt_extensions, extension_mimetypes_cache
from ..component_model.module_property import create_module_properties
from ..component_model.platform_path import construct_path
from ..execution_context import current_execution_context

_module_properties = create_module_properties()

logger = logging.getLogger(__name__)


@_module_properties.getter
def _supported_pt_extensions() -> frozenset[str]:
    return supported_pt_extensions


@_module_properties.getter
def _extension_mimetypes_cache() -> dict[str, str]:
    return extension_mimetypes_cache


# todo: this needs to be wrapped in a context and configurable
@_module_properties.getter
def _base_path():
    return _folder_names_and_paths().base_paths[0]


def _resolve_path_with_compatibility(path: Path | str) -> PurePosixPath | Path:
    """
    Absolute posix style paths (aka, paths starting with `/`) are always returned as-is, otherwise this is resolved.
    :param path: a path or string to a path
    :return: the resolved path
    """
    if isinstance(path, PurePosixPath) and path.is_absolute():
        return path
    return Path(path).resolve()


def init_default_paths(folder_names_and_paths: FolderNames, configuration: Optional[Configuration] = None, create_all_directories=False):
    """
    Populates the folder names and paths object with the default, upstream model directories and custom_nodes directory.
    :param folder_names_and_paths: the object to populate with paths
    :param configuration: a configuration whose base_paths and other path settings will be used to set the values on this object
    :param create_all_directories: create all the possible directories by calling create_directories() after the object is populated
    :return:
    """
    from ..cmd.main_pre import args
    configuration = configuration or args
    base_paths = [Path(configuration.cwd) if configuration.cwd is not None else None] + configuration.base_paths
    base_paths = [path for path in base_paths if path is not None]
    if len(base_paths) == 0:
        base_paths = [Path(os.getcwd())]
    for base_path in base_paths:
        folder_names_and_paths.add_base_path(base_path)
    folder_names_and_paths.add(ModelPaths(["checkpoints"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["configs"], additional_absolute_directory_paths={get_package_as_path("comfy.configs")}, supported_extensions={".yaml"}))
    folder_names_and_paths.add(ModelPaths(["vae"], supported_extensions={".yaml"}))
    folder_names_and_paths.add(ModelPaths(["clip"], supported_extensions={".yaml"}))
    folder_names_and_paths.add(ModelPaths(["loras"], supported_extensions={".yaml"}))
    folder_names_and_paths.add(ModelPaths(["diffusion_models", "unet"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["clip_vision"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["style_models"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["embeddings"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["diffusers"], supported_extensions=set()))
    folder_names_and_paths.add(ModelPaths(["vae_approx"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["controlnet", "t2i_adapter"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["gligen"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["upscale_models"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["custom_nodes"], folder_name_base_path_subdir=construct_path(""), supported_extensions=set()))
    folder_names_and_paths.add(ModelPaths(["hypernetworks"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["photomaker"], supported_extensions=set(supported_pt_extensions)))
    folder_names_and_paths.add(ModelPaths(["classifiers"], supported_extensions=set()))
    folder_names_and_paths.add(ModelPaths(["huggingface"], supported_extensions=set()))
    hf_cache_paths = ModelPaths(["huggingface_cache"], supported_extensions=set())
    # TODO: explore if there is a better way to do this
    if "HF_HUB_CACHE" in os.environ:
        hf_cache_paths.additional_absolute_directory_paths.add(os.environ.get("HF_HUB_CACHE"))
    folder_names_and_paths.add(hf_cache_paths)
    if create_all_directories:
        create_directories(folder_names_and_paths)

    if configuration.output_directory is not None:
        folder_names_and_paths.application_paths.output_directory = Path(configuration.output_directory)

    if configuration.input_directory is not None:
        folder_names_and_paths.application_paths.input_directory = Path(configuration.input_directory)

    if configuration.temp_directory is not None:
        folder_names_and_paths.application_paths.temp_directory = Path(configuration.temp_directory)

    if configuration.user_directory is not None:
        folder_names_and_paths.application_paths.user_directory = Path(configuration.user_directory)


@_module_properties.getter
def _folder_names_and_paths():
    return current_execution_context().folder_names_and_paths


@_module_properties.getter
def _models_dir():
    return str(Path(current_execution_context().folder_names_and_paths.base_paths[0]) / construct_path("models"))


@_module_properties.getter
def _user_directory() -> str:
    return str(_resolve_path_with_compatibility(current_execution_context().folder_names_and_paths.application_paths.user_directory))


@_module_properties.getter
def _temp_directory() -> str:
    return str(_resolve_path_with_compatibility(current_execution_context().folder_names_and_paths.application_paths.temp_directory))


@_module_properties.getter
def _input_directory() -> str:
    return str(_resolve_path_with_compatibility(current_execution_context().folder_names_and_paths.application_paths.input_directory))


@_module_properties.getter
def _output_directory() -> str:
    return str(_resolve_path_with_compatibility(current_execution_context().folder_names_and_paths.application_paths.output_directory))


@_deprecate_method(version="0.2.3", message="Mapping of previous folder names is already done by other mechanisms.")
def map_legacy(folder_name: str) -> str:
    legacy = {"unet": "diffusion_models"}
    return legacy.get(folder_name, folder_name)


def set_output_directory(output_dir: str | Path):
    _folder_names_and_paths().application_paths.output_directory = construct_path(output_dir)


def set_temp_directory(temp_dir: str | Path):
    _folder_names_and_paths().application_paths.temp_directory = construct_path(temp_dir)


def set_input_directory(input_dir: str | Path):
    _folder_names_and_paths().application_paths.input_directory = construct_path(input_dir)


def get_output_directory() -> str:
    return str(_resolve_path_with_compatibility(_folder_names_and_paths().application_paths.output_directory))


def get_temp_directory() -> str:
    return str(_resolve_path_with_compatibility(_folder_names_and_paths().application_paths.temp_directory))


def get_input_directory() -> str:
    return str(_resolve_path_with_compatibility(_folder_names_and_paths().application_paths.input_directory))


def get_user_directory() -> str:
    return str(_resolve_path_with_compatibility(_folder_names_and_paths().application_paths.user_directory))


def set_user_directory(user_dir: str | Path) -> None:
    _folder_names_and_paths().application_paths.user_directory = construct_path(user_dir)


# NOTE: used in http server so don't put folders that should not be accessed remotely
def get_directory_by_type(type_name) -> str | None:
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None


# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format
# otherwise use default_path as base_dir
def annotated_filepath(name: str) -> tuple[str, str | None]:
    if name.endswith("[output]"):
        base_dir = get_output_directory()
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()
        name = name[:-7]
    else:
        return name, None

    return name, base_dir


def get_annotated_filepath(name, default_dir=None) -> str:
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path

    return os.path.join(base_dir, name)


def exists_annotated_filepath(name):
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        base_dir = get_input_directory()  # fallback path

    filepath = os.path.join(base_dir, name)
    return os.path.exists(filepath)


def add_model_folder_path(folder_name, full_folder_path: Optional[str] = None, extensions: Optional[set[str] | frozenset[str]] = None, is_default: bool = False) -> str:
    """
    Registers a model path for the given canonical name.
    :param folder_name: the folder name
    :param full_folder_path: When none, defaults to os.path.join(models_dir, folder_name) aka the folder as
    a subpath to the default models directory
    :param extensions: supported file extensions
    :return: the folder path
    """
    folder_names_and_paths = _folder_names_and_paths()
    if full_folder_path is None:
        # todo: this should use the subdir patter

        full_folder_path = os.path.join(_models_dir(), folder_name)

    folder_path = folder_names_and_paths[folder_name]
    if full_folder_path not in folder_path.paths:
        if is_default:
            folder_path.paths.insert(0, full_folder_path)
        else:
            folder_path.paths.append(full_folder_path)

    if extensions is not None:
        folder_path.supported_extensions |= extensions

    return full_folder_path


def get_folder_paths(folder_name) -> List[str]:
    return [path for path in _folder_names_and_paths()[folder_name].paths]


@_deprecate_method(version="1.0.0", message="Use os.scandir instead.")
def recursive_search(directory, excluded_dir_names=None) -> tuple[list[str], dict[str, float]]:
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs = {}

    # Attempt to add the initial directory to dirs with error handling
    try:
        dirs[directory] = os.path.getmtime(directory)
    except FileNotFoundError:
        logger.warning(f"Warning: Unable to access {directory}. Skipping this path.")

    logger.debug("recursive file list on directory {}".format(directory))
    dirpath: str
    subdirs: list[str]
    filenames: list[str]

    for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            try:
                relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
                result.append(relative_path)
            except:
                logger.warning(f"Warning: Unable to access {file_name}. Skipping this file.")
                continue

        for d in subdirs:
            path: str = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                logger.warning(f"Warning: Unable to access {path}. Skipping this path.")
                continue
    logger.debug("found {} files".format(len(result)))
    return result, dirs


def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions or len(extensions) == 0, files)))


def get_full_path(folder_name, filename) -> Optional[str | bytes | os.PathLike]:
    """
    Gets the path to a filename inside a folder.

    :param folder_name:
    :param filename:
    :return:
    """
    path = _folder_names_and_paths().first_existing_or_none(folder_name, construct_path(filename))
    return str(path) if path is not None else None


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    full_path = get_full_path(folder_name, filename)
    if full_path is None:
        # todo: probably shouldn't say model
        raise FileNotFoundError(f"Model in folder '{folder_name}' with filename '{filename}' not found.")
    return full_path


def get_filename_list(folder_name: str) -> list[str]:
    return [str(path) for path in _folder_names_and_paths().file_paths(folder_name=folder_name, relative=True)]


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0) -> SaveImagePathTuple:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1:].split('_')[0])
        except:
            digits = 0
        return digits, prefix

    def compute_vars(input: str, image_width: int, image_height: int) -> str:
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        now = time.localtime()
        input = input.replace("%year%", str(now.tm_year))
        input = input.replace("%month%", str(now.tm_mon).zfill(2))
        input = input.replace("%day%", str(now.tm_mday).zfill(2))
        input = input.replace("%hour%", str(now.tm_hour).zfill(2))
        input = input.replace("%minute%", str(now.tm_min).zfill(2))
        input = input.replace("%second%", str(now.tm_sec).zfill(2))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = str(os.path.join(output_dir, subfolder))

    try:
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return SaveImagePathTuple(full_output_folder, filename, counter, subfolder, filename_prefix)


def create_directories(paths: FolderNames | None):
    # all configured paths should be created
    paths = paths or _folder_names_and_paths()
    for folder_path_spec in paths.values():
        for path in folder_path_spec.paths:
            os.makedirs(path, exist_ok=True)
    for path in paths.application_paths:
        path.mkdir(exist_ok=True)


@_deprecate_method(version="0.2.3", message="Caching has been removed.")
def invalidate_cache(folder_name):
    pass


def filter_files_content_types(files: list[str], content_types: list[Literal["image", "video", "audio"]]) -> list[str]:
    """
    Example:
        files = os.listdir(folder_paths.get_input_directory())
        filter_files_content_types(files, ["image", "audio", "video"])
    """
    extension_mimetypes_cache = _extension_mimetypes_cache()
    result = []
    for file in files:
        extension = file.split('.')[-1]
        if extension not in extension_mimetypes_cache:
            mime_type, _ = mimetypes.guess_type(file, strict=False)
            if not mime_type:
                continue
            content_type = mime_type.split('/')[0]
            extension_mimetypes_cache[extension] = content_type
        else:
            content_type = extension_mimetypes_cache[extension]

        if content_type in content_types:
            result.append(file)
    return result


@_module_properties.getter
def _cache_helper():
    return nullcontext()


# todo: can this be done side effect free?
init_default_paths(_folder_names_and_paths())

__all__ = [
    "supported_pt_extensions",
    "extension_mimetypes_cache",
    "base_path",
    "folder_names_and_paths",
    "models_dir",
    "user_directory",
    "output_directory",
    "temp_directory",
    "input_directory",

    # Public functions
    "init_default_paths",
    "map_legacy",
    "set_output_directory",
    "set_temp_directory",
    "set_input_directory",
    "get_output_directory",
    "get_temp_directory",
    "get_input_directory",
    "get_user_directory",
    "set_user_directory",
    "get_directory_by_type",
    "annotated_filepath",
    "get_annotated_filepath",
    "exists_annotated_filepath",
    "add_model_folder_path",
    "get_folder_paths",
    "recursive_search",
    "filter_files_extensions",
    "get_full_path",
    "get_full_path_or_raise",
    "get_filename_list",
    "get_save_image_path",
    "create_directories",
    "invalidate_cache",
    "filter_files_content_types"
]

from __future__ import annotations

import logging
import mimetypes
import os
import time
from typing import Optional, List, Final, Literal

from .folder_paths_pre import get_base_path
from ..component_model.files import get_package_as_path
from ..component_model.folder_path_types import FolderPathsTuple, FolderNames, SaveImagePathResponse
from ..component_model.folder_path_types import extension_mimetypes_cache as _extension_mimetypes_cache
from ..component_model.folder_path_types import supported_pt_extensions as _supported_pt_extensions
from ..component_model.module_property import module_property

supported_pt_extensions: Final[frozenset[str]] = _supported_pt_extensions
extension_mimetypes_cache: Final[dict[str, str]] = _extension_mimetypes_cache


# todo: this needs to be wrapped in a context and configurable
@module_property
def _base_path():
    return get_base_path()


models_dir = os.path.join(get_base_path(), "models")
folder_names_and_paths = FolderNames(models_dir)
folder_names_and_paths["checkpoints"] = FolderPathsTuple("checkpoints", [os.path.join(models_dir, "checkpoints")], set(supported_pt_extensions))
folder_names_and_paths["configs"] = FolderPathsTuple("configs", [os.path.join(models_dir, "configs"), get_package_as_path("comfy.configs")], {".yaml"})
folder_names_and_paths["loras"] = FolderPathsTuple("loras", [os.path.join(models_dir, "loras")], set(supported_pt_extensions))
folder_names_and_paths["vae"] = FolderPathsTuple("vae", [os.path.join(models_dir, "vae")], set(supported_pt_extensions))
folder_names_and_paths["clip"] = FolderPathsTuple("clip", [os.path.join(models_dir, "clip")], set(supported_pt_extensions))
folder_names_and_paths["unet"] = folder_names_and_paths["diffusion_models"] = FolderPathsTuple("diffusion_models", [os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")], set(supported_pt_extensions))
folder_names_and_paths["clip_vision"] = FolderPathsTuple("clip_vision", [os.path.join(models_dir, "clip_vision")], set(supported_pt_extensions))
folder_names_and_paths["style_models"] = FolderPathsTuple("style_models", [os.path.join(models_dir, "style_models")], set(supported_pt_extensions))
folder_names_and_paths["embeddings"] = FolderPathsTuple("embeddings", [os.path.join(models_dir, "embeddings")], set(supported_pt_extensions))
folder_names_and_paths["diffusers"] = FolderPathsTuple("diffusers", [os.path.join(models_dir, "diffusers")], {"folder"})
folder_names_and_paths["vae_approx"] = FolderPathsTuple("vae_approx", [os.path.join(models_dir, "vae_approx")], set(supported_pt_extensions))
folder_names_and_paths["controlnet"] = FolderPathsTuple("controlnet", [os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], set(supported_pt_extensions))
folder_names_and_paths["gligen"] = FolderPathsTuple("gligen", [os.path.join(models_dir, "gligen")], set(supported_pt_extensions))
folder_names_and_paths["upscale_models"] = FolderPathsTuple("upscale_models", [os.path.join(models_dir, "upscale_models")], set(supported_pt_extensions))
folder_names_and_paths["custom_nodes"] = FolderPathsTuple("custom_nodes", [os.path.join(get_base_path(), "custom_nodes")], set())
folder_names_and_paths["hypernetworks"] = FolderPathsTuple("hypernetworks", [os.path.join(models_dir, "hypernetworks")], set(supported_pt_extensions))
folder_names_and_paths["photomaker"] = FolderPathsTuple("photomaker", [os.path.join(models_dir, "photomaker")], set(supported_pt_extensions))
folder_names_and_paths["classifiers"] = FolderPathsTuple("classifiers", [os.path.join(models_dir, "classifiers")], {""})
folder_names_and_paths["huggingface"] = FolderPathsTuple("huggingface", [os.path.join(models_dir, "huggingface")], {""})
folder_names_and_paths["huggingface_cache"] = FolderPathsTuple("huggingface_cache", [os.path.join(models_dir, "huggingface_cache")], {""})

output_directory = os.path.join(get_base_path(), "output")
temp_directory = os.path.join(get_base_path(), "temp")
input_directory = os.path.join(get_base_path(), "input")
user_directory = os.path.join(get_base_path(), "user")

filename_list_cache = {}


class CacheHelper:
    """
    Helper class for managing file list cache data.
    """

    def __init__(self):
        self.cache: dict[str, tuple[list[str], dict[str, float], float]] = {}
        self.active = False

    def get(self, key: str, default=None) -> tuple[list[str], dict[str, float], float]:
        if not self.active:
            return default
        return self.cache.get(key, default)

    def set(self, key: str, value: tuple[list[str], dict[str, float], float]) -> None:
        if self.active:
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self.clear()


cache_helper = CacheHelper()


def map_legacy(folder_name: str) -> str:
    legacy = {"unet": "diffusion_models"}
    return legacy.get(folder_name, folder_name)


if not os.path.exists(input_directory):
    try:
        os.makedirs(input_directory)
    except:
        logging.error("Failed to create input directory")


def set_output_directory(output_dir):
    global output_directory
    output_directory = output_dir


def set_temp_directory(temp_dir):
    global temp_directory
    temp_directory = temp_dir


def set_input_directory(input_dir):
    global input_directory
    input_directory = input_dir


def get_output_directory():
    global output_directory
    return output_directory


def get_temp_directory():
    global temp_directory
    return temp_directory


def get_input_directory():
    global input_directory
    return input_directory


def get_user_directory() -> str:
    return user_directory


def set_user_directory(user_dir: str) -> None:
    global user_directory
    user_directory = user_dir


# NOTE: used in http server so don't put folders that should not be accessed remotely
def get_directory_by_type(type_name):
    if type_name == "output":
        return get_output_directory()
    if type_name == "temp":
        return get_temp_directory()
    if type_name == "input":
        return get_input_directory()
    return None


# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format
# otherwise use default_path as base_dir
def annotated_filepath(name):
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


def get_annotated_filepath(name, default_dir=None):
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


def add_model_folder_path(folder_name, full_folder_path: Optional[str] = None, extensions: Optional[set[str]] = None, is_default: bool = False) -> str:
    """
    Registers a model path for the given canonical name.
    :param folder_name: the folder name
    :param full_folder_path: When none, defaults to os.path.join(models_dir, folder_name) aka the folder as
    a subpath to the default models directory
    :param extensions: supported file extensions
    :return: the folder path
    """
    global folder_names_and_paths
    if full_folder_path is None:
        full_folder_path = os.path.join(models_dir, folder_name)

    folder_path = folder_names_and_paths[folder_name]
    if full_folder_path not in folder_path.paths:
        if is_default:
            folder_path.paths.insert(0, full_folder_path)
        else:
            folder_path.paths.append(full_folder_path)

    if extensions is not None:
        folder_path.supported_extensions |= extensions

    invalidate_cache(folder_name)
    return full_folder_path


def get_folder_paths(folder_name) -> List[str]:
    return folder_names_and_paths[folder_name].paths[:]


def recursive_search(directory, excluded_dir_names=None):
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
        logging.warning(f"Warning: Unable to access {directory}. Skipping this path.")

    for dirpath, subdirs, filenames in os.walk(directory, followlinks=True, topdown=True):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
            result.append(relative_path)

        for d in subdirs:
            path = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                logging.warning(f"Warning: Unable to access {path}. Skipping this path.")
                continue
    return result, dirs


def filter_files_extensions(files, extensions):
    return sorted(list(filter(lambda a: os.path.splitext(a)[-1].lower() in extensions or len(extensions) == 0, files)))


def get_full_path(folder_name, filename) -> Optional[str | bytes | os.PathLike]:
    """
    Gets the path to a filename inside a folder.

    Works with untrusted filenames.
    :param folder_name:
    :param filename:
    :return:
    """
    global folder_names_and_paths
    folders = folder_names_and_paths[folder_name].paths
    filename_split = os.path.split(filename)

    trusted_paths = []
    for folder in folders:
        folder_split = os.path.split(folder)
        abs_file_path = os.path.abspath(os.path.join(*folder_split, *filename_split))
        abs_folder_path = os.path.abspath(folder)
        if os.path.commonpath([abs_file_path, abs_folder_path]) == abs_folder_path:
            trusted_paths.append(abs_file_path)
        else:
            logging.error(f"attempted to access untrusted path {abs_file_path} in {folder_name} for filename {filename}")

    for trusted_path in trusted_paths:
        if os.path.isfile(trusted_path):
            return trusted_path

    return None


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    full_path = get_full_path(folder_name, filename)
    if full_path is None:
        raise FileNotFoundError(f"Model in folder '{folder_name}' with filename '{filename}' not found.")
    return full_path


def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}

    return sorted(list(output_list)), output_folders, time.perf_counter()


def cached_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float] | None:
    strong_cache = cache_helper.get(folder_name)
    if strong_cache is not None:
        return strong_cache

    global filename_list_cache
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:
                return None

    return out


def get_filename_list(folder_name: str) -> list[str]:
    folder_name = map_legacy(folder_name)
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    cache_helper.set(folder_name, out)
    return list(out[0])


def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
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

    if str(os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))) != str(output_dir):
        err = f"""**** ERROR: Saving image outside the output folder is not allowed.
                  full_output_folder: {os.path.abspath(full_output_folder)}
                        output_dir: {output_dir}
                        commonpath: {os.path.commonpath((output_dir, os.path.abspath(full_output_folder)))}"""
        logging.error(err)
        raise Exception(err)

    try:
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return SaveImagePathResponse(full_output_folder, filename, counter, subfolder, filename_prefix)


def create_directories():
    # all configured paths should be created
    for folder_path_spec in folder_names_and_paths.values():
        for path in folder_path_spec.paths:
            os.makedirs(path, exist_ok=True)
    for path in (temp_directory, input_directory, output_directory, user_directory):
        os.makedirs(path, exist_ok=True)


def invalidate_cache(folder_name):
    global filename_list_cache
    filename_list_cache.pop(folder_name, None)


def filter_files_content_types(files: list[str], content_types: list[Literal["image", "video", "audio"]]) -> list[str]:
    """
    Example:
        files = os.listdir(folder_paths.get_input_directory())
        filter_files_content_types(files, ["image", "audio", "video"])
    """
    global extension_mimetypes_cache
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

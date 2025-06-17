import collections
import os
from pathlib import Path
from typing import Optional, List, Literal, Tuple, Union, Dict

from ..cli_args_types import Configuration
from ..component_model.folder_path_types import FolderNames, SaveImagePathTuple

# Variables
base_path: str
folder_names_and_paths: FolderNames
models_dir: str
user_directory: str
output_directory: str
temp_directory: str
input_directory: str
supported_pt_extensions: set[str]


# Functions
def init_default_paths(folder_names_and_paths: FolderNames, configuration: Optional[Configuration] = None, create_all_directories: bool = ..., replace_existing: bool = ..., base_paths_from_configuration: bool = ...): ...


def map_legacy(folder_name: str) -> str: ...


def set_output_directory(output_dir: Union[str, Path]) -> None: ...


def set_temp_directory(temp_dir: Union[str, Path]) -> None: ...


def set_input_directory(input_dir: Union[str, Path]) -> None: ...


def get_output_directory() -> str: ...


def get_temp_directory() -> str: ...


def get_input_directory() -> str: ...


def get_user_directory() -> str: ...


def set_user_directory(user_dir: Union[str, Path]) -> None: ...


def get_directory_by_type(type_name: str) -> Optional[str]: ...


def annotated_filepath(name: str) -> Tuple[str, Optional[str]]: ...


def get_annotated_filepath(name: str, default_dir: Optional[str] = ...) -> str: ...


def exists_annotated_filepath(name: str) -> bool: ...


def add_model_folder_path(
        folder_name: str,
        full_folder_path: Optional[str] = ...,
        extensions: Optional[Union[set[str], frozenset[str]]] = ...,
        is_default: bool = ...,
        folder_names_and_paths: Optional[FolderNames] = ...,
) -> str: ...


def get_folder_paths(folder_name: str) -> List[str]: ...


def recursive_search(
        directory: str,
        excluded_dir_names: Optional[List[str]] = ...
) -> Tuple[List[str], Dict[str, float]]: ...


def filter_files_extensions(files: collections.abc.Collection[str], extensions: collections.abc.Collection[str]) -> List[str]: ...


def get_full_path(folder_name: str, filename: str) -> Optional[Union[str, bytes, os.PathLike]]: ...


def get_full_path_or_raise(folder_name: str, filename: str) -> str: ...


def get_filename_list(folder_name: str) -> List[str]: ...


def get_save_image_path(
        filename_prefix: str,
        output_dir: str,
        image_width: int = 0,
        image_height: int = 0
) -> SaveImagePathTuple: ...


def create_directories(paths: Optional[FolderNames] = ...) -> None: ...


def invalidate_cache(folder_name: str) -> None: ...


def filter_files_content_types(files: List[str], content_types: List[Literal["image", "video", "audio", "model"]]) -> List[str]: ...


def get_input_subfolders() -> list[str]: ...

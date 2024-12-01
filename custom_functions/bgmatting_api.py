import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        # path = os.getcwd()
        path = os.path.dirname(__file__)

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()

from nodes import LoadImage, SaveImage, NODE_CLASS_MAPPINGS, init_extra_nodes


def bg_matting(image, abs_path=True, is_ndarray=False):
    init_extra_nodes(True)
    with torch.inference_mode():
        loadimage = LoadImage()
        loadimage_12 = loadimage.load_image(image=image, abs_path=abs_path, is_ndarray=is_ndarray)

        bria_rmbg_modelloader_zho = NODE_CLASS_MAPPINGS["BRIA_RMBG_ModelLoader_Zho"]()
        bria_rmbg_modelloader_zho_19 = bria_rmbg_modelloader_zho.load_model()

        bria_rmbg_zho = NODE_CLASS_MAPPINGS["BRIA_RMBG_Zho"]()
        masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
        saveimage = SaveImage()

        bria_rmbg_zho_20 = bria_rmbg_zho.remove_background(
            rmbgmodel=get_value_at_index(bria_rmbg_modelloader_zho_19, 0),
            image=get_value_at_index(loadimage_12, 0),
        )

        # masktoimage_21 = masktoimage.mask_to_image(
        #     mask=get_value_at_index(bria_rmbg_zho_20, 1)
        # )

        outimg = get_value_at_index(bria_rmbg_zho_20, 0) * 255.

        return outimg.cpu().numpy().astype(np.uint8)


if __name__ == "__main__":
    pass
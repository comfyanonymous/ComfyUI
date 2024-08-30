import os
import sys
from typing import Sequence, Mapping, Any, Union
import torch


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
        path = os.getcwd()

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


from nodes import NODE_CLASS_MAPPINGS, LoadImage, init_extra_nodes


def image_text_matting(image_path, text, abs_path=True):
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()
    init_extra_nodes(True)

    with torch.inference_mode():
        sammodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "SAMModelLoader (segment anything)"
        ]()
        sammodelloader_segment_anything_1 = sammodelloader_segment_anything.main(
            model_name="sam_hq_vit_h (2.57GB)"
        )

        groundingdinomodelloader_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoModelLoader (segment anything)"
        ]()
        groundingdinomodelloader_segment_anything_2 = (
            groundingdinomodelloader_segment_anything.main(
                model_name="GroundingDINO_SwinT_OGC (694MB)"
            )
        )

        mattingmodelloader = NODE_CLASS_MAPPINGS["MattingModelLoader"]()
        mattingmodelloader_8 = mattingmodelloader.main(
            model_name="vitmatte_small (103 MB)"
        )

        loadimage = LoadImage()
        loadimage_12 = loadimage.load_image(image=image_path, abs_path=abs_path)

        groundingdinosamsegment_segment_anything = NODE_CLASS_MAPPINGS[
            "GroundingDinoSAMSegment (segment anything)"
        ]()
        createtrimap = NODE_CLASS_MAPPINGS["CreateTrimap"]()
        applymatting = NODE_CLASS_MAPPINGS["ApplyMatting"]()

        groundingdinosamsegment_segment_anything_3 = (
            groundingdinosamsegment_segment_anything.main(
                prompt=text,
                threshold=0.3,
                sam_model=get_value_at_index(sammodelloader_segment_anything_1, 0),
                grounding_dino_model=get_value_at_index(
                    groundingdinomodelloader_segment_anything_2, 0
                ),
                image=get_value_at_index(loadimage_12, 0),
            )
        )

        createtrimap_11 = createtrimap.main(
            kernel_size=20.86,
            mask=get_value_at_index(groundingdinosamsegment_segment_anything_3, 1),
        )


        applymatting_9 = applymatting.main(
            matting_model=get_value_at_index(mattingmodelloader_8, 0),
            matting_preprocessor=get_value_at_index(mattingmodelloader_8, 1),
            image=get_value_at_index(loadimage_12, 0),
            trimap=get_value_at_index(createtrimap_11, 0),
        )

        output_image = get_value_at_index(applymatting_9, 1)
        return output_image



if __name__ == "__main__":
    pass
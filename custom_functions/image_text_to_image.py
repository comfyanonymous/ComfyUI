import os
import random
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

add_comfyui_directory_to_sys_path()
add_extra_model_paths()


from nodes import (
    NODE_CLASS_MAPPINGS,
    SaveImage,
    CLIPTextEncode,
    LoadImage,
    CheckpointLoaderSimple,
    KSampler,
    VAEDecode,
    VAEEncode,
)


def image_text_to_image(image_path, pos_text, neg_text=r'watermark, text'):

    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_14 = checkpointloadersimple.load_checkpoint(
            ckpt_name="v1-5-pruned-emaonly.ckpt"
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text=pos_text,
            clip=get_value_at_index(checkpointloadersimple_14, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text=neg_text,
            clip=get_value_at_index(checkpointloadersimple_14, 1),
        )

        loadimage = LoadImage()
        loadimage_10 = loadimage.load_image(image=image_path, abs_path=True)

        vaeencode = VAEEncode()
        vaeencode_12 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_10, 0),
            vae=get_value_at_index(checkpointloadersimple_14, 2),
        )

        ksampler = KSampler()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        ksampler_3 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=20,
            cfg=8,
            sampler_name="dpmpp_2m",
            scheduler="normal",
            denoise=0.8700000000000001,
            model=get_value_at_index(checkpointloadersimple_14, 0),
            positive=get_value_at_index(cliptextencode_6, 0),
            negative=get_value_at_index(cliptextencode_7, 0),
            latent_image=get_value_at_index(vaeencode_12, 0),
        )

        vaedecode_8 = vaedecode.decode(
            samples=get_value_at_index(ksampler_3, 0),
            vae=get_value_at_index(checkpointloadersimple_14, 2),
        )

        output_image = get_value_at_index(vaedecode_8, 0)
        # print("output", output_image.shape, torch.max(output_image), torch.min(output_image))

        return output_image

if __name__ == "__main__":
    image_path = r'D:\VisualForge\ComfyUI\input\example.png'
    pos_text = r'photograph of victorian woman with wings, sky clouds, meadow grass'

    out_image = image_text_to_image(image_path, pos_text)
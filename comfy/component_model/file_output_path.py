from pathlib import Path
from typing import Literal, Optional

from ..cmd import folder_paths


def file_output_path(filename: str, type: Literal["input", "output", "temp"] = "output",
                     subfolder: Optional[str] = None) -> str:
    """
    Takes the contents of a file output node and returns an actual path to the file referenced in it.

    This is adapted from the /view code
    :param filename:
    :param type:
    :param subfolder:
    :return:
    """
    filename, output_dir = folder_paths.annotated_filepath(str(filename))

    if output_dir is None:
        output_dir = folder_paths.get_directory_by_type(type)
    if output_dir is None:
        raise ValueError(f"no such output directory because invalid type specified (type={type})")
    output_dir = Path(output_dir)
    # seems to misbehave
    subfolder = subfolder or ""
    subfolder = subfolder.replace("\\", "/")
    subfolder = Path(subfolder)
    try:
        relative_to = (output_dir / subfolder / filename).relative_to(output_dir)
    except ValueError:
        raise PermissionError(f"{output_dir / subfolder / filename} is not a subpath of {output_dir}")
    return str((output_dir / relative_to).resolve(strict=True))

import os
from typing import Literal, Optional
from pathlib import Path

from ..cmd import folder_paths


def _is_strictly_below_root(path: Path) -> bool:
    resolved_path = path.resolve()
    return ".." not in resolved_path.parts and resolved_path.is_absolute()


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
    if not _is_strictly_below_root(Path(filename)):
        raise PermissionError("insecure")

    if output_dir is None:
        output_dir = folder_paths.get_directory_by_type(type)
    if output_dir is None:
        raise ValueError(f"no such output directory because invalid type specified (type={type})")
    if subfolder is not None:
        full_output_dir = os.path.join(output_dir, subfolder)
        if os.path.commonpath((os.path.abspath(full_output_dir), output_dir)) != output_dir:
            raise PermissionError("insecure")
        output_dir = full_output_dir

    filename = os.path.basename(filename)
    file = os.path.join(output_dir, filename)
    return file

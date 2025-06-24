import asyncio
import warnings
from pathlib import Path

from comfy.component_model.folder_path_types import FolderNames

if __name__ == "__main__":
    warnings.warn("main.py is deprecated. Start comfyui by installing the package through the instructions in the README, not by cloning the repository.", DeprecationWarning)
    this_file_parent_dir = Path(__file__).parent
    from comfy.cmd.main import _start_comfyui
    from comfy.cmd.folder_paths import folder_names_and_paths  # type: FolderNames
    fn: FolderNames = folder_names_and_paths
    fn.base_paths.clear()
    fn.base_paths.append(this_file_parent_dir)

    asyncio.run(_start_comfyui(from_script_dir=this_file_parent_dir))

import asyncio
import warnings
from pathlib import Path

if __name__ == "__main__":
    from comfy.cmd.folder_paths_pre import set_base_path

    warnings.warn("main.py is deprecated. Start comfyui by installing the package through the instructions in the README, not by cloning the repository.", DeprecationWarning)
    this_file_parent_dir = Path(__file__).parent
    set_base_path(str(this_file_parent_dir))

    from comfy.cmd.main import main

    asyncio.run(main(from_script_dir=this_file_parent_dir))

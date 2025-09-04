import inspect
import logging
import os
import shutil
import sys
from typing import Optional

# mitigations for comfyui workspace chaos
logger = logging.getLogger(__name__)


def auto_patch_workspace_and_restart(workspace_dir: Optional[str] = None, skip_restart=False):
    """
    Detects a specific workspace structure, creates necessary __init__.py files
    to make directories proper Python packages, and then restarts the application.
    If the workspace is a Git repository, it locally ignores the created files.
    """
    if workspace_dir is None:
        try:
            main_frame = next(f for f in reversed(inspect.stack()) if f.frame.f_globals.get('__name__') == '__main__')
            main_file_path = main_frame.filename
            workspace_dir = os.path.dirname(os.path.abspath(main_file_path))
        except (StopIteration, AttributeError):
            logger.debug("workspace_dir was none and comfyui's main.py is not being run, skipping injection of __init__.py files")
            return

    if not os.path.isfile(os.path.join(workspace_dir, 'nodes.py')):
        logger.debug(f"did not find a nodes.py inside the workspace_dir {workspace_dir}, skipping injection of __init__.py files")
        return

    git_is_available = False
    if shutil.which('git') and os.path.isdir(os.path.join(workspace_dir, '.git')):
        git_is_available = True

    target_base_dirs = [
        'comfy',
        'comfy_extras',
        'comfy_execution',
        'comfy_api',
        'comfy_config'
    ]

    patched_any_file = False
    files_to_ignore = []

    for dir_name in target_base_dirs:
        start_dir = os.path.join(workspace_dir, dir_name)
        if not os.path.isdir(start_dir):
            continue

        dirs_with_py_files = set()
        for dirpath, _, filenames in os.walk(start_dir):
            if any(fname.endswith('.py') for fname in filenames):
                dirs_with_py_files.add(dirpath)

        if not dirs_with_py_files:
            continue

        dirs_to_initialize = set()
        for dirpath in dirs_with_py_files:
            parent = dirpath
            while len(parent) >= len(start_dir):
                dirs_to_initialize.add(parent)
                new_parent = os.path.dirname(parent)
                if new_parent == parent:
                    break
                parent = new_parent

        for dirpath in sorted(list(dirs_to_initialize)):
            init_py_path = os.path.join(dirpath, '__init__.py')
            if not os.path.exists(init_py_path):
                logger.debug(f"initializing package: {dirpath}")
                try:
                    with open(init_py_path, 'w') as f:
                        pass
                    patched_any_file = True
                    files_to_ignore.append(init_py_path)
                except OSError as e:
                    logger.debug(f"could not create {init_py_path}. Error: {e}")

    if git_is_available and files_to_ignore:
        try:
            exclude_file = os.path.join(workspace_dir, '.git', 'info', 'exclude')
            os.makedirs(os.path.dirname(exclude_file), exist_ok=True)

            existing_lines = set()
            if os.path.exists(exclude_file):
                with open(exclude_file, 'r') as f_read:
                    existing_lines = set(line.strip() for line in f_read)

            with open(exclude_file, 'a') as f_append:
                for init_py_path in files_to_ignore:
                    relative_path = os.path.relpath(init_py_path, workspace_dir).replace(os.sep, '/')
                    if relative_path not in existing_lines:
                        f_append.write(f"\n{relative_path}")
                        existing_lines.add(relative_path)
                        logger.debug(f"ignoring via .git/info/exclude: {relative_path}")

        except Exception as e:
            logger.debug(f"could not update Git exclude file. Error: {e}")

    if not skip_restart and patched_any_file:
        logger.info("Found and initialized Python package directories in your workspace. This is a one-time operation to enable proper imports. Now restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

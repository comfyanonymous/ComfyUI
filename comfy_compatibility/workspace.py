import inspect
import logging
import os
import shutil
import sys

# mitigations for comfyui workspace chaos
logger = logging.getLogger(__name__)


def auto_patch_workspace_and_restart():
    """
    Detects a specific workspace structure, creates necessary __init__.py files
    to make directories proper Python packages, and then restarts the application.
    If the workspace is a Git repository, it locally ignores the created files.
    """
    try:
        main_frame = next(f for f in reversed(inspect.stack()) if f.frame.f_globals.get('__name__') == '__main__')
        main_file_path = main_frame.filename
        workspace_dir = os.path.dirname(os.path.abspath(main_file_path))
    except (StopIteration, AttributeError):
        return

    if not os.path.isfile(os.path.join(workspace_dir, 'nodes.py')):
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

    for dir_name in target_base_dirs:
        start_dir = os.path.join(workspace_dir, dir_name)
        if not os.path.isdir(start_dir):
            continue

        for dirpath, _, filenames in os.walk(start_dir):
            init_py_path = os.path.join(dirpath, '__init__.py')

            if os.path.exists(init_py_path):
                continue

            if any(fname.endswith('.py') for fname in filenames):
                logger.debug(f"  Initializing package: {dirpath}")
                try:
                    with open(init_py_path, 'w') as f:
                        pass
                    patched_any_file = True

                    if git_is_available:
                        try:
                            relative_path = os.path.relpath(init_py_path, workspace_dir).replace(os.sep, '/')

                            exclude_file = os.path.join(workspace_dir, '.git', 'info', 'exclude')
                            os.makedirs(os.path.dirname(exclude_file), exist_ok=True)

                            content = ""
                            if os.path.exists(exclude_file):
                                with open(exclude_file, 'r') as f_read:
                                    content = f_read.read()

                            if relative_path not in content.splitlines():
                                with open(exclude_file, 'a') as f_append:
                                    f_append.write(f"\n{relative_path}")
                                logger.debug(f"  Ignoring via .git/info/exclude: {relative_path}")

                        except Exception as e:
                            logger.debug(f"Warning: Could not add {relative_path} to Git exclude file. Error: {e}")

                except OSError as e:
                    logger.debug(f"Warning: Could not create {init_py_path}. Error: {e}")

    if patched_any_file:
        logger.debug("Found and initialized Python package directories in your workspace. This is a one-time operation to enable proper imports. Now restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)
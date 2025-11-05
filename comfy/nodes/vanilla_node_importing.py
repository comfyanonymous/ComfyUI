from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import logging
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from os.path import join, basename, dirname, isdir, isfile, exists, abspath, split, splitext, realpath
from typing import Iterable, Any, Generator
from unittest.mock import patch, MagicMock

from comfy_compatibility.vanilla import prepare_vanilla_environment, patch_pip_install_subprocess_run, patch_pip_install_popen
from . import base_nodes
from .comfyui_v3_package_imports import _comfy_entrypoint_upstream_v3_imports
from .package_typing import ExportedNodes
from ..cmd import folder_paths
from ..component_model.plugins import prompt_server_instance_routes
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context

logger = logging.getLogger(__name__)


class StreamToLogger:
    """
    File-like stream object that redirects writes to a logger instance.
    This is used to capture print() statements from modules during import.
    """

    def __init__(self, logger: logging.Logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        # Process each line from the buffer. Print statements usually end with a newline.
        for line in buf.rstrip().splitlines():
            # Log the line, removing any trailing whitespace
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        # The logger handles its own flushing, so this can be a no-op.
        pass

    @property
    def encoding(self):
        return "utf-8"


class _PromptServerStub(ServerStub):
    def __init__(self):
        super().__init__()
        self.routes = prompt_server_instance_routes
        self.on_prompt_handlers = []

    def add_on_prompt_handler(self, handler):
        # todo: these need to be added to a real prompt server if the loading order is behaving in a complex way
        self.on_prompt_handlers.append(handler)

    def send_sync(self, *args, **kwargs):
        logger.warning(f"Node tried to send a message over the websocket while importing, args={args} kwargs={kwargs}")


def _vanilla_load_importing_execute_prestartup_script(node_paths: Iterable[str]) -> None:
    def execute_script(script_path):
        module_name = splitext(script_path)[0]
        try:
            with _stdout_intercept(module_name):
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return True
        except Exception as e:
            logger.error(f"Failed to execute startup-script: {script_path}", exc_info=e)
        return False

    node_prestartup_times = []
    for custom_node_path in node_paths:
        # patched
        if not isdir(custom_node_path):
            continue
        # end patch
        possible_modules = os.listdir(custom_node_path)

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            script_path = join(module_path, "prestartup_script.py")
            if exists(script_path):
                if "comfyui-manager" in module_path.lower():
                    os.environ['COMFYUI_PATH'] = str(folder_paths.base_path)
                    os.environ['COMFYUI_FOLDERS_BASE_PATH'] = str(folder_paths.models_dir)
                    # Monkey-patch ComfyUI-Manager's security check to prevent it from crashing on startup
                    # and its logging handler to prevent it from taking over logging.
                    glob_path = join(module_path, "glob")
                    glob_path_added = False
                    original_add_handler = logging.Logger.addHandler

                    def no_op_add_handler(self, handler):
                        logger.info(f"Skipping addHandler for {type(handler).__name__} during ComfyUI-Manager prestartup.")

                    try:
                        sys.path.insert(0, glob_path)
                        glob_path_added = True
                        # Patch security_check
                        import security_check  # pylint: disable=import-error
                        original_check = security_check.security_check

                        def patched_security_check():
                            try:
                                return original_check()
                            except Exception as e:
                                logger.error(f"ComfyUI-Manager security_check failed but was caught gracefully: {e}", exc_info=e)

                        security_check.security_check = patched_security_check
                        logger.debug("Patched ComfyUI-Manager's security_check to fail gracefully.")

                        # Patch logging
                        logging.Logger.addHandler = no_op_add_handler
                        logger.debug("Patched logging.Logger.addHandler to prevent ComfyUI-Manager from adding a logging handler.")

                        time_before = time.perf_counter()
                        success = execute_script(script_path)
                        node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
                    except Exception as e:
                        logger.error(f"Failed to patch and execute ComfyUI-Manager's prestartup script: {e}", exc_info=e)
                    finally:
                        if glob_path_added and glob_path in sys.path:
                            sys.path.remove(glob_path)
                        logging.Logger.addHandler = original_add_handler
                else:
                    time_before = time.perf_counter()
                    success = execute_script(script_path)
                    node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))


@contextmanager
def _exec_mitigations(module: types.ModuleType, module_path: str) -> Generator[ExportedNodes, Any, None]:
    if module.__name__.lower() in (
            "comfyui-manager",
            "comfyui_ryanonyheinside",
            "comfyui-easy-use",
            "comfyui_custom_nodes_alekpet",
    ):
        from ..cmd import folder_paths
        old_file = folder_paths.__file__

        try:
            # mitigate path
            new_path = join(abspath(join(dirname(old_file), "..", "..")), basename(old_file))
            config = current_execution_context()

            block_installation = config and config.configuration and config.configuration.block_runtime_package_installation
            with (
                patch.object(folder_paths, "__file__", new_path),
                # mitigate packages installing things dynamically
                patch_pip_install_subprocess_run() if block_installation else nullcontext(),
                patch_pip_install_popen() if block_installation else nullcontext(),
            ):
                yield ExportedNodes()
        finally:
            # todo: mitigate "/manager/reboot"
            # todo: mitigate process_wrap
            # todo: unfortunately, we shouldn't restore the patches here, they will have to be applied forever.
            # concurrent.futures.ThreadPoolExecutor = _ThreadPoolExecutor
            # threading.Thread.start = original_thread_start
            logger.info(f"Exec mitigations were applied for {module.__name__}, due to using the folder_paths.__file__ symbol and manipulating EXTENSION_WEB_DIRS")
    else:
        yield ExportedNodes()


@contextmanager
def _stdout_intercept(name: str):
    original_stdout = sys.stdout

    try:
        module_logger = logging.getLogger(name)
        sys.stdout = StreamToLogger(module_logger, logging.INFO)
        yield
    finally:
        sys.stdout = original_stdout


def _vanilla_load_custom_nodes_1(module_path, ignore: set = None) -> ExportedNodes:
    if ignore is None:
        ignore = set()
    exported_nodes = ExportedNodes()
    module_name = basename(module_path)
    if isfile(module_path):
        sp = splitext(module_path)
        module_name = sp[0]
    try:
        if isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module

        with _exec_mitigations(module, module_path) as mitigated_exported_nodes, _stdout_intercept(module_name):
            module_spec.loader.exec_module(module)
            exported_nodes.update(mitigated_exported_nodes)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = abspath(join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if isdir(web_dir):
                exported_nodes.EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            for name in module.NODE_CLASS_MAPPINGS:
                if name not in ignore:
                    exported_nodes.NODE_CLASS_MAPPINGS[name] = module.NODE_CLASS_MAPPINGS[name]
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module,
                                                                         "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        else:
            logger.error(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")

        exported_nodes.update(_comfy_entrypoint_upstream_v3_imports(module))
    except Exception as e:
        logger.error(f"Cannot import {module_path} module for custom nodes:", exc_info=e)
    return exported_nodes


def _vanilla_load_custom_nodes_2(node_paths: Iterable[str]) -> ExportedNodes:
    from ..cli_args import args
    base_node_names = set(base_nodes.NODE_CLASS_MAPPINGS.keys())
    node_import_times = []
    exported_nodes = ExportedNodes()
    for custom_node_path in node_paths:
        if not exists(custom_node_path) or not isdir(custom_node_path):
            continue
        possible_modules = os.listdir(realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) and splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logger.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue
            if any(fnmatch.fnmatch(possible_module, pattern) for pattern in args.blacklist_custom_nodes):
                logger.info(f"Skipping {possible_module} due to blacklist_custom_nodes")
                continue
            time_before = time.perf_counter()
            possible_exported_nodes = _vanilla_load_custom_nodes_1(module_path, ignore=base_node_names)
            # comfyui-manager mitigation
            import_succeeded = len(possible_exported_nodes.NODE_CLASS_MAPPINGS) > 0 or "ComfyUI-Manager" in module_path
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, import_succeeded))
            exported_nodes.update(possible_exported_nodes)

    if len(node_import_times) > 0:
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logger.debug(f"{n[0]:6.1f} seconds{import_message}: {n[1]}")
    return exported_nodes


def mitigated_import_of_vanilla_custom_nodes() -> ExportedNodes:
    # only vanilla custom nodes will ever go into the custom_nodes directory
    # this mitigation puts files that custom nodes expects are at the root of the repository back where they should be
    # found. we're in the middle of executing the import of execution and server, in all likelihood, so like all things,
    # the way community custom nodes is pretty radioactive
    # there's a lot of subtle details here, and unfortunately, once this is called, there are some things that have
    # to be activated later, in different places, to make all the hacks necessary for custom nodes to work
    prepare_vanilla_environment()

    from ..cmd import folder_paths
    node_paths = folder_paths.get_folder_paths("custom_nodes")

    potential_git_dir_parent = join(dirname(__file__), "..", "..")
    is_git_repository = exists(join(potential_git_dir_parent, ".git"))
    if is_git_repository:
        node_paths += [abspath(join(potential_git_dir_parent, "custom_nodes"))]

    node_paths = frozenset(abspath(custom_node_path) for custom_node_path in node_paths)
    _vanilla_load_importing_execute_prestartup_script(node_paths)
    vanilla_custom_nodes = _vanilla_load_custom_nodes_2(node_paths)
    return vanilla_custom_nodes

from __future__ import annotations

import contextvars
import importlib
import logging
import os
import sys
import time
import types
from contextlib import contextmanager
from functools import partial
from os.path import join, basename, dirname, isdir, isfile, exists, abspath, split, splitext, realpath
from typing import Dict, Iterable

from . import base_nodes
from .package_typing import ExportedNodes
from ..cmd import folder_paths
from ..component_model.plugins import prompt_server_instance_routes
from ..distributed.executors import ContextVarExecutor

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


class _PromptServerStub():
    def __init__(self):
        self.routes = prompt_server_instance_routes


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
            logger.error(f"Failed to execute startup-script: {script_path} / {e}")
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
                        logger.info("Patched ComfyUI-Manager's security_check to fail gracefully.")

                        # Patch logging
                        logging.Logger.addHandler = no_op_add_handler
                        logger.info("Patched logging.Logger.addHandler to prevent ComfyUI-Manager from adding a logging handler.")

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
def _exec_mitigations(module: types.ModuleType, module_path: str) -> ExportedNodes:
    if module.__name__.lower() == "comfyui-manager":
        from ..cmd import folder_paths
        old_file = folder_paths.__file__

        try:
            # mitigate path
            new_path = join(abspath(join(dirname(old_file), "..", "..")), basename(old_file))
            folder_paths.__file__ = new_path
            # mitigate JS copy
            sys.modules['nodes'].EXTENSION_WEB_DIRS = {}

            yield ExportedNodes()
        finally:
            folder_paths.__file__ = old_file
            # todo: mitigate "/manager/reboot"
            # todo: mitigate process_wrap
            # todo: unfortunately, we shouldn't restore the patches here, they will have to be applied forever.
            # concurrent.futures.ThreadPoolExecutor = _ThreadPoolExecutor
            # threading.Thread.start = original_thread_start
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



def _vanilla_load_custom_nodes_1(module_path, ignore=set()) -> ExportedNodes:
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
            return exported_nodes
        else:
            logger.error(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return exported_nodes
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
            time_before = time.perf_counter()
            possible_exported_nodes = _vanilla_load_custom_nodes_1(module_path, base_node_names)
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
    from ..cmd import cuda_malloc, folder_paths, latent_preview, protocol
    from .. import node_helpers
    from .. import __version__
    import concurrent.futures
    import threading
    for module in (cuda_malloc, folder_paths, latent_preview, node_helpers, protocol):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module
    sys.modules['nodes'] = base_nodes
    # apparently this is also something that happens
    sys.modules['comfy.nodes'] = base_nodes
    comfyui_version = types.ModuleType('comfyui_version', '')
    setattr(comfyui_version, "__version__", __version__)
    sys.modules['comfyui_version'] = comfyui_version
    from ..cmd import execution, server
    for module in (execution, server):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module

    if server.PromptServer.instance is None:
        server.PromptServer.instance = _PromptServerStub()

    # Impact Pack wants to find model_patcher
    from .. import model_patcher
    sys.modules['model_patcher'] = model_patcher

    comfy_extras_mitigation: Dict[str, types.ModuleType] = {}

    import comfy_extras
    for module_name, module in sys.modules.items():
        if not module_name.startswith("comfy_extras.nodes"):
            continue
        module_short_name = module_name.split(".")[-1]
        setattr(comfy_extras, module_short_name, module)
        comfy_extras_mitigation[f'comfy_extras.{module_short_name}'] = module
    sys.modules.update(comfy_extras_mitigation)
    node_paths = folder_paths.get_folder_paths("custom_nodes")

    potential_git_dir_parent = join(dirname(__file__), "..", "..")
    is_git_repository = exists(join(potential_git_dir_parent, ".git"))
    if is_git_repository:
        node_paths += [abspath(join(potential_git_dir_parent, "custom_nodes"))]

    node_paths = frozenset(abspath(custom_node_path) for custom_node_path in node_paths)

    _ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor
    original_thread_start = threading.Thread.start
    concurrent.futures.ThreadPoolExecutor = ContextVarExecutor

    # mitigate missing folder names and paths context
    def patched_start(self, *args, **kwargs):
        if not hasattr(self.run, '__wrapped_by_context__'):
            ctx = contextvars.copy_context()
            self.run = partial(ctx.run, self.run)
            setattr(self.run, '__wrapped_by_context__', True)
        original_thread_start(self, *args, **kwargs)

    if not getattr(threading.Thread.start, '__is_patched_by_us', False):
        threading.Thread.start = patched_start
        setattr(threading.Thread.start, '__is_patched_by_us', True)
        logger.info("Patched `threading.Thread.start` to propagate contextvars.")
    _vanilla_load_importing_execute_prestartup_script(node_paths)
    vanilla_custom_nodes = _vanilla_load_custom_nodes_2(node_paths)
    return vanilla_custom_nodes

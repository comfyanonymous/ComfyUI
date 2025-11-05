from __future__ import annotations

import collections.abc
import contextvars
import logging
import subprocess
import sys
import types
from contextlib import contextmanager, nullcontext
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from threading import RLock
from typing import Dict

import wrapt

logger = logging.getLogger(__name__)

# there isn't a way to do this per-thread, it's only per process, so the global is valid
# we don't want some kind of multiprocessing lock, because this is munging the sys.modules
# wrapt.synchronized will be used to synchronize this
_in_environment = False


class _NodeClassMappingsShim(collections.abc.Mapping):
    def __init__(self):
        super().__init__()
        self._active = 0
        self._active_lock = RLock()

    def activate(self):
        with self._active_lock:
            self._active += 1

    def deactivate(self):
        with self._active_lock:
            self._active -= 1

    def __iter__(self):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            for key in get_nodes().NODE_CLASS_MAPPINGS:
                yield key
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            for key in NODE_CLASS_MAPPINGS:
                yield key

    def __getitem__(self, item):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            return get_nodes().NODE_CLASS_MAPPINGS[item]
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            return NODE_CLASS_MAPPINGS[item]

    def __len__(self):
        if self._active > 0:
            from comfy.nodes_context import get_nodes
            return len(get_nodes().NODE_CLASS_MAPPINGS)
        else:
            from comfy.nodes.base_nodes import NODE_CLASS_MAPPINGS
            return len(NODE_CLASS_MAPPINGS)

    # todo: does this need to be mutable?


class _NodeShim:
    def __init__(self):
        self.__name__ = 'nodes'
        self.__package__ = ''

        nodes_file = None
        try:
            # the 'nodes' module is expected to be in the directory above 'comfy'
            spec = find_spec('comfy')
            if spec and spec.origin:
                comfy_package_path = Path(spec.origin).parent
                nodes_module_dir = comfy_package_path.parent
                nodes_file = str(nodes_module_dir / 'nodes.py')
        except (ImportError, AttributeError):
            # don't do anything exotic
            pass

        self.__file__ = nodes_file
        self.__loader__ = None
        self.__spec__ = None

    def __node_class_mappings(self) -> _NodeClassMappingsShim:
        return getattr(self, "NODE_CLASS_MAPPINGS")

    def activate(self):
        self.__node_class_mappings().activate()

    def deactivate(self):
        self.__node_class_mappings().deactivate()


@wrapt.synchronized
def prepare_vanilla_environment():
    global _in_environment
    if _in_environment:
        return
    try:
        from comfy.cmd import cuda_malloc, folder_paths, latent_preview, protocol
    except (ImportError, ModuleNotFoundError):
        if "comfy" in sys.modules:
            logger.debug("not running with ComfyUI LTS installed, skipping vanilla environment prep because we're already in it")
            _in_environment = True
        else:
            logger.warning("unexpectedly, comfy is not in sys.modules nor can we import from the LTS packages")
        return

    # only need to set this up once
    _in_environment = True

    from comfy.distributed.executors import ContextVarExecutor
    from comfy.nodes import base_nodes
    from comfy.nodes.vanilla_node_importing import _PromptServerStub
    from comfy import node_helpers
    from comfy import __version__
    import concurrent.futures
    import threading
    for module in (cuda_malloc, folder_paths, latent_preview, node_helpers, protocol):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module

    # easy-use needs a shim
    # this ensures NODE_CLASS_MAPPINGS is loaded lazily and contains all the nodes loaded so far, not just the base nodes
    # easy-use and other nodes expect NODE_CLASS_MAPPINGS to contain all the nodes in the environment
    # the shim must be activated after importing, which happens in a tightly coupled way
    # todo: it's not clear if we should skip the dunder methods or not
    nodes_shim_dir = {k: getattr(base_nodes, k) for k in dir(base_nodes) if not k.startswith("__")}
    nodes_shim_dir['NODE_CLASS_MAPPINGS'] = _NodeClassMappingsShim()
    nodes_shim_dir['EXTENSION_WEB_DIRS'] = {}

    nodes_shim = _NodeShim()
    for k, v in nodes_shim_dir.items():
        setattr(nodes_shim, k, v)

    sys.modules['nodes'] = nodes_shim

    comfyui_version = types.ModuleType('comfyui_version', '')
    setattr(comfyui_version, "__version__", __version__)
    sys.modules['comfyui_version'] = comfyui_version
    from comfy.cmd import execution, server
    for module in (execution, server):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module
    if server.PromptServer.instance is None:
        server.PromptServer.instance = _PromptServerStub()
    # Impact Pack wants to find model_patcher
    from comfy import model_patcher
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
        logger.debug("Patched `threading.Thread.start` to propagate contextvars.")


@contextmanager
def patch_pip_install_subprocess_run():
    from unittest.mock import patch, MagicMock
    original_subprocess_run = subprocess.run

    def custom_side_effect(*args, **kwargs):
        command_list = args[0] if args else []

        # from easy-use
        is_pip_install_call = (
                isinstance(command_list, list) and
                len(command_list) == 6 and
                command_list[0] == sys.executable and
                command_list[1] == '-s' and
                command_list[2] == '-m' and
                command_list[3] == 'pip' and
                command_list[4] == 'install' and
                isinstance(command_list[5], str)
        )

        if is_pip_install_call:
            package_name = command_list[5]
            logger.info(f"Intercepted and mocked `pip install` for: {package_name}")
            mock_result = MagicMock()
            mock_result.returncode = 0
            return mock_result
        else:
            return original_subprocess_run(*args, **kwargs)

    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = custom_side_effect
        yield


@contextmanager
def patch_pip_install_popen():
    from unittest.mock import patch, MagicMock
    original_subprocess_popen = subprocess.Popen

    def custom_side_effect(*args, **kwargs):
        command_list = args[0] if args else []

        is_pip_install_call = (
                isinstance(command_list, list) and
                len(command_list) >= 5 and
                command_list[0] == sys.executable and
                command_list[1] == "-m" and
                command_list[2] == "pip" and
                command_list[3] == "install" and
                # special case nunchaku
                "nunchaku" not in command_list[4:]
        )

        if is_pip_install_call:
            package_names = command_list[4:]
            logger.info(f"Intercepted and mocked `subprocess.Popen` for: pip install {' '.join(package_names)}")

            mock_popen_instance = MagicMock()
            # make stdout and stderr empty iterables so loops over them complete immediately.
            mock_popen_instance.stdout = []
            mock_popen_instance.stderr = []

            return mock_popen_instance
        else:
            return original_subprocess_popen(*args, **kwargs)

    with patch('subprocess.Popen') as mock_popen:
        mock_popen.side_effect = custom_side_effect
        yield mock_popen


@contextmanager
def vanilla_environment_node_execution_hooks():
    # this handles activating the NODE_CLASS_MAPPINGS shim
    from comfy.execution_context import current_execution_context
    ctx = current_execution_context()

    if 'nodes' in sys.modules and isinstance(sys.modules['nodes'], _NodeShim):
        nodes_shim: _NodeShim = sys.modules['nodes']
        try:
            nodes_shim.activate()

            block_installs = ctx and ctx.configuration and ctx.configuration.block_runtime_package_installation is True
            with (
                patch_pip_install_subprocess_run() if block_installs else nullcontext(),
                patch_pip_install_popen() if block_installs else nullcontext(),
            ):
                yield
        finally:
            nodes_shim.deactivate()
    else:
        yield

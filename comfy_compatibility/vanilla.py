from __future__ import annotations

import contextvars
import logging
import sys
import types
from functools import partial
from typing import Dict

logger = logging.getLogger(__name__)
_in_environment = False


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
    sys.modules['nodes'] = base_nodes
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

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import time
import types
import typing

from . import base_nodes
from comfy_extras import nodes as comfy_extras_nodes

try:
    import custom_nodes
except:
    custom_nodes: typing.Optional[types.ModuleType] = None
from .package_typing import ExportedNodes
from functools import reduce
from pkg_resources import resource_filename
from importlib.metadata import entry_points

_comfy_nodes = ExportedNodes()


def _import_nodes_in_module(exported_nodes: ExportedNodes, module: types.ModuleType):
    node_class_mappings = getattr(module, 'NODE_CLASS_MAPPINGS', None)
    node_display_names = getattr(module, 'NODE_DISPLAY_NAME_MAPPINGS', None)
    web_directory = getattr(module, "WEB_DIRECTORY", None)
    if node_class_mappings:
        exported_nodes.NODE_CLASS_MAPPINGS.update(node_class_mappings)
    if node_display_names:
        exported_nodes.NODE_DISPLAY_NAME_MAPPINGS.update(node_display_names)
    if web_directory:
        # load the extension resources path
        abs_web_directory = os.path.abspath(resource_filename(module.__name__, web_directory))
        if not os.path.isdir(abs_web_directory):
            abs_web_directory = os.path.abspath(os.path.join(os.path.dirname(module.__file__), web_directory))
        if not os.path.isdir(abs_web_directory):
            raise ImportError(path=abs_web_directory)
        exported_nodes.EXTENSION_WEB_DIRS[module.__name__] = abs_web_directory
    return node_class_mappings and len(node_class_mappings) > 0 or web_directory


def _import_and_enumerate_nodes_in_module(module: types.ModuleType, print_import_times=False) -> ExportedNodes:
    exported_nodes = ExportedNodes()
    timings = []
    if _import_nodes_in_module(exported_nodes, module):
        pass
    else:
        # Iterate through all the submodules
        for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
            full_name = module.__name__ + "." + name
            time_before = time.perf_counter()
            success = True

            if full_name.endswith(".disabled"):
                continue
            try:
                submodule = importlib.import_module(full_name)
                # Recursively call the function if it's a package
                exported_nodes.update(
                    _import_and_enumerate_nodes_in_module(submodule, print_import_times=print_import_times))
            except KeyboardInterrupt as interrupted:
                raise interrupted
            except Exception as x:
                logging.error(f"{full_name} import failed", exc_info=x)
                success = False
            timings.append((time.perf_counter() - time_before, full_name, success))

    if print_import_times and len(timings) > 0 or any(not success for (_, _, success) in timings):
        for (duration, module_name, success) in sorted(timings):
            print(f"{duration:6.1f} seconds{'' if success else ' (IMPORT FAILED)'}, {module_name}")
    return exported_nodes


def import_all_nodes_in_workspace() -> ExportedNodes:
    if len(_comfy_nodes) == 0:
        base_and_extra = reduce(lambda x, y: x.update(y),
                                map(_import_and_enumerate_nodes_in_module, [
                                    # this is the list of default nodes to import
                                    base_nodes,
                                    comfy_extras_nodes
                                ]),
                                ExportedNodes())
        custom_nodes_mappings = ExportedNodes()
        if custom_nodes is not None:
            custom_nodes_mappings.update(_import_and_enumerate_nodes_in_module(custom_nodes, print_import_times=True))

        # load from entrypoints
        for entry_point in entry_points().select(group='comfyui.custom_nodes'):
            # Load the module associated with the current entry point
            module = entry_point.load()

            # Ensure that what we've loaded is indeed a module
            if isinstance(module, types.ModuleType):
                custom_nodes_mappings.update(
                    _import_and_enumerate_nodes_in_module(module, print_import_times=True))
        # don't allow custom nodes to overwrite base nodes
        custom_nodes_mappings -= base_and_extra

        _comfy_nodes.update(base_and_extra + custom_nodes_mappings)
    return _comfy_nodes

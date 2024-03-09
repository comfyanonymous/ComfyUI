from __future__ import annotations

import importlib
import os
import sys
import time
import types
from typing import Dict

from . import base_nodes
from .package_typing import ExportedNodes


def _vanilla_load_custom_nodes_1(module_path, ignore=set()) -> ExportedNodes:
    exported_nodes = ExportedNodes()
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module_dir = os.path.split(module_path)[0]
        else:
            module_spec = importlib.util.spec_from_file_location(module_name, os.path.join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = os.path.abspath(os.path.join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if os.path.isdir(web_dir):
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
            print(f"Skip {module_path} module for custom nodes due to the lack of NODE_CLASS_MAPPINGS.")
            return exported_nodes
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return exported_nodes


def _vanilla_load_custom_nodes_2() -> ExportedNodes:
    from ..cmd import folder_paths
    base_node_names = set(base_nodes.NODE_CLASS_MAPPINGS.keys())
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    node_import_times = []
    exported_nodes = ExportedNodes()
    for custom_node_path in node_paths:
        if not os.path.exists(custom_node_path) or not os.path.isdir(custom_node_path):
            continue
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            time_before = time.perf_counter()
            possible_exported_nodes = _vanilla_load_custom_nodes_1(module_path, base_node_names)
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, len(possible_exported_nodes.NODE_CLASS_MAPPINGS) > 0))
            exported_nodes.update(possible_exported_nodes)

    if len(node_import_times) > 0:
        print("\nImport times for custom nodes:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()
    return exported_nodes


def mitigated_import_of_vanilla_custom_nodes() -> ExportedNodes:
    # only vanilla custom nodes will ever go into the custom_nodes directory
    # this mitigation puts files that custom nodes expects are at the root of the repository back where they should be
    # found. we're in the middle of executing the import of execution and server, in all likelihood, so like all things,
    # the way community custom nodes is pretty radioactive
    from ..cmd import cuda_malloc, folder_paths, latent_preview
    for module in (cuda_malloc, folder_paths, latent_preview):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module
    sys.modules['nodes'] = base_nodes
    from ..cmd import execution, server
    for module in (execution, server):
        module_short_name = module.__name__.split(".")[-1]
        sys.modules[module_short_name] = module

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
    vanilla_custom_nodes = _vanilla_load_custom_nodes_2()
    return vanilla_custom_nodes

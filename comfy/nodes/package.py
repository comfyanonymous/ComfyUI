from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import time
import types
from functools import reduce
from importlib.metadata import entry_points

from opentelemetry.trace import Span, Status, StatusCode

from .package_typing import ExportedNodes
from ..cmd.main_pre import tracer
from ..component_model.files import get_package_as_path

_comfy_nodes: ExportedNodes = ExportedNodes()


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
        abs_web_directory = os.path.abspath(get_package_as_path(module.__name__, web_directory))
        if not os.path.isdir(abs_web_directory):
            abs_web_directory = os.path.abspath(os.path.join(os.path.dirname(module.__file__), web_directory))
        if not os.path.isdir(abs_web_directory):
            raise ImportError(path=abs_web_directory)
        exported_nodes.EXTENSION_WEB_DIRS[module.__name__] = abs_web_directory
    return node_class_mappings and len(node_class_mappings) > 0 or web_directory


def _import_and_enumerate_nodes_in_module(module: types.ModuleType,
                                          print_import_times=False,
                                          raise_on_failure=False,
                                          depth=100) -> ExportedNodes:
    exported_nodes = ExportedNodes()
    timings = []
    exceptions = []
    with tracer.start_as_current_span("Load Node") as span:
        time_before = time.perf_counter()
        try:
            module_decl = _import_nodes_in_module(exported_nodes, module)
            full_name = module.__name__
            span.set_attribute("full_name", full_name)
            timings.append((time.perf_counter() - time_before, full_name, True, exported_nodes))
        except Exception as exc:
            logging.error(f"{full_name} import failed", exc_info=exc)
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(exc)
            exceptions.append(exc)
    if module_decl is None or not module_decl:
        # Iterate through all the submodules
        for _, name, is_pkg in pkgutil.iter_modules(module.__path__):
            span: Span
            with tracer.start_as_current_span("Load Node") as span:
                full_name = module.__name__ + "." + name
                time_before = time.perf_counter()
                success = True
                span.set_attribute("full_name", full_name)
                new_nodes = ExportedNodes()
                if full_name.endswith(".disabled"):
                    continue
                try:
                    submodule = importlib.import_module(full_name)
                    # Recursively call the function if it's a package
                    new_nodes = _import_and_enumerate_nodes_in_module(submodule, print_import_times=print_import_times, depth=depth - 1)
                    span.set_attribute("new_nodes.length", len(new_nodes))
                    exported_nodes.update(new_nodes)
                except KeyboardInterrupt as interrupted:
                    raise interrupted
                except Exception as x:
                    if isinstance(x, AttributeError):
                        potential_path_error: AttributeError = x
                        if potential_path_error.name == '__path__':
                            continue
                    logging.error(f"{full_name} import failed", exc_info=x)
                    success = False
                    exceptions.append(x)
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(x)
                timings.append((time.perf_counter() - time_before, full_name, success, new_nodes))

    if print_import_times and len(timings) > 0 or any(not success for (_, _, success, _) in timings):
        for (duration, module_name, success, new_nodes) in sorted(timings):
            logging.info(f"{duration:6.1f} seconds{'' if success else ' (IMPORT FAILED)'}, {module_name} ({len(new_nodes)} nodes loaded)")
    if raise_on_failure and len(exceptions) > 0:
        try:
            raise ExceptionGroup("Node import failed", exceptions)
        except NameError:
            raise exceptions[0]
    return exported_nodes


@tracer.start_as_current_span("Import All Nodes In Workspace")
def import_all_nodes_in_workspace(vanilla_custom_nodes=True, raise_on_failure=False) -> ExportedNodes:
    # now actually import the nodes, to improve control of node loading order
    from comfy_extras import nodes as comfy_extras_nodes
    from ..cli_args import args
    from . import base_nodes
    from .vanilla_node_importing import mitigated_import_of_vanilla_custom_nodes
    # only load these nodes once
    if len(_comfy_nodes) == 0:
        base_and_extra = reduce(lambda x, y: x.update(y),
                                map(lambda module_inner: _import_and_enumerate_nodes_in_module(module_inner, raise_on_failure=raise_on_failure), [
                                    # this is the list of default nodes to import
                                    base_nodes,
                                    comfy_extras_nodes
                                ]),
                                ExportedNodes())
        custom_nodes_mappings = ExportedNodes()

        if args.disable_all_custom_nodes:
            logging.info("Loading custom nodes was disabled, only base and extra nodes were loaded")
            _comfy_nodes.update(base_and_extra)
            return _comfy_nodes

        # load from entrypoints
        for entry_point in entry_points().select(group='comfyui.custom_nodes'):
            # Load the module associated with the current entry point
            try:
                module = entry_point.load()
            except ModuleNotFoundError as module_not_found_error:
                logging.error(f"A module was not found while importing nodes via an entry point: {entry_point}. Please ensure the entry point in setup.py is named correctly", exc_info=module_not_found_error)
                continue

            # Ensure that what we've loaded is indeed a module
            if isinstance(module, types.ModuleType):
                custom_nodes_mappings.update(
                    _import_and_enumerate_nodes_in_module(module, print_import_times=True))

        # load the vanilla custom nodes last
        if vanilla_custom_nodes:
            custom_nodes_mappings += mitigated_import_of_vanilla_custom_nodes()

        # don't allow custom nodes to overwrite base nodes
        custom_nodes_mappings -= base_and_extra

        _comfy_nodes.update(base_and_extra + custom_nodes_mappings)
    return _comfy_nodes

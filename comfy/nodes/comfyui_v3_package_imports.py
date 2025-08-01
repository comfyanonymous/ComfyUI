import asyncio
import inspect
import logging

from comfy.nodes.package_typing import ExportedNodes
from comfy_api.latest import ComfyExtension

logger = logging.getLogger(__name__)


def _comfy_entrypoint_upstream_v3_imports(module) -> ExportedNodes:
    exported_nodes = ExportedNodes()
    if hasattr(module, "comfy_entrypoint"):
        entrypoint = getattr(module, "comfy_entrypoint")
        if not callable(entrypoint):
            logger.debug(f"comfy_entrypoint in {module} is not callable, skipping.")
        else:
            if inspect.iscoroutinefunction(entrypoint):
                # todo: I seriously doubt anything is going to be an async entrypoint, ever
                extension_coro = entrypoint()
                extension = asyncio.run(extension_coro)
            else:
                extension = entrypoint()
            if not isinstance(extension, ComfyExtension):
                logger.debug(f"comfy_entrypoint in {module} did not return a ComfyExtension, skipping.")
            else:
                node_list_coro = extension.get_node_list()
                node_list = asyncio.run(node_list_coro)
                if not isinstance(node_list, list):
                    logger.debug(f"comfy_entrypoint in {module} did not return a list of nodes, skipping.")
                else:
                    for node_cls in node_list:
                        from comfy_api.latest import io
                        node_cls: io.ComfyNode
                        schema = node_cls.GET_SCHEMA()
                        # todo: implement ignore list
                        ignore = {}
                        if schema.node_id not in ignore:
                            exported_nodes.NODE_CLASS_MAPPINGS[schema.node_id] = node_cls
                            # todo: truly, why in the world would you need this?
                            node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format("", "")
                        if schema.display_name is not None:
                            exported_nodes.NODE_DISPLAY_NAME_MAPPINGS[schema.node_id] = schema.display_name
    return exported_nodes

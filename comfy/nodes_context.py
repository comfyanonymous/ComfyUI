# todo: this should be defined in a common place, the fact that the nodes are imported by execution the way that they are is pretty radioactive
import threading

import lazy_object_proxy

from .execution_context import current_execution_context
from .nodes.package import import_all_nodes_in_workspace
from .nodes.package_typing import ExportedNodes, exported_nodes_view

_nodes_local = threading.local()


def invalidate():
    _nodes_local.nodes = lazy_object_proxy.Proxy(import_all_nodes_in_workspace)


def get_nodes() -> ExportedNodes:
    current_ctx = current_execution_context()
    try:
        nodes = _nodes_local.nodes
    except (LookupError, AttributeError):
        nodes = _nodes_local.nodes = lazy_object_proxy.Proxy(import_all_nodes_in_workspace)

    if len(current_ctx.custom_nodes) == 0:
        return nodes
    return exported_nodes_view(nodes, current_ctx.custom_nodes)

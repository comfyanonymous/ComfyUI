# todo: this should be defined in a common place, the fact that the nodes are imported by execution the way that they are is pretty radioactive
import collections.abc
import sys
import threading
from contextlib import contextmanager
from unittest.mock import patch

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


class _NodeClassMappingsShim(collections.abc.Mapping):
    def __iter__(self):
        for key in get_nodes().NODE_CLASS_MAPPINGS:
            yield key

    def __getitem__(self, item):
        return get_nodes().NODE_CLASS_MAPPINGS[item]

    def __len__(self):
        return len(get_nodes().NODE_CLASS_MAPPINGS)

    # todo: does this need to be mutable?


@contextmanager
def vanilla_node_execution_environment():
    # check if we're running with patched nodes
    if 'nodes' in sys.modules:
        # this ensures NODE_CLASS_MAPPINGS is loaded lazily and contains all the nodes loaded so far, not just the base nodes
        # easy-use and other nodes expect NODE_CLASS_MAPPINGS to contain all the nodes in the environment
        with patch('nodes.NODE_CLASS_MAPPINGS', _NodeClassMappingsShim()):
            yield
    else:
        yield

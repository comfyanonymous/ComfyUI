from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Optional, Final

from .component_model import cvpickle
from .component_model.executor_types import ExecutorToClientProgress
from .component_model.folder_path_types import FolderNames
from .distributed.server_stub import ServerStub
from .nodes.package_typing import ExportedNodes, exported_nodes_view
from .progress_types import AbstractProgressRegistry, ProgressRegistryStub


@dataclass(frozen=True)
class ExecutionContext:
    server: ExecutorToClientProgress
    folder_names_and_paths: FolderNames
    custom_nodes: ExportedNodes
    node_id: Optional[str] = None
    task_id: Optional[str] = None
    list_index: Optional[int] = None
    inference_mode: bool = True
    progress_registry: Optional[AbstractProgressRegistry] = None

    def __iter__(self):
        """
        Provides tuple-like unpacking behavior, similar to a NamedTuple.
        Yields task_id, node_id, and list_index.
        """
        yield self.task_id
        yield self.node_id
        yield self.list_index


comfyui_execution_context: Final[ContextVar] = ContextVar("comfyui_execution_context", default=ExecutionContext(server=ServerStub(), folder_names_and_paths=FolderNames(is_root=True), custom_nodes=ExportedNodes(), progress_registry=ProgressRegistryStub()))
# enables context var propagation across process boundaries for process pool executors
cvpickle.register_contextvar(comfyui_execution_context, __name__)


def current_execution_context() -> ExecutionContext:
    return comfyui_execution_context.get()


@contextmanager
def _new_execution_context(ctx: ExecutionContext):
    token = comfyui_execution_context.set(ctx)
    try:
        yield ctx
    finally:
        comfyui_execution_context.reset(token)


@contextmanager
def context_folder_names_and_paths(folder_names_and_paths: FolderNames):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, folder_names_and_paths=folder_names_and_paths)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_execute_prompt(server: ExecutorToClientProgress, prompt_id: str, progress_registry: AbstractProgressRegistry, inference_mode: bool = True):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, server=server, task_id=prompt_id, inference_mode=inference_mode, progress_registry=progress_registry)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_execute_node(node_id: str):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, node_id=node_id)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_add_custom_nodes(exported_nodes: ExportedNodes):
    """
    Adds custom nodes to the execution context
    :param exported_nodes: an object that represents a gathering of custom node export symbols
    :return: a context
    """
    current_ctx = current_execution_context()
    if len(exported_nodes) == 0:
        yield current_ctx

    if len(current_ctx.custom_nodes) == 0:
        merged_custom_nodes = exported_nodes
    else:
        merged_custom_nodes = exported_nodes_view(current_ctx.custom_nodes, exported_nodes)

    new_ctx = replace(current_ctx, custom_nodes=merged_custom_nodes)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_set_node_and_prompt(prompt_id: str, node_id: str, list_index: Optional[int] = None):
    """
    A context manager to set the prompt_id (task_id), node_id, and optional list_index for the current execution.
    This is useful for fine-grained context setting within a node's execution, especially for batch processing.

    Replaces the @guill code upstream
    """
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, task_id=prompt_id, node_id=node_id, list_index=list_index)
    with _new_execution_context(new_ctx):
        yield new_ctx

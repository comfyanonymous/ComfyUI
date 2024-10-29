from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Optional, Final

from .component_model.executor_types import ExecutorToClientProgress
from .component_model.folder_path_types import FolderNames
from .distributed.server_stub import ServerStub

_current_context: Final[ContextVar] = ContextVar("comfyui_execution_context")


@dataclass(frozen=True)
class ExecutionContext:
    server: ExecutorToClientProgress
    folder_names_and_paths: FolderNames
    node_id: Optional[str] = None
    task_id: Optional[str] = None
    inference_mode: bool = True


_current_context.set(ExecutionContext(server=ServerStub(), folder_names_and_paths=FolderNames()))


def current_execution_context() -> ExecutionContext:
    return _current_context.get()


@contextmanager
def _new_execution_context(ctx: ExecutionContext):
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


@contextmanager
def context_folder_names_and_paths(folder_names_and_paths: FolderNames):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, folder_names_and_paths=folder_names_and_paths)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_execute_prompt(server: ExecutorToClientProgress, prompt_id: str, inference_mode: bool = True):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, server=server, task_id=prompt_id, inference_mode=inference_mode)
    with _new_execution_context(new_ctx):
        yield new_ctx


@contextmanager
def context_execute_node(node_id: str):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, node_id=node_id)
    with _new_execution_context(new_ctx):
        yield new_ctx

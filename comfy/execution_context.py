from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Optional, Final

from .component_model.executor_types import ExecutorToClientProgress
from .distributed.server_stub import ServerStub

_current_context: Final[ContextVar] = ContextVar("comfyui_execution_context")


@dataclass(frozen=True)
class ExecutionContext:
    server: ExecutorToClientProgress
    node_id: Optional[str] = None
    task_id: Optional[str] = None
    inference_mode: bool = True


_empty_execution_context: Final[ExecutionContext] = ExecutionContext(server=ServerStub())


def current_execution_context() -> ExecutionContext:
    try:
        return _current_context.get()
    except LookupError:
        return _empty_execution_context


@contextmanager
def new_execution_context(ctx: ExecutionContext):
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)


@contextmanager
def context_execute_node(node_id: str, prompt_id: str):
    current_ctx = current_execution_context()
    new_ctx = replace(current_ctx, node_id=node_id, task_id=prompt_id)
    with new_execution_context(new_ctx):
        yield new_ctx

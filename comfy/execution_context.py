from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple, Optional

from .component_model.executor_types import ExecutorToClientProgress
from .distributed.server_stub import ServerStub

_current_context = ContextVar("comfyui_execution_context")


class ExecutionContext(NamedTuple):
    server: ExecutorToClientProgress
    node_id: Optional[str] = None
    task_id: Optional[str] = None


_empty_execution_context = ExecutionContext(ServerStub())


def current_execution_context() -> ExecutionContext:
    try:
        return _current_context.get()
    except LookupError:
        return _empty_execution_context


@contextmanager
def new_execution_context(ctx: ExecutionContext):
    token = _current_context.set(ctx)
    try:
        yield
    finally:
        _current_context.reset(token)


@contextmanager
def context_execute_node(node_id: str, prompt_id: str):
    current_ctx = current_execution_context()
    new_ctx = ExecutionContext(current_ctx.server, node_id, prompt_id)
    with new_execution_context(new_ctx):
        yield

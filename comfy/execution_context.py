from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

from comfy.component_model.executor_types import ExecutorToClientProgress
from comfy.distributed.server_stub import ServerStub

_current_context = ContextVar("comfyui_execution_context")


class ExecutionContext(NamedTuple):
    server: ExecutorToClientProgress


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

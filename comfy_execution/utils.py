from __future__ import annotations

from typing import Optional

from comfy import execution_context as core_execution_context

ExecutionContext = core_execution_context.ExecutionContext
"""
Context information about the currently executing node.
This is a compatibility wrapper around the core execution context.

Attributes:
    prompt_id: The ID of the currently executing prompt (task_id in core context)
    node_id: The ID of the currently executing node
    list_index: The index in a list being processed (for operations on batches/lists)
"""


def get_executing_context() -> Optional[ExecutionContext]:
    """
    Gets the current execution context from the core context provider.
    Returns a compatibility ExecutionContext object or None if not in an execution context.
    """
    ctx = core_execution_context.current_execution_context()
    if ctx.task_id is None or ctx.node_id is None:
        return None
    return ctx


class CurrentNodeContext:
    """
    Context manager for setting the current executing node context.
    This is a wrapper around the core `context_set_node_and_prompt` context manager.

    Example:
        with CurrentNodeContext(prompt_id="abc", node_id="123", list_index=0):
            # Code that should run with the current node context set
            process_image()
    """

    def __init__(self, prompt_id: str, node_id: str, list_index: Optional[int] = None):
        self._cm = core_execution_context.context_set_node_and_prompt(
            prompt_id=prompt_id,
            node_id=node_id,
            list_index=list_index,
        )

    def __enter__(self):
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cm.__exit__(exc_type, exc_val, exc_tb)

import contextvars
from typing import Optional, NamedTuple

class ExecutionContext(NamedTuple):
    """
    Context information about the currently executing node.

    Attributes:
        node_id: The ID of the currently executing node
        list_index: The index in a list being processed (for operations on batches/lists)
    """
    prompt_id: str
    node_id: str
    list_index: Optional[int]

current_executing_context: contextvars.ContextVar[Optional[ExecutionContext]] = contextvars.ContextVar("current_executing_context", default=None)
_client_id_unset = object()
current_client_id = contextvars.ContextVar("current_client_id", default=_client_id_unset)

def get_executing_context() -> Optional[ExecutionContext]:
    return current_executing_context.get(None)

def get_current_client_id() -> Optional[str]:
    value = current_client_id.get()
    return None if value is _client_id_unset else value

def has_current_client_id() -> bool:
    return current_client_id.get() is not _client_id_unset

def set_current_client_id(client_id: Optional[str]):
    return current_client_id.set(client_id)

def reset_current_client_id(token):
    current_client_id.reset(token)

class CurrentNodeContext:
    """
    Context manager for setting the current executing node context.

    Sets the current_executing_context on enter and resets it on exit.

    Example:
        with CurrentNodeContext(node_id="123", list_index=0):
            # Code that should run with the current node context set
            process_image()
    """
    def __init__(self, prompt_id: str, node_id: str, list_index: Optional[int] = None):
        self.context = ExecutionContext(
            prompt_id= prompt_id,
            node_id= node_id,
            list_index= list_index
        )
        self.token = None

    def __enter__(self):
        self.token = current_executing_context.set(self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            current_executing_context.reset(self.token)

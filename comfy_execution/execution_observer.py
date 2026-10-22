"""In-process observers of prompt and node execution, for extensions such as tracing or profiling.

Observers are called synchronously by the prompt worker, inside the asyncio task that runs the
prompt. Context variables an observer sets in a start method are therefore visible to the node
code that runs next, and it can reset them in the matching end method. Observer methods should
return quickly. Exceptions they raise are logged and never break execution.
"""

import logging
import threading
from typing import List, Optional, Tuple

_logger = logging.getLogger(__name__)


class ExecutionObserver:
    """Base class with no-op methods; override the ones you need."""

    def on_prompt_start(self, prompt_id: str) -> None:
        pass

    def on_nodes_cached(self, prompt_id: str, node_ids: List[str]) -> None:
        """Called once per prompt with the nodes whose outputs come from the cache. These nodes get no
        on_node_start or on_node_end calls."""
        pass

    def on_node_start(self, prompt_id: str, node_id: str, class_type: str, parent_node_id: Optional[str]) -> None:
        """Called each time the executor runs a node. parent_node_id is set for nodes created by
        another node's expansion."""
        pass

    def on_node_end(self, prompt_id: str, node_id: str, outcome: str, exception: Optional[BaseException]) -> None:
        """outcome is "success", "error", "interrupted" or "pending". A pending node is waiting on async
        work or on the nodes it expanded into, and is started again when it resumes."""
        pass

    def on_prompt_end(self, prompt_id: str, outcome: str) -> None:
        """Always called after on_prompt_start. outcome is "success", "error" or "interrupted"."""
        pass


_observers: List[ExecutionObserver] = []
_observers_lock = threading.Lock()
_observers_snapshot: Tuple[ExecutionObserver, ...] = ()


def register_execution_observer(observer: ExecutionObserver) -> None:
    """Register an execution observer. Observers are called in registration order."""
    global _observers_snapshot
    with _observers_lock:
        if observer in _observers:
            _logger.warning(f"Execution observer {observer.__class__.__name__} already registered")
            return
        _observers.append(observer)
        _observers_snapshot = tuple(_observers)


def unregister_execution_observer(observer: ExecutionObserver) -> None:
    global _observers_snapshot
    with _observers_lock:
        try:
            _observers.remove(observer)
            _observers_snapshot = tuple(_observers)
        except ValueError:
            _logger.warning(f"Execution observer {observer.__class__.__name__} was not registered")


def has_execution_observers() -> bool:
    return bool(_observers_snapshot)


def notify_execution_observers(method: str, *args) -> None:
    for observer in _observers_snapshot:
        try:
            getattr(observer, method)(*args)
        except Exception as e:
            _logger.warning(f"Execution observer {observer.__class__.__name__} error in {method}: {e}")

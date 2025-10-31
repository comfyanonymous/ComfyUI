from .execution_context import current_execution_context
from typing import Any, Generator, Sequence


def is_input_pending(*arg_names: Sequence[str]) -> Generator[bool, Any, None]:
    """
    returns true if the given argument in the context of an executing node is not scheduled nor executed
    this will be true for inputs that are marked as lazy, and this method is more robust against nodes that return None
    :param arg_names: each arg to evaluate
    :return: True for each arg that is not scheduled nor executed
    """
    context = current_execution_context()
    if context is None or context.execution_list is None:
        raise LookupError("Not executing a node")
    # assert context.execution_list is not None
    # dynprompt = context.execution_list.dynprompt
    executed = context.executed or frozenset()
    # execution_list = context.execution_list
    inputs = context.inputs
    # unscheduled_unexecuted = dynprompt.all_node_ids() - executed - set(execution_list.pendingNodes.keys())
    for arg_name in arg_names:
        if arg_name not in inputs:
            raise ValueError(f"Input {arg_name} not found")
        input_ = inputs[arg_name]
        if isinstance(input_, list) or isinstance(input_, tuple) and len(input_) == 2:
            node_id, *_ = input_
            yield node_id not in executed
        else:
            yield False

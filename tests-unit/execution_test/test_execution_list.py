from comfy_execution.graph import ExecutionList, PROJECTED_BLOCKER


def test_increment_pending_data_dependency_still_blocks():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": {"producer"}, "producer": set()}
    execution_list.blocking = {"producer": {"consumer": {0: True}}}
    execution_list.execution_cache = {"consumer": {"producer": None}}
    execution_list.increment_pending_nodes = {"producer"}

    assert execution_list.get_ready_nodes() == ["producer"]


def test_cached_data_dependency_does_not_requeue():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": {"producer"}, "producer": set()}
    execution_list.blocking = {"producer": {"consumer": {0: True}}}
    execution_list.execution_cache = {"consumer": {"producer": object()}}
    execution_list.increment_pending_nodes = {"producer"}

    assert execution_list.get_ready_nodes() == ["consumer"]


def test_completed_projected_node_does_not_block_projector():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"projector": True, "projected": True}
    execution_list.blockers = {"projector": {"projected"}, "projected": set()}
    execution_list.blocking = {"projected": {"projector": {PROJECTED_BLOCKER: True}}}
    execution_list.execution_cache = {"projector": {}}
    execution_list.increment_pending_nodes = {"projected"}

    assert execution_list.get_ready_nodes() == ["projector"]


def test_unfinished_projected_node_blocks_projector():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"projector": True, "projected": True}
    execution_list.blockers = {"projector": {"projected"}, "projected": set()}
    execution_list.blocking = {"projected": {"projector": {PROJECTED_BLOCKER: True}}}
    execution_list.execution_cache = {"projector": {}}
    execution_list.increment_pending_nodes = set()

    assert execution_list.get_ready_nodes() == ["projected"]

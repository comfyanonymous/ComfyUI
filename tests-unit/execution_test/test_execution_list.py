from comfy_execution.graph import ExecutionList, LAZY_BLOCKER, PROJECTED_BLOCKER


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


def test_previous_iteration_lazy_dependency_does_not_requeue():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": {"producer"}, "producer": set()}
    execution_list.blocking = {"producer": {"consumer": {0: True, LAZY_BLOCKER: True}}}
    execution_list.execution_cache = {"consumer": {"producer": None}}
    execution_list.increment_pending_nodes = {"producer"}

    assert execution_list.get_ready_nodes() == ["consumer"]


def test_requested_lazy_dependency_requeues():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": {"producer"}, "producer": set()}
    execution_list.blocking = {"producer": {"consumer": {0: True, LAZY_BLOCKER: True}}}
    execution_list.execution_cache = {"consumer": {"producer": None}}
    execution_list.increment_pending_nodes = set()

    assert execution_list.get_ready_nodes() == ["producer"]


class _DynamicPrompt:
    def get_node(self, node_id):
        assert node_id == "consumer"
        return {"inputs": {"selected": ["producer", 0]}}


class _OutputCache:
    def get_local(self, node_id):
        return None


def test_lazy_dependency_is_marked_when_requested():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.dynprompt = _DynamicPrompt()
    execution_list.output_cache = _OutputCache()
    execution_list.output_link_callback = None
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": set(), "producer": set()}
    execution_list.blocking = {"consumer": {}, "producer": {}}
    execution_list.execution_cache = {}
    execution_list.execution_cache_listeners = {}
    execution_list.deferred_output_cache = {}
    execution_list.projected_node_counts = {}

    execution_list.make_input_strong_link("consumer", "selected")

    assert execution_list.blocking["producer"]["consumer"] == {0: True, LAZY_BLOCKER: True}

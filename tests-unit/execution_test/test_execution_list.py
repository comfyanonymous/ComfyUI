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


def test_dormant_consumer_does_not_reactivate_its_dependencies():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {"consumer": True, "producer": True}
    execution_list.blockers = {"consumer": {"producer"}, "producer": set()}
    execution_list.blocking = {"producer": {"consumer": {0: True}}}
    execution_list.execution_cache = {"consumer": {"producer": None}}
    execution_list.increment_pending_nodes = {"consumer", "producer"}

    assert execution_list.get_ready_nodes() == []
    assert execution_list.increment_pending_nodes == {"consumer", "producer"}


def test_dependency_reactivation_propagates_only_from_active_nodes():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {
        "leaf": True,
        "middle": True,
        "consumer": True,
        "stale_lazy_input": True,
    }
    execution_list.blockers = {
        "leaf": set(),
        "middle": {"leaf", "stale_lazy_input"},
        "consumer": {"middle"},
        "stale_lazy_input": set(),
    }
    execution_list.blocking = {
        "leaf": {"middle": {0: True}},
        "middle": {"consumer": {0: True}},
        "consumer": {},
        "stale_lazy_input": {"middle": {0: True, LAZY_BLOCKER: True}},
    }
    execution_list.execution_cache = {
        "middle": {"leaf": None, "stale_lazy_input": None},
        "consumer": {"middle": None},
    }
    execution_list.increment_pending_nodes = {
        "leaf",
        "middle",
        "stale_lazy_input",
    }

    assert execution_list.get_ready_nodes() == ["leaf"]
    assert execution_list.increment_pending_nodes == {"stale_lazy_input"}


def test_inherited_dynamic_nodes_only_schedule_output_nodes():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.pendingNodes = {
        "projector": True,
        "parent": True,
        "branch": True,
        "output": True,
    }
    execution_list.blockers = {
        "projector": set(),
        "parent": set(),
        "branch": set(),
        "output": set(),
    }
    execution_list.blocking = {
        "projector": {},
        "parent": {},
        "branch": {},
        "output": {},
    }
    execution_list.increment_pending_nodes = set()
    execution_list.projection_nodes = {"projector": {"parent"}}
    execution_list.projection_scheduled_nodes = {"projector": {"parent"}}
    execution_list.projected_node_counts = {"parent": 1}
    execution_list.projected_node_owners = {"parent": {"projector"}}

    execution_list.inherit_projected_nodes(
        "parent", {"branch", "output"}, {"output"}
    )

    assert execution_list.projection_nodes["projector"] == {
        "parent",
        "branch",
        "output",
    }
    assert execution_list.projection_scheduled_nodes["projector"] == {
        "parent",
        "output",
    }
    assert execution_list.blocking["branch"]["projector"] == {
        PROJECTED_BLOCKER: True
    }


def test_requeue_activates_selected_dynamic_branch_but_not_stale_lazy_input():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.output_cache = _OutputCache()
    execution_list.pendingNodes = {
        "parent": True,
        "branch": True,
        "stale_lazy_input": True,
        "output": True,
    }
    execution_list.blockers = {
        "parent": {"branch"},
        "branch": {"stale_lazy_input"},
        "stale_lazy_input": set(),
        "output": set(),
    }
    execution_list.blocking = {
        "parent": {},
        "branch": {"parent": {0: True}},
        "stale_lazy_input": {
            "branch": {0: True, LAZY_BLOCKER: True},
        },
        "output": {},
    }
    execution_list.execution_cache = {
        "parent": {"branch": object()},
        "branch": {"stale_lazy_input": object()},
    }
    execution_list.projected_node_counts = {
        "parent": 1,
        "branch": 1,
        "stale_lazy_input": 1,
        "output": 1,
    }
    execution_list.increment_pending_nodes = set()

    execution_list.requeue_nodes(
        {"parent", "output"},
        {"parent", "branch", "stale_lazy_input", "output"},
    )

    assert set(execution_list.get_ready_nodes()) == {"branch", "output"}
    assert "stale_lazy_input" in execution_list.increment_pending_nodes


def test_nested_projection_does_not_schedule_dormant_outer_nodes():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.staged_node_id = "inner_loop"
    execution_list.pendingNodes = {
        "inner_loop": True,
        "active": True,
        "dormant": True,
    }
    execution_list.blockers = {
        "inner_loop": set(),
        "active": set(),
        "dormant": set(),
    }
    execution_list.blocking = {
        "inner_loop": {},
        "active": {},
        "dormant": {},
    }
    execution_list.increment_pending_nodes = {"dormant"}
    execution_list.projection_nodes = {}
    execution_list.projection_scheduled_nodes = {}
    execution_list.projected_node_counts = {"dormant": 1}
    execution_list.projected_node_owners = {"dormant": {"outer_loop"}}

    projected, scheduled = execution_list.project_nodes(
        {"active", "dormant"}, set()
    )

    assert projected == {"active", "dormant"}
    assert scheduled == {"active"}
    assert "active" in execution_list.blockers["inner_loop"]
    assert "dormant" not in execution_list.blockers["inner_loop"]


def test_ordering_link_holds_nested_close_until_opener_finishes():
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.output_cache = _OutputCache()
    execution_list.pendingNodes = {"opener": True, "close": True}
    execution_list.blockers = {"opener": set(), "close": set()}
    execution_list.blocking = {"opener": {}, "close": {}}
    execution_list.execution_cache = {}
    execution_list.execution_cache_listeners = {}
    execution_list.deferred_output_cache = {}
    execution_list.increment_pending_nodes = set()

    execution_list.add_ordering_link("opener", "close")

    assert execution_list.get_ready_nodes() == ["opener"]
    assert execution_list.blocking["opener"]["close"] == {None: True}
    assert execution_list.execution_cache == {}

    execution_list.pop_node("opener")
    assert execution_list.get_ready_nodes() == ["close"]


def _projection_release_execution_list(count, owners):
    execution_list = ExecutionList.__new__(ExecutionList)
    execution_list.staged_node_id = "inner"
    execution_list.pendingNodes = {"inner": True, "active": True}
    execution_list.blockers = {"inner": {"active"}, "active": set()}
    execution_list.blocking = {
        "inner": {},
        "active": {"inner": {PROJECTED_BLOCKER: True}},
    }
    execution_list.increment_pending_nodes = set()
    execution_list.spent_nodes = set()
    execution_list.projection_nodes = {"inner": {"active"}}
    execution_list.projection_scheduled_nodes = {"inner": {"active"}}
    execution_list.projected_node_counts = {"active": count}
    execution_list.projected_node_owners = {"active": set(owners)}
    return execution_list


def test_releasing_last_projection_retires_active_pending_nodes():
    execution_list = _projection_release_execution_list(1, {"inner"})

    execution_list.release_projected_nodes()

    assert "active" in execution_list.spent_nodes
    assert "active" not in execution_list.projected_node_counts


def test_releasing_nested_projection_keeps_outer_owned_node_live():
    execution_list = _projection_release_execution_list(2, {"outer", "inner"})

    execution_list.release_projected_nodes()

    assert "active" not in execution_list.spent_nodes
    assert execution_list.projected_node_counts["active"] == 1
    assert execution_list.projected_node_owners["active"] == {"outer"}

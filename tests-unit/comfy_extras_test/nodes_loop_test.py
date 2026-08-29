from types import SimpleNamespace

import pytest

import comfy_extras.nodes_loop as nodes_loop


class DynPrompt:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_node(self, node_id):
        return self.prompt[node_id]

    def all_node_ids(self):
        return set(self.prompt)


class ExecutionList:
    pendingNodes = {"close"}

    def __init__(self):
        self.states = {}

    def get_projection_state(self, node_id):
        return self.states.get(node_id)

    def set_projection_state(self, node_id, state):
        self.states[node_id] = state

    def clear_projection_state(self, node_id):
        self.states.pop(node_id, None)

    def project_nodes(self, projected_nodes, scheduled_nodes):
        return projected_nodes, scheduled_nodes

    def requeue_nodes(self, *args):
        pass

    def defer_staged_node(self):
        pass

    def release_projected_nodes(self):
        pass

    def add_external_block(self, node_id):
        return lambda: None

    def add_strong_link(self, *args):
        pass

    def cache_link(self, *args):
        pass

    def get_input_info(self, *args):
        return None, None, None


def run_loop(monkeypatch, mode, initial_value=None):
    execution_list = ExecutionList()
    dynprompt = DynPrompt({
        "loop": {"class_type": "Loop", "inputs": {}},
        "close": {
            "class_type": "CloseLoop",
            "inputs": {
                "output_value": ["loop", 0],
                "next_value": ["loop", 0],
                "accumulate": False,
            },
        },
    })
    nodes_loop.Loop.hidden = SimpleNamespace(
        dynprompt=dynprompt,
        execution_list=execution_list,
        unique_id="loop",
    )
    monkeypatch.setattr(nodes_loop, "loop_projection", lambda dynprompt, node_id: (set(), "close"))
    monkeypatch.setattr(
        nodes_loop,
        "PromptServer",
        SimpleNamespace(instance=SimpleNamespace(send_progress_text=lambda *args: None)),
    )

    outputs = []
    while True:
        output = nodes_loop.Loop.execute(mode, initial_value=initial_value)
        outputs.append(output.result)
        if output[0] is None:
            return outputs, execution_list


def test_loop_list_mode_iterates_list_items(monkeypatch):
    outputs, _ = run_loop(monkeypatch, {"mode": ["List"], "list": ["first", "second"]})

    assert outputs == [
        (0, True, False, "first", None),
        (1, False, True, "second", 0),
        (None, False, True, None, None),
    ]
    assert run_loop(monkeypatch, {"mode": ["List"], "list": []})[0] == [
        (None, False, True, None, None),
    ]


def test_loop_count_modes_remain_list_aware(monkeypatch):
    assert run_loop(monkeypatch, {"mode": ["simple"], "num_iterations": [2]}, ["initial"])[0] == [
        (0, True, False, None, "initial"),
        (1, False, True, None, 0),
        (None, False, True, None, None),
    ]
    assert run_loop(
        monkeypatch,
        {
            "mode": ["For"],
            "start_iteration": [3],
            "max_iteration": [0],
            "step": [-2],
        },
    )[0] == [
        (3, True, False, None, None),
        (1, False, True, None, 3),
        (None, False, True, None, None),
    ]


def test_close_loop_selects_final_or_accumulated_output(monkeypatch):
    _, execution_list = run_loop(monkeypatch, {"mode": ["simple"], "num_iterations": [2]})
    nodes_loop.CloseLoop.hidden = SimpleNamespace(execution_list=execution_list, unique_id="close")

    assert nodes_loop.CloseLoop.execute(None, None, [False]).result == ([1],)

    _, execution_list = run_loop(monkeypatch, {"mode": ["simple"], "num_iterations": [2]})
    nodes_loop.CloseLoop.hidden = SimpleNamespace(execution_list=execution_list, unique_id="close")

    assert nodes_loop.CloseLoop.execute(None, None, [True]).result == ([0, 1],)


def test_loop_schema_has_integrated_carried_value():
    inputs = nodes_loop.Loop.INPUT_TYPES()

    assert list(inputs["optional"]) == ["iteration_outer", "initial_value"]
    assert nodes_loop.Loop.RETURN_NAMES == [
        "iteration",
        "is_first",
        "is_last",
        "list_item",
        "current_value",
    ]
    assert "LoopVariable" not in nodes_loop.NODE_CLASS_MAPPINGS


def test_close_loop_schema_has_one_list_output_and_dynamic_terminations():
    inputs = nodes_loop.CloseLoop.INPUT_TYPES()

    assert list(inputs["required"]) == ["output_value", "next_value", "accumulate"]
    assert list(inputs["optional"]) == ["terminations"]
    assert nodes_loop.CloseLoop.RETURN_NAMES == ["output"]
    assert nodes_loop.CloseLoop.OUTPUT_IS_LIST == [True]


def test_loop_step_must_not_be_zero(monkeypatch):
    with pytest.raises(ValueError, match="step must not be 0"):
        run_loop(
            monkeypatch,
            {"mode": ["For"], "start_iteration": [0], "max_iteration": [1], "step": [0]},
        )

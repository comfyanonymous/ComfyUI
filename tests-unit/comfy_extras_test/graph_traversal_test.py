import pytest

from comfy_extras.graph_traversal import loop_projection


class DynPrompt:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_node(self, node_id):
        return self.prompt[node_id]

    def all_node_ids(self):
        return set(self.prompt)


class NavigableNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


@pytest.fixture(autouse=True)
def node_classes(monkeypatch):
    import nodes

    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "Loop": NavigableNode,
            "CloseLoop": NavigableNode,
            "Body": NavigableNode,
            "Preview": NavigableNode,
            "Output": NavigableNode,
        },
    )


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


def test_loop_projection_contains_only_nodes_which_reach_close():
    prompt = DynPrompt({
        "loop": node("Loop"),
        "body": node("Body", value=["loop", 0]),
        "close": node(
            "CloseLoop",
            output_value=["body", 0],
            next_value=["body", 0],
            accumulate=False,
        ),
        "after": node("Output", value=["close", 0]),
    })

    assert loop_projection(prompt, "loop") == ({"body"}, "close")


def test_loop_projection_requires_exactly_one_close():
    no_close = DynPrompt({
        "loop": node("Loop"),
        "body": node("Body", value=["loop", 0]),
    })
    with pytest.raises(ValueError, match="exactly one Close Loop, found 0"):
        loop_projection(no_close, "loop")

    two_closes = DynPrompt({
        "loop": node("Loop"),
        "body": node("Body", value=["loop", 0]),
        "close1": node("CloseLoop", output_value=["body", 0], next_value=["body", 0]),
        "close2": node("CloseLoop", output_value=["body", 0], next_value=["body", 0]),
    })
    with pytest.raises(ValueError, match="exactly one Close Loop, found 2"):
        loop_projection(two_closes, "loop")


def test_loop_projection_rejects_unterminated_downstream_branch():
    prompt = DynPrompt({
        "loop": node("Loop"),
        "body": node("Body", value=["loop", 0]),
        "preview": node("Preview", value=["body", 0]),
        "close": node("CloseLoop", output_value=["body", 0], next_value=["body", 0]),
    })

    with pytest.raises(ValueError, match="do not terminate.*preview"):
        loop_projection(prompt, "loop")


def test_termination_input_brings_preview_branch_into_loop_body():
    prompt = DynPrompt({
        "loop": node("Loop"),
        "body": node("Body", value=["loop", 0]),
        "preview": node("Preview", value=["body", 0]),
        "close": node(
            "CloseLoop",
            output_value=["body", 0],
            next_value=["body", 0],
            termination0=["preview", 0],
        ),
    })

    assert loop_projection(prompt, "loop") == ({"body", "preview"}, "close")


def test_nested_phase_step_and_window_loops_have_distinct_closes():
    prompt = DynPrompt({
        "phase": node("Loop"),
        "step": node("Loop", iteration_outer=["phase", 0], initial_value=["phase", 4]),
        "sample": node("Body", latent=["step", 4], iteration=["step", 0]),
        "step_close": node(
            "CloseLoop",
            output_value=["sample", 0],
            next_value=["sample", 0],
            accumulate=False,
        ),
        "split": node("Body", latent=["step_close", 0]),
        "window": node("Loop", list=["split", 0]),
        "process": node("Body", latent=["window", 3]),
        "save": node("Preview", image=["process", 0], last=["window", 2]),
        "window_close": node(
            "CloseLoop",
            output_value=["process", 0],
            next_value=["process", 0],
            accumulate=True,
            termination0=["save", 0],
        ),
        "merge": node("Body", windows=["window_close", 0]),
        "phase_close": node(
            "CloseLoop",
            output_value=["merge", 0],
            next_value=["merge", 0],
            accumulate=False,
        ),
        "output": node("Output", value=["phase_close", 0]),
    })

    assert loop_projection(prompt, "step") == ({"sample"}, "step_close")
    assert loop_projection(prompt, "window") == (
        {"process", "save"},
        "window_close",
    )
    assert loop_projection(prompt, "phase") == (
        {
            "step",
            "sample",
            "step_close",
            "split",
            "window",
            "process",
            "save",
            "window_close",
            "merge",
        },
        "phase_close",
    )

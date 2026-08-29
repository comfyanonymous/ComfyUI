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

from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker
from .tools import VariantSupport

NUM_FLOW_SOCKETS = 5
@VariantSupport()
class TestWhileLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"][f"initial_value{i}"] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["FLOW_CONTROL"] + [f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_open"

    CATEGORY = "Testing/Flow"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get(f"initial_value{i}", None))
        return tuple(["stub"] + values)

@VariantSupport()
class TestWhileLoopClose:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow_control": ("FLOW_CONTROL", {"rawLink": True}),
                "condition": ("BOOLEAN", {"forceInput": True}),
            },
            "optional": {
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"][f"initial_value{i}"] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple([f"value{i}" for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_close"

    CATEGORY = "Testing/Flow"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)


    def while_loop_close(self, flow_control, condition, dynprompt=None, unique_id=None, **kwargs):
        assert dynprompt is not None
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get(f"initial_value{i}", None))
            return tuple(values)

        # We want to loop
        upstream = {}
        # Get the list of all nodes between the open and close nodes
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # We'll use the default prefix, but to avoid having node names grow exponentially in size,
        # we'll use "Recurse" for the name of the recursively-generated copy of this node.
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            assert node is not None
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    assert parent is not None
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        assert new_open is not None
        for i in range(NUM_FLOW_SOCKETS):
            key = f"initial_value{i}"
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse")
        assert my_clone is not None
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

@VariantSupport()
class TestExecutionBlockerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "input": ("*",),
                "block": ("BOOLEAN",),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }
        return inputs

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execution_blocker"

    CATEGORY = "Testing/Flow"

    def execution_blocker(self, input, block, verbose):
        if block:
            return (ExecutionBlocker("Blocked Execution" if verbose else None),)
        return (input,)

FLOW_CONTROL_NODE_CLASS_MAPPINGS = {
    "TestWhileLoopOpen": TestWhileLoopOpen,
    "TestWhileLoopClose": TestWhileLoopClose,
    "TestExecutionBlocker": TestExecutionBlockerNode,
}
FLOW_CONTROL_NODE_DISPLAY_NAME_MAPPINGS = {
    "TestWhileLoopOpen": "While Loop Open",
    "TestWhileLoopClose": "While Loop Close",
    "TestExecutionBlocker": "Execution Blocker",
}

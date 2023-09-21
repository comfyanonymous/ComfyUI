from comfy.graph_utils import GraphBuilder, is_link
from comfy.graph import ExecutionBlocker

NUM_FLOW_SOCKETS = 5
class WhileLoopOpen:
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
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["FLOW_CONTROL"] + ["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_open"

    CATEGORY = "InversionDemo Nodes/Flow"

    def while_loop_open(self, condition, **kwargs):
        values = []
        for i in range(NUM_FLOW_SOCKETS):
            values.append(kwargs.get("initial_value%d" % i, None))
        return tuple(["stub"] + values)

class WhileLoopClose:
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
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["*"] * NUM_FLOW_SOCKETS)
    RETURN_NAMES = tuple(["value%d" % i for i in range(NUM_FLOW_SOCKETS)])
    FUNCTION = "while_loop_close"

    CATEGORY = "InversionDemo Nodes/Flow"

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
        if not condition:
            # We're done with the loop
            values = []
            for i in range(NUM_FLOW_SOCKETS):
                values.append(kwargs.get("initial_value%d" % i, None))
            return tuple(values)

        # We want to loop
        this_node = dynprompt.get_node(unique_id)
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
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        for i in range(NUM_FLOW_SOCKETS):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node("Recurse" )
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

class ExecutionBlockerNode:
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

    CATEGORY = "InversionDemo Nodes/Flow"

    def execution_blocker(self, input, block, verbose):
        if block:
            return (ExecutionBlocker("Blocked Execution" if verbose else None),)
        return (input,)

FLOW_CONTROL_NODE_CLASS_MAPPINGS = {
    "WhileLoopOpen": WhileLoopOpen,
    "WhileLoopClose": WhileLoopClose,
    "ExecutionBlocker": ExecutionBlockerNode,
}
FLOW_CONTROL_NODE_DISPLAY_NAME_MAPPINGS = {
    "WhileLoopOpen": "While Loop Open",
    "WhileLoopClose": "While Loop Close",
    "ExecutionBlocker": "Execution Blocker",
}

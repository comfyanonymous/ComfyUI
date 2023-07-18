from comfy.graph_utils import GraphBuilder

NUM_FLOW_SOCKETS = 5
class WhileLoopOpen:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("INT", {"default": 1, "min": 0, "max": 1, "step": 1}),
            },
            "optional": {
            },
        }
        for i in range(NUM_FLOW_SOCKETS):
            inputs["optional"]["initial_value%d" % i] = ("*",)
        return inputs

    RETURN_TYPES = tuple(["FLOW_CONTROL"] + ["*"] * NUM_FLOW_SOCKETS)
    FUNCTION = "while_loop_open"

    CATEGORY = "Flow Control"

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
                "flow_control": ("FLOW_CONTROL",),
                "condition": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
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
    FUNCTION = "while_loop_close"

    CATEGORY = "Flow Control"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if isinstance(v, list) and len(v) == 2:
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
        open_node = this_node["inputs"]["flow_control"][0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node(node_id)
            for k, v in original_node["inputs"].items():
                if isinstance(v, list) and len(v) == 2 and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)
        new_open = graph.lookup_node(open_node)
        for i in range(NUM_FLOW_SOCKETS):
            key = "initial_value%d" % i
            new_open.set_input(key, kwargs.get(key, None))
        my_clone = graph.lookup_node(unique_id)
        result = map(lambda x: my_clone.out(x), range(NUM_FLOW_SOCKETS))
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }

FLOW_CONTROL_NODE_CLASS_MAPPINGS = {
    "WhileLoopOpen": WhileLoopOpen,
    "WhileLoopClose": WhileLoopClose,
}
FLOW_CONTROL_NODE_DISPLAY_NAME_MAPPINGS = {
    "WhileLoopOpen": "While Loop Open",
    "WhileLoopClose": "While Loop Close",
}

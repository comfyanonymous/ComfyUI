import nodes

from comfy.graph_utils import is_link

class DynamicPrompt:
    def __init__(self, original_prompt):
        # The original prompt provided by the user
        self.original_prompt = original_prompt
        # Any extra pieces of the graph created during execution
        self.ephemeral_prompt = {}
        self.ephemeral_parents = {}
        self.ephemeral_display = {}

    def get_node(self, node_id):
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        return None

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id

    def get_real_node_id(self, node_id):
        while node_id in self.ephemeral_parents:
            node_id = self.ephemeral_parents[node_id]
        return node_id

    def get_parent_node_id(self, node_id):
        return self.ephemeral_parents.get(node_id, None)

    def get_display_node_id(self, node_id):
        while node_id in self.ephemeral_display:
            node_id = self.ephemeral_display[node_id]
        return node_id

    def all_node_ids(self):
        return set(self.original_prompt.keys()).union(set(self.ephemeral_prompt.keys()))

def get_input_info(class_def, input_name):
    valid_inputs = class_def.INPUT_TYPES()
    input_info = None
    input_category = None
    if "required" in valid_inputs and input_name in valid_inputs["required"]:
        input_category = "required"
        input_info = valid_inputs["required"][input_name]
    elif "optional" in valid_inputs and input_name in valid_inputs["optional"]:
        input_category = "optional"
        input_info = valid_inputs["optional"][input_name]
    elif "hidden" in valid_inputs and input_name in valid_inputs["hidden"]:
        input_category = "hidden"
        input_info = valid_inputs["hidden"][input_name]
    if input_info is None:
        return None, None, None
    input_type = input_info[0]
    if len(input_info) > 1:
        extra_info = input_info[1]
    else:
        extra_info = {}
    return input_type, input_category, extra_info

class TopologicalSort:
    def __init__(self, dynprompt):
        self.dynprompt = dynprompt
        self.pendingNodes = {}
        self.blockCount = {} # Number of nodes this node is directly blocked by
        self.blocking = {} # Which nodes are blocked by this node

    def get_input_info(self, unique_id, input_name):
        class_type = self.dynprompt.get_node(unique_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return get_input_info(class_def, input_name)

    def make_input_strong_link(self, to_node_id, to_input):
        inputs = self.dynprompt.get_node(to_node_id)["inputs"]
        if to_input not in inputs:
            raise Exception("Node %s says it needs input %s, but there is no input to that node at all" % (to_node_id, to_input))
        value = inputs[to_input]
        if not is_link(value):
            raise Exception("Node %s says it needs input %s, but that value is a constant" % (to_node_id, to_input))
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        self.add_node(from_node_id)
        if to_node_id not in self.blocking[from_node_id]:
            self.blocking[from_node_id][to_node_id] = {}
            self.blockCount[to_node_id] += 1
        self.blocking[from_node_id][to_node_id][from_socket] = True

    def add_node(self, unique_id, include_lazy=False, subgraph_nodes=None):
        if unique_id in self.pendingNodes:
            return
        self.pendingNodes[unique_id] = True
        self.blockCount[unique_id] = 0
        self.blocking[unique_id] = {}

        inputs = self.dynprompt.get_node(unique_id)["inputs"]
        for input_name in inputs:
            value = inputs[input_name]
            if is_link(value):
                from_node_id, from_socket = value
                if subgraph_nodes is not None and from_node_id not in subgraph_nodes:
                    continue
                input_type, input_category, input_info = self.get_input_info(unique_id, input_name)
                is_lazy = "lazy" in input_info and input_info["lazy"]
                if include_lazy or not is_lazy:
                    self.add_strong_link(from_node_id, from_socket, unique_id)

    def get_ready_nodes(self):
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    def pop_node(self, unique_id):
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]

    def is_empty(self):
        return len(self.pendingNodes) == 0

# ExecutionList implements a topological dissolve of the graph. After a node is staged for execution,
# it can still be returned to the graph after having further dependencies added.
class ExecutionList(TopologicalSort):
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        if self.output_cache.get(from_node_id) is not None:
            # Nothing to do
            return
        super().add_strong_link(from_node_id, from_socket, to_node_id)

    def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None
        available = self.get_ready_nodes()
        if len(available) == 0:
            raise Exception("Dependency cycle detected")
        next_node = available[0]
        # If an output node is available, do that first.
        # Technically this has no effect on the overall length of execution, but it feels better as a user
        # for a PreviewImage to display a result as soon as it can
        # Some other heuristics could probably be used here to improve the UX further.
        for node_id in available:
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE == True:
                next_node = node_id
                break
        self.staged_node_id = next_node
        return self.staged_node_id

    def unstage_node_execution(self):
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        self.staged_node_id = None

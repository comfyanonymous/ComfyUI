import os
from __future__ import annotations
from typing import Type, Literal, Optional

import nodes
from comfy_execution.graph_utils import is_link
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, InputTypeOptions

# Optional debug flag: set `MAGIX_DEBUG=1` in your env to see batch picks.
_ENABLE_MAGIX_LOGS = os.getenv("MAGIX_DEBUG", "0") == "1"

class DependencyCycleError(Exception):
    pass

class NodeInputError(Exception):
    pass

class NodeNotFoundError(Exception):
    pass

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
        raise NodeNotFoundError(f"Node {node_id} not found")

    def has_node(self, node_id):
        return node_id in self.original_prompt or node_id in self.ephemeral_prompt

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

    def get_original_prompt(self):
        return self.original_prompt

def get_input_info(
    class_def: Type[ComfyNodeABC],
    input_name: str,
    valid_inputs: InputTypeDict | None = None
) -> tuple[str, Literal["required", "optional", "hidden"], InputTypeOptions] | tuple[None, None, None]:
    """Get the input type, category, and extra info for a given input name.

    Arguments:
        class_def: The class definition of the node.
        input_name: The name of the input to get info for.
        valid_inputs: The valid inputs for the node, or None to use the class_def.INPUT_TYPES().

    Returns:
        tuple[str, str, dict] | tuple[None, None, None]: The input type, category, and extra info for the input name.
    """

    valid_inputs = valid_inputs or class_def.INPUT_TYPES()
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
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but there is no input to that node at all")
        value = inputs[to_input]
        if not is_link(value):
            raise NodeInputError(f"Node {to_node_id} says it needs input {to_input}, but that value is a constant")
        from_node_id, from_socket = value
        self.add_strong_link(from_node_id, from_socket, to_node_id)

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        if not self.is_cached(from_node_id):
            self.add_node(from_node_id)
            if to_node_id not in self.blocking[from_node_id]:
                self.blocking[from_node_id][to_node_id] = {}
                self.blockCount[to_node_id] += 1
            self.blocking[from_node_id][to_node_id][from_socket] = True

    def add_node(self, node_unique_id, include_lazy=False, subgraph_nodes=None):
        node_ids = [node_unique_id]
        links = []

        while len(node_ids) > 0:
            unique_id = node_ids.pop()
            if unique_id in self.pendingNodes:
                continue

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
                    _, _, input_info = self.get_input_info(unique_id, input_name)
                    is_lazy = input_info is not None and "lazy" in input_info and input_info["lazy"]
                    if (include_lazy or not is_lazy) and not self.is_cached(from_node_id):
                        node_ids.append(from_node_id)
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    def is_cached(self, node_id):
        return False

    def get_ready_nodes(self):
        return [node_id for node_id in self.pendingNodes if self.blockCount[node_id] == 0]

    def pop_node(self, unique_id):
        del self.pendingNodes[unique_id]
        for blocked_node_id in self.blocking[unique_id]:
            self.blockCount[blocked_node_id] -= 1
        del self.blocking[unique_id]

    def is_empty(self):
        return len(self.pendingNodes) == 0

class ExecutionList(TopologicalSort):
    """
    A topological scheduler that favours executing many nodes of the same
    `class_type` back-to-back.  The idea is simple:

      • Keep a *current group* (= a class_type).                          ✨
      • While at least one ready node of that group exists, pick from it. ✨
      • When the group is exhausted, choose the next group that has the   ✨
        largest number of simultaneously ready nodes.                     ✨
      • Inside a group, keep the original UX heuristics that prioritise
        early outputs / previews.

    Magix add-on
    ------------
    If the environment variable **MAGIX_DEBUG=1** is set, the scheduler
    prints a one-liner each time it switches batches, e.g.:

        [Magix] 🎯 Switched batch → 'CLIPTextEncode' (5 ready)

    That’s all—no functional changes, no performance tax when disabled.
    """

    # ------------------------------------------------------------------
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id: Optional[str] = None

        # remember which type we are currently batching
        self._current_group_class: Optional[str] = None

    # ------------------------------------------------------------------
    # group selection helpers
    # ------------------------------------------------------------------
    def _pick_largest_group(self, node_list):
        """Return the class_type that has most representatives in `node_list`."""
        counts = {}
        for nid in node_list:
            ctype = self.dynprompt.get_node(nid)["class_type"]
            counts[ctype] = counts.get(ctype, 0) + 1
        # largest group wins – ties are resolved deterministically by name
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def _filter_by_group(self, node_list, group_cls):
        """Keep only nodes that belong to the given class."""
        return [nid for nid in node_list
                if self.dynprompt.get_node(nid)["class_type"] == group_cls]

    # ------------------------------------------------------------------
    # node-picking logic
    # ------------------------------------------------------------------
    def ux_friendly_pick_node(self, node_list):
        """
        Choose which ready node to execute next, honouring the current batch.
        """

        # step 1 – ensure we have a valid *current* group
        if (self._current_group_class is None or
            not any(self.dynprompt.get_node(nid)["class_type"]
                    == self._current_group_class for nid in node_list)):
            # Either first call, or the old batch is finished → pick a new one
            self._current_group_class = self._pick_largest_group(node_list)

            # 🌟  Magix (opt-in) log
            if _ENABLE_MAGIX_LOGS:
                ready_cnt = sum(
                    1 for nid in node_list
                    if self.dynprompt.get_node(nid)["class_type"]
                    == self._current_group_class
                )
                print(f"[Magix] 🎯 Switched batch → "
                      f"'{self._current_group_class}' ({ready_cnt} ready)")

        # candidate set = nodes of the current batch
        candidates = self._filter_by_group(node_list, self._current_group_class)

        # -------------------- original UX heuristics --------------------
        def is_output(node_id):
            class_type = self.dynprompt.get_node(node_id)["class_type"]
            class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
            return getattr(class_def, 'OUTPUT_NODE', False) is True

        # 1️⃣  execute an output node as soon as one appears *within the batch*
        for nid in candidates:
            if is_output(nid):
                return nid

        # 2️⃣  decoder-before-preview pattern (still inside the batch)
        for nid in candidates:
            for blocked in self.blocking[nid]:
                if is_output(blocked):
                    return nid

        for nid in candidates:
            for blocked in self.blocking[nid]:
                for blocked2 in self.blocking[blocked]:
                    if is_output(blocked2):
                        return nid

        # 3️⃣  otherwise just take the first candidate
        return candidates[0]

    # ------------------------------------------------------------------
    # staging / completion plumbing – unchanged except for group bookkeeping
    # ------------------------------------------------------------------
    def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None, None, None

        available = self.get_ready_nodes()
        if not available:
            cycled_nodes = self.get_nodes_in_cycle()
            blamed_node = cycled_nodes[0]
            for nid in cycled_nodes:
                disp = self.dynprompt.get_display_node_id(nid)
                if disp != nid:
                    blamed_node = disp
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            err = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            return None, err, ex

        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None

    def unstage_node_execution(self):
        """Called when an execution turned out to be PENDING."""
        self.staged_node_id = None
        # do *not* clear the current group – the node wasn’t executed

    def complete_node_execution(self):
        """Called after successful (or cached) execution."""
        nid = self.staged_node_id
        assert nid is not None, "complete_node_execution with no node staged"
        self.pop_node(nid)
        self.staged_node_id = None
        # keep `_current_group_class` as-is; it will be updated automatically
        # in `ux_friendly_pick_node` when its batch runs dry

    # ------------------------------------------------------------------
    # cycle detection helper – untouched
    # ------------------------------------------------------------------
    def get_nodes_in_cycle(self):
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        blocked_by = { node_id: {} for node_id in self.pendingNodes }
        for from_node_id in self.blocking:
            for to_node_id in self.blocking[from_node_id]:
                if True in self.blocking[from_node_id][to_node_id].values():
                    blocked_by[to_node_id][from_node_id] = True
        to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        while len(to_remove) > 0:
            for node_id in to_remove:
                for to_node_id in blocked_by:
                    if node_id in blocked_by[to_node_id]:
                        del blocked_by[to_node_id][node_id]
                del blocked_by[node_id]
            to_remove = [node_id for node_id in blocked_by if len(blocked_by[node_id]) == 0]
        return list(blocked_by.keys())


class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
    """
    def __init__(self, message):
        self.message = message


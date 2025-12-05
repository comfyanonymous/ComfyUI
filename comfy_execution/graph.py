# graph.py — grouped/batched scheduler on top of the updated ExecutionList
# Implements model-class batching to reduce device/context swaps while preserving
# the new execution_cache behavior added upstream.


from __future__ import annotations
from typing import Type, Literal, Optional

import os
import nodes
import asyncio
import inspect
from comfy_execution.graph_utils import is_link, ExecutionBlocker
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, InputTypeOptions

# NOTE: ExecutionBlocker code got moved to graph_utils.py to prevent torch being imported too soon during unit tests
ExecutionBlocker = ExecutionBlocker


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
        self.blockCount = {}  # Number of nodes this node is directly blocked by
        self.blocking = {}    # Which nodes are blocked by this node
        self.externalBlocks = 0
        self.unblockedEvent = asyncio.Event()

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
                    if (include_lazy or not is_lazy):
                        if not self.is_cached(from_node_id):
                            node_ids.append(from_node_id)
                        links.append((from_node_id, from_socket, unique_id))

        for link in links:
            self.add_strong_link(*link)

    def add_external_block(self, node_id):
        assert node_id in self.blockCount, "Can't add external block to a node that isn't pending"
        self.externalBlocks += 1
        self.blockCount[node_id] += 1

        def unblock():
            self.externalBlocks -= 1
            self.blockCount[node_id] -= 1
            self.unblockedEvent.set()
        return unblock

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
    ExecutionList implements a topological dissolve of the graph with batching.
    After a node is staged for execution, it can still be returned to the graph
    after having further dependencies added.

    Batching: we favor running nodes of the same class_type back-to-back
    to reduce device/context thrash (e.g., model swaps). Within a batch we still
    apply UX-friendly priorities (output/async early, VAEDecode→preview, etc.).
    """

    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id: Optional[str] = None

        # Upstream execution cache (kept intact)
        self.execution_cache = {}
        self.execution_cache_listeners = {}

        # Batching state
        self._current_group_class: Optional[str] = None

    # ----------------------------- cache ---------------------------------
    def is_cached(self, node_id):
        return self.output_cache.get(node_id) is not None

    def cache_link(self, from_node_id, to_node_id):
        if to_node_id not in self.execution_cache:
            self.execution_cache[to_node_id] = {}
        self.execution_cache[to_node_id][from_node_id] = self.output_cache.get(from_node_id)
        if from_node_id not in self.execution_cache_listeners:
            self.execution_cache_listeners[from_node_id] = set()
        self.execution_cache_listeners[from_node_id].add(to_node_id)

    def get_cache(self, from_node_id, to_node_id):
        if to_node_id not in self.execution_cache:
            return None
        value = self.execution_cache[to_node_id].get(from_node_id)
        if value is None:
            return None
        # Write back to the main cache on touch.
        self.output_cache.set(from_node_id, value)
        return value

    def cache_update(self, node_id, value):
        if node_id in self.execution_cache_listeners:
            for to_node_id in self.execution_cache_listeners[node_id]:
                if to_node_id in self.execution_cache:
                    self.execution_cache[to_node_id][node_id] = value

    def add_strong_link(self, from_node_id, from_socket, to_node_id):
        super().add_strong_link(from_node_id, from_socket, to_node_id)
        self.cache_link(from_node_id, to_node_id)

    # --------------------------- group utils ------------------------------
    def _pick_largest_group(self, node_list):
        """Return the class_type with the most representatives in node_list.
        Ties are resolved deterministically by class name."""
        counts = {}
        for nid in node_list:
            ctype = self.dynprompt.get_node(nid)["class_type"]
            counts[ctype] = counts.get(ctype, 0) + 1
        # max by (count, class_name) for deterministic tie-break
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def _filter_by_group(self, node_list, group_cls):
        """Keep only nodes that belong to the given class."""
        return [nid for nid in node_list if self.dynprompt.get_node(nid)["class_type"] == group_cls]

    # ------------------------- node classification ------------------------
    def _is_output(self, node_id):
        class_type = self.dynprompt.get_node(node_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return getattr(class_def, 'OUTPUT_NODE', False) is True

    def _is_async(self, node_id):
        class_type = self.dynprompt.get_node(node_id)["class_type"]
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        return inspect.iscoroutinefunction(getattr(class_def, class_def.FUNCTION))

    # ------------------------- UX within a batch --------------------------
    def _pick_in_batch_with_ux(self, candidates):
        """
        Original UX heuristics, but applied *within* the current batch.
        """
        # 1) Output nodes ASAP
        for nid in candidates:
            if self._is_output(nid):
                return nid
        # 1b) Async nodes early to overlap
        for nid in candidates:
            if self._is_async(nid):
                return nid
        # 2) decoder-before-preview pattern (within the batch)
        for nid in candidates:
            for blocked in self.blocking[nid]:
                if self._is_output(blocked):
                    return nid
        # 3) VAELoader -> VAEDecode -> preview (within the batch)
        for nid in candidates:
            for blocked in self.blocking[nid]:
                for blocked2 in self.blocking[blocked]:
                    if self._is_output(blocked2):
                        return nid
        # 4) Otherwise, first candidate
        return candidates[0]

    # ------------------------- batch-aware picking ------------------------
    def ux_friendly_pick_node(self, available):
        """
        Choose which ready node to execute next, honoring the current batch.
        When the current batch runs dry, switch to the largest ready group.
        """

        # Ensure current batch is still present; otherwise pick a new largest group.
        has_current = (
            self._current_group_class is not None and
            any(self.dynprompt.get_node(nid)["class_type"] == self._current_group_class for nid in available)
        )
        if not has_current:
            new_group = self._pick_largest_group(available)
            self._current_group_class = new_group

        # Restrict to nodes of the current batch
        candidates = self._filter_by_group(available, self._current_group_class)
        return self._pick_in_batch_with_ux(candidates)

    # --------------------------- staging / run ----------------------------
    async def stage_node_execution(self):
        assert self.staged_node_id is None
        if self.is_empty():
            return None, None, None

        available = self.get_ready_nodes()

        # If nothing ready but there are external blockers, wait for unblocks.
        while len(available) == 0 and self.externalBlocks > 0:
            await self.unblockedEvent.wait()
            self.unblockedEvent.clear()
            available = self.get_ready_nodes()

        if len(available) == 0:
            cycled_nodes = self.get_nodes_in_cycle()
            # Because cycles composed entirely of static nodes are caught during initial validation,
            # we will 'blame' the first node in the cycle that is not a static node.
            blamed_node = cycled_nodes[0]
            for node_id in cycled_nodes:
                display_node_id = self.dynprompt.get_display_node_id(node_id)
                if display_node_id != node_id:
                    blamed_node = display_node_id
                    break
            ex = DependencyCycleError("Dependency cycle detected")
            error_details = {
                "node_id": blamed_node,
                "exception_message": str(ex),
                "exception_type": "graph.DependencyCycleError",
                "traceback": [],
                "current_inputs": []
            }
            return None, error_details, ex

        # Batch-aware pick
        self.staged_node_id = self.ux_friendly_pick_node(available)
        return self.staged_node_id, None, None

    def unstage_node_execution(self):
        # If a node execution resolves to PENDING, return it to the pool
        # but keep the current batch so we continue batching next time.
        assert self.staged_node_id is not None
        self.staged_node_id = None

    def complete_node_execution(self):
        node_id = self.staged_node_id
        self.pop_node(node_id)
        # Maintain current batch; it will switch automatically when empty.
        self.execution_cache.pop(node_id, None)
        self.execution_cache_listeners.pop(node_id, None)
        self.staged_node_id = None

    # ------------------------- cycle detection ----------------------------
    def get_nodes_in_cycle(self):
        # We'll dissolve the graph in reverse topological order to leave only the nodes in the cycle.
        # We're skipping some of the performance optimizations from the original TopologicalSort to keep
        # the code simple (and because having a cycle in the first place is a catastrophic error)
        blocked_by = {node_id: {} for node_id in self.pendingNodes}
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

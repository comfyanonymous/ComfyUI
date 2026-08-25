from comfy_extras.graph_traversal import loop_projection
from comfy_execution.graph_utils import is_link


def close_state(execution_list, node_id):
    state = execution_list.get_projection_state(node_id)
    if state is None:
        state = {}
        execution_list.set_projection_state(node_id, state)
    return state


class ForLoopOpen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start": ("INT", {"default": 0}),
                "end": ("INT", {"default": 4}),
                "increment": ("INT", {"default": 1}),
            },
            "optional": {
                "i_outer": ("INT",),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "execution_list": "EXECUTION_LIST",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("i", "first", "last")
    FUNCTION = "open"
    CATEGORY = "looping"

    def open(self, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        if increment == 0:
            raise ValueError("ForLoopOpen increment must not be 0")

        state = execution_list.get_projection_state(unique_id)
        if state is None:
            projected_nodes, close_nodes = loop_projection(dynprompt, unique_id)
            projected_nodes.difference_update(close_nodes)
            nested_openers = {
                node_id for node_id in projected_nodes
                if dynprompt.get_node(node_id)["class_type"] == "ForLoopOpen"
            }
            nested_links = []
            for node_id in projected_nodes.union(close_nodes):
                for input_name, value in dynprompt.get_node(node_id)["inputs"].items():
                    if not is_link(value) or value[0] not in nested_openers:
                        continue
                    _, _, input_info = execution_list.get_input_info(node_id, input_name)
                    if input_info is None or not input_info.get("lazy", False):
                        nested_links.append((value[0], value[1], node_id))
            projected_nodes.difference_update(nested_openers)
            close_sources = {
                node_id: tuple(dynprompt.get_node(node_id)["inputs"]["value"])
                for node_id in close_nodes
            }
            state = {
                "values": list(range(start, end, increment)),
                "index": -1,
                "projected_nodes": projected_nodes,
                "scheduled_nodes": projected_nodes.intersection(execution_list.pendingNodes),
                "nested_openers": nested_openers.intersection(execution_list.pendingNodes),
                "nested_links": nested_links,
                "invalidated_nodes": projected_nodes.union(nested_openers),
                "close_sources": close_sources,
            }
            execution_list.set_projection_state(unique_id, state)
            execution_list.project_nodes(state["projected_nodes"], state["scheduled_nodes"])
            execution_list.requeue_nodes(close_nodes)
            for node_id, (source_id, source_socket) in close_sources.items():
                node_state = close_state(execution_list, node_id)
                node_state["values"] = []
                node_state["unblock"] = execution_list.add_external_block(node_id)
                if source_id != unique_id:
                    execution_list.cache_link(source_id, unique_id, source_socket)
            state["projected_nodes"] = execution_list.get_projected_nodes(unique_id)
            state["scheduled_nodes"] = execution_list.get_projection_scheduled_nodes(unique_id)
        elif state["index"] >= 0:
            for node_id, (source_id, source_socket) in state["close_sources"].items():
                values = execution_list.get_projection_state(node_id)["values"]
                if source_id == unique_id:
                    values.append(state["opener_outputs"][source_socket])
                else:
                    source = execution_list.get_cache(source_id, unique_id)
                    if source is None:
                        raise RuntimeError(f"Loop Close {node_id} input was not produced during the iteration")
                    values.extend(source.outputs[source_socket])

        state["index"] += 1
        if state["index"] >= len(state["values"]):
            for node_id in state["close_sources"]:
                execution_list.get_projection_state(node_id)["unblock"]()
            execution_list.release_projected_nodes(state["projected_nodes"])
            execution_list.clear_projection_state(unique_id)
            return {"ui": {"text": ("<complete>",)}, "result": (None, False, True)}

        execution_list.requeue_nodes(
            state["scheduled_nodes"].union(state["nested_openers"]),
            state["invalidated_nodes"],
        )
        for link in state["nested_links"]:
            execution_list.add_strong_link(*link)
        for source_id, source_socket in state["close_sources"].values():
            if source_id != unique_id:
                execution_list.cache_link(source_id, unique_id, source_socket)
        execution_list.defer_staged_node()
        value = state["values"][state["index"]]
        outputs = (value, state["index"] == 0, state["index"] == len(state["values"]) - 1)
        state["opener_outputs"] = outputs
        return outputs

    @classmethod
    def IS_CHANGED(cls, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        return float("NaN")


class LoopClose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*", {"lazy": True}),
            },
            "hidden": {
                "execution_list": "EXECUTION_LIST",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("*", "*")
    RETURN_NAMES = ("all_outputs", "last_output")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "close"
    CATEGORY = "looping"

    def check_lazy_status(self, value, execution_list=None, unique_id=None):
        if execution_list.get_projection_state(unique_id) is None:
            return ["value"]
        return []

    def close(self, value, execution_list=None, unique_id=None):
        state = execution_list.get_projection_state(unique_id)
        if state is None or "values" not in state:
            raise ValueError(f"Loop Close {unique_id} does not belong to a For Loop")
        execution_list.clear_projection_state(unique_id)
        values = state["values"]
        return (values, values[-1] if values else None)

    @classmethod
    def IS_CHANGED(cls, value, execution_list=None, unique_id=None):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ForLoopOpen": ForLoopOpen,
    "LoopClose": LoopClose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForLoopOpen": "For Loop",
    "LoopClose": "Loop Close",
}

from comfy_extras.graph_traversal import descendants


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
            projected_nodes = descendants(dynprompt, unique_id)
            state = {
                "values": list(range(start, end, increment)),
                "index": -1,
                "projected_nodes": projected_nodes,
                "scheduled_nodes": projected_nodes.intersection(execution_list.pendingNodes),
            }
            execution_list.set_projection_state(unique_id, state)
            execution_list.project_nodes(state["projected_nodes"], state["scheduled_nodes"])
            state["projected_nodes"] = execution_list.get_projected_nodes(unique_id)
            state["scheduled_nodes"] = execution_list.get_projection_scheduled_nodes(unique_id)

        state["index"] += 1
        if state["index"] >= len(state["values"]):
            execution_list.release_projected_nodes(state["projected_nodes"])
            execution_list.clear_projection_state(unique_id)
            return {"ui": {"text": ("<complete>",)}, "result": (None, False, True)}

        execution_list.requeue_nodes(state["scheduled_nodes"], state["projected_nodes"])
        execution_list.defer_staged_node()
        value = state["values"][state["index"]]
        return (value, state["index"] == 0, state["index"] == len(state["values"]) - 1)

    @classmethod
    def IS_CHANGED(cls, start, end, increment, i_outer=None, dynprompt=None, execution_list=None, unique_id=None):
        return float("NaN")


NODE_CLASS_MAPPINGS = {
    "ForLoopOpen": ForLoopOpen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForLoopOpen": "For Loop",
}

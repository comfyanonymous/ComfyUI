from comfy_api.latest import io
from comfy_extras.graph_traversal import loop_projection
from comfy_execution.graph_utils import is_link
from server import PromptServer


class StartLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        list_item_type = io.MatchType.Template("list_item")
        carried_type = io.MatchType.Template("carried_value")
        return io.Schema(
            node_id="StartLoop",
            display_name="Start Loop",
            category="utilities/looping",
            is_input_list=True,
            inputs=[
                io.DynamicCombo.Input("mode", options=[
                    io.DynamicCombo.Option("simple", [
                        io.Int.Input(
                            "num_iterations",
                            default=4,
                            min=0,
                            tooltip="Number of times to execute the loop body.",
                        ),
                    ]),
                    io.DynamicCombo.Option("For", [
                        io.Int.Input(
                            "start_iteration_index",
                            default=0,
                            tooltip="Index of the first iteration when using For loop mode.",
                        ),
                        io.Int.Input(
                            "max_iteration",
                            default=4,
                            max=0xffffffffffffffff,
                            tooltip="The exclusive stopping value for iteration_index in For mode.",
                        ),
                        io.Int.Input(
                            "step",
                            default=1,
                            min=1,
                            tooltip="The index step size between each iteration when using For loop mode.",
                        ),
                    ]),
                    io.DynamicCombo.Option("List", [
                        io.MatchType.Input(
                            "list",
                            list_item_type,
                            tooltip="List of items the loop iterates on. The loop body executes once per item.",
                        ),
                    ]),
                ], tooltip="The loop iteration mode."),
                io.Int.Input(
                    "parent_iteration",
                    optional=True,
                    force_input=True,
                    tooltip="Connect iteration_index from an outer Start Loop to nest this loop.",
                ),
                io.MatchType.Input(
                    "initial_iteration_value",
                    carried_type,
                    optional=True,
                    tooltip="Value exposed as current_iteration_value on the first iteration.",
                ),
            ],
            outputs=[
                io.Int.Output("iteration_index", tooltip="Index of the current loop iteration."),
                io.Boolean.Output("is_first", tooltip="True during the first iteration of the loop."),
                io.Boolean.Output("is_last", tooltip="True during the last iteration of the loop."),
                io.MatchType.Output(
                    list_item_type,
                    id="list_item",
                    tooltip="Current item from the list when using List mode. None in Simple and For modes.",
                ),
                io.MatchType.Output(
                    carried_type,
                    id="current_iteration_value",
                    tooltip="Loop-carried value for the current iteration: initial_iteration_value on the first iteration, then next_iteration_value from End Loop on each subsequent iteration.",
                ),
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.execution_list, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mode, parent_iteration=None, initial_iteration_value=None):
        selected_mode = mode.get("mode", ["simple"])[0]
        if selected_mode == "simple":
            values = range(mode.get("num_iterations", [4])[0])
            list_items = None
        elif selected_mode == "For":
            step = mode.get("step", [1])[0]
            if step == 0:
                raise ValueError("Start Loop step must not be 0")
            values = range(mode.get("start_iteration_index", [0])[0], mode.get("max_iteration", [4])[0], step)
            list_items = None
        else:
            list_items = mode["list"]
            values = range(len(list_items))

        dynprompt = cls.hidden.dynprompt
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id

        state = execution_list.get_projection_state(unique_id)
        if state is None:
            projected_nodes, close_id = loop_projection(dynprompt, unique_id)
            nested_openers = {
                node_id for node_id in projected_nodes
                if dynprompt.get_node(node_id)["class_type"] == "StartLoop"
            }
            nested_links = []
            for node_id in projected_nodes.union((close_id,)):
                for input_name, value in dynprompt.get_node(node_id)["inputs"].items():
                    if not is_link(value) or value[0] not in nested_openers:
                        continue
                    _, _, input_info = execution_list.get_input_info(node_id, input_name)
                    if input_info is None or not input_info.get("lazy", False):
                        nested_links.append((value[0], value[1], node_id))
            projected_nodes.difference_update(nested_openers)
            nested_ordering_links = [
                (opener_id, loop_projection(dynprompt, opener_id)[1])
                for opener_id in nested_openers
            ]
            close_inputs = dynprompt.get_node(close_id)["inputs"]
            output_value = close_inputs.get("output_value")
            output_source = tuple(output_value) if is_link(output_value) else None
            concat_output_lists = False
            if output_source is not None:
                output_node = dynprompt.get_node(output_source[0])
                nested_output = output_node["inputs"].get("output_value")
                if output_node["class_type"] == "EndLoop" and is_link(nested_output):
                    import nodes
                    source_node = dynprompt.get_node(nested_output[0])
                    source_class = nodes.NODE_CLASS_MAPPINGS.get(source_node["class_type"])
                    if source_class is not None:
                        source_output_is_list = getattr(source_class, "OUTPUT_IS_LIST", ())
                        if nested_output[1] < len(source_output_is_list):
                            concat_output_lists = source_output_is_list[nested_output[1]]
            next_iteration_value = close_inputs.get("next_iteration_value")
            next_source = tuple(next_iteration_value) if is_link(next_iteration_value) else None
            termination_sources = {
                input_name: tuple(source)
                for input_name, source in close_inputs.items()
                if input_name.startswith("termination") and is_link(source)
            }
            close_sources = tuple(termination_sources.values())
            if next_source is not None:
                close_sources = (next_source, *close_sources)
            if output_source is not None:
                close_sources = (output_source, *close_sources)
            internal_source_nodes = {
                source_id for source_id, _ in close_sources
                if source_id != unique_id and source_id in projected_nodes
            }
            external_sources = {
                source for source in close_sources
                if source[0] != unique_id and source[0] not in projected_nodes
            }
            state = {
                "values": values,
                "list_items": list_items,
                "index": -1,
                "carried_value": initial_iteration_value[0] if initial_iteration_value else None,
                "projected_nodes": projected_nodes,
                "scheduled_nodes": projected_nodes.intersection(
                    execution_list.pendingNodes.keys() - execution_list.increment_pending_nodes
                ).union(internal_source_nodes),
                "nested_openers": nested_openers.intersection(
                    execution_list.pendingNodes.keys() - execution_list.increment_pending_nodes
                ),
                "nested_links": nested_links,
                "nested_ordering_links": nested_ordering_links,
                "invalidated_nodes": projected_nodes.union(nested_openers),
                "close_id": close_id,
                "output_source": output_source,
                "concat_output_lists": concat_output_lists,
                "next_source": next_source,
                "termination_sources": termination_sources,
                "internal_sources": {
                    source for source in close_sources
                    if source[0] == unique_id or source[0] in projected_nodes
                },
            }
            execution_list.set_projection_state(unique_id, state)
            state["projected_nodes"], state["scheduled_nodes"] = execution_list.project_nodes(
                state["projected_nodes"], state["scheduled_nodes"]
            )
            execution_list.requeue_nodes((close_id,))
            close_state = {
                "outputs": [],
                "last_output": [],
                "output_width": None,
                "unblock": execution_list.add_external_block(close_id),
            }
            execution_list.set_projection_state(close_id, close_state)
            for source_id, source_socket in external_sources:
                execution_list.add_strong_link(source_id, source_socket, unique_id)
            for source_id, source_socket in state["internal_sources"]:
                if source_id != unique_id:
                    execution_list.cache_link(source_id, unique_id, source_socket)
        elif state["index"] >= 0:
            def source_values(source, input_name):
                source_id, source_socket = source
                if source_id == unique_id:
                    return [state["opener_outputs"][source_socket]]
                cached = execution_list.get_cache(source_id, unique_id)
                if cached is None:
                    raise RuntimeError(
                        f"End Loop {state['close_id']} {input_name} was not produced during the iteration"
                    )
                return cached.outputs[source_socket]

            close_state = execution_list.get_projection_state(state["close_id"])
            if state["output_source"] is not None:
                output_values = source_values(state["output_source"], "output_value")
                if close_state["output_width"] is None:
                    close_state["output_width"] = len(output_values)
                    if len(output_values) > 1:
                        close_state["outputs"] = [[] for _ in output_values]
                elif close_state["output_width"] != len(output_values):
                    raise RuntimeError(
                        f"End Loop {state['close_id']} output_value changed list length between iterations"
                    )
                if len(output_values) == 1:
                    close_state["outputs"].append(output_values[0])
                else:
                    for outputs, output in zip(close_state["outputs"], output_values):
                        if state["concat_output_lists"] and isinstance(output, list):
                            outputs.extend(output)
                        else:
                            outputs.append(output)
                close_state["last_output"] = output_values

            state["carried_value"] = None
            if state["next_source"] is not None:
                next_values = source_values(state["next_source"], "next_iteration_value")
                if not next_values:
                    raise RuntimeError(
                        f"End Loop {state['close_id']} next_iteration_value did not produce a value"
                    )
                state["carried_value"] = next_values[0]
            for input_name, source in state["termination_sources"].items():
                source_values(source, input_name)

        state["index"] += 1
        if state["index"] >= len(state["values"]):
            if not state["values"]:
                PromptServer.instance.send_progress_text("Iteration 0 / 0", unique_id)
            execution_list.get_projection_state(state["close_id"])["unblock"]()
            execution_list.release_projected_nodes()
            execution_list.clear_projection_state(unique_id)
            return io.NodeOutput(None, False, True, None, None)

        execution_list.requeue_nodes(
            state["scheduled_nodes"].union(state["nested_openers"]),
            state["invalidated_nodes"].union(state["projected_nodes"]),
        )
        for link in state["nested_links"]:
            if link[2] in execution_list.pendingNodes:
                execution_list.add_strong_link(*link)
        for opener_id, nested_close_id in state["nested_ordering_links"]:
            if nested_close_id in execution_list.pendingNodes:
                execution_list.add_ordering_link(opener_id, nested_close_id)
        for source_id, source_socket in state["internal_sources"]:
            if source_id != unique_id:
                execution_list.cache_link(source_id, unique_id, source_socket)
        execution_list.defer_staged_node()
        value = state["values"][state["index"]]
        list_item = state["list_items"][state["index"]] if state["list_items"] is not None else None
        outputs = (
            value,
            state["index"] == 0,
            state["index"] == len(state["values"]) - 1,
            list_item,
            state["carried_value"],
        )
        state["opener_outputs"] = outputs
        PromptServer.instance.send_progress_text(
            f"Iteration {state['index'] + 1} / {len(state['values'])}",
            unique_id,
        )
        return io.NodeOutput(*outputs)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")


class EndLoop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        output_type = io.MatchType.Template("output_value")
        carried_type = io.MatchType.Template("carried_value")
        terminations = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input(
                "termination",
                tooltip="Connect a preview or side-effect output that must execute on every iteration. Its value is not returned.",
            ),
            prefix="termination",
            min=0,
            max=50,
        )
        return io.Schema(
            node_id="EndLoop",
            display_name="End Loop",
            category="utilities/looping",
            is_input_list=True,
            inputs=[
                io.MatchType.Input(
                    "output_value",
                    output_type,
                    optional=True,
                    tooltip="Value returned by End Loop. It returns the final iteration or all iterations according to accumulate.",
                ),
                io.MatchType.Input(
                    "next_iteration_value",
                    carried_type,
                    optional=True,
                    tooltip="Value sent from the End Loop node back to the Start Loop for the next iteration. The value is passed as current_iteration_value on the Start Loop node.",
                ),
                io.Boolean.Input(
                    "accumulate",
                    default=False,
                    tooltip="Return output_value from every iteration when enabled; otherwise return only the final iteration.",
                ),
                io.Autogrow.Input(
                    "terminations",
                    template=terminations,
                    optional=True,
                    tooltip="Connect an output node that must execute on every iteration. Its value is not returned.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    output_type,
                    id="outputs",
                    is_output_list=True,
                    tooltip="The final iteration's output_value, or the values accumulated across iterations when accumulate is enabled.",
                ),
            ],
            hidden=[io.Hidden.execution_list, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, accumulate, output_value=None, next_iteration_value=None, **terminations):
        execution_list = cls.hidden.execution_list
        unique_id = cls.hidden.unique_id
        state = execution_list.get_projection_state(unique_id)
        if state is None or "outputs" not in state:
            raise ValueError(f"End Loop {unique_id} does not belong to a Start Loop")
        execution_list.clear_projection_state(unique_id)
        values = []
        if state["output_width"] is not None:
            values = state["outputs"] if accumulate[0] else state["last_output"]
        return io.NodeOutput(values)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("NaN")

NODE_CLASS_MAPPINGS = {
    "StartLoop": StartLoop,
    "EndLoop": EndLoop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StartLoop": "Start Loop",
    "EndLoop": "End Loop",
}

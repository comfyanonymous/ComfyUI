import torch
from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_api.latest import ComfyExtension, io
from comfy.nested_tensor import NestedTensor
import comfy.utils


NUM_FLOW_SOCKETS = 5

def _is_nested(x):
    return isinstance(x, NestedTensor)

def _temporal_frame_count(samples):
    """Count temporal frames from a samples tensor or NestedTensor."""
    if _is_nested(samples):
        return samples.tensors[0].shape[2]  # count from first sub-tensor (video)
    if samples.ndim == 5:
        return samples.shape[2]  # (B,C,T,H,W)
    return samples.shape[0]  # (B,C,H,W) — batch count

def _accum_count(accum, blend_overlap=0):
    """Count items in an accumulation, handling tensors (Image/Mask) and dicts (Latent/Video Latent).
    blend_overlap: frames consumed per seam by post-loop crossfade blending (fade modes)."""
    if not isinstance(accum, dict) or "accum" not in accum:
        return 0
    total = 0
    for item in accum["accum"]:
        if isinstance(item, dict):
            total += _temporal_frame_count(item["samples"])
        else:
            total += item.shape[0]  # IMAGE/MASK: count batch items
    if blend_overlap > 0 and len(accum["accum"]) > 1:
        total -= (len(accum["accum"]) - 1) * blend_overlap
    return total


class TensorLoopOpen(io.ComfyNode):
    """
    Opens a loop that collects outputs. Supports two modes:
    - iterations: runs a fixed number of iterations
    - total_frames: runs until the target number of frames is accumulated
    Wire flow_control → TensorLoopClose, use `previous_value` as input to your
    generation, and connect the generated output → TensorLoopClose.processed.
    Supports IMAGE, MASK, and LATENT types.
    """
    MATCHTYPE = io.MatchType.Template("data", allowed_types=[io.Image, io.Mask, io.Latent])

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorLoopOpen",
            display_name="Tensor Loop Open",
            category="looping/accumulation",
            inputs=[
                io.DynamicCombo.Input("mode", tooltip="Loop termination mode.", options=[
                    io.DynamicCombo.Option("iterations", [
                        io.Int.Input("iterations", default=4, min=0,
                                     tooltip="Number of loop iterations. Set to 0 to skip looping and pass through initial_value."),
                    ]),
                    io.DynamicCombo.Option("total_frames", [
                        io.Int.Input("total_frames", default=100, min=1,
                                     tooltip="Target number of output frames. The loop continues until this many frames are accumulated."),
                    ]),
                ]),
                io.MatchType.Input("initial_value", template=cls.MATCHTYPE, optional=True,
                                   tooltip="Optional value to use as `previous_value` on the first iteration."),
            ],
            outputs=[
                io.FlowControl.Output("flow_control"),
                io.MatchType.Output(cls.MATCHTYPE, id="previous_value",
                                    tooltip="The value from the previous iteration (or initial_value on first pass)."),
                io.Int.Output("accumulated_count", tooltip="Number of items collected so far (0 on first iteration)."),
                io.Int.Output("current_iteration", tooltip="Current iteration index (1-based)."),
            ],
            hidden=[io.Hidden.unique_id],
            accept_all_inputs=True,
        )


    @classmethod
    def execute(cls, mode: dict, initial_value=None, **kwargs) -> io.NodeOutput:
        unique_id = cls.hidden.unique_id
        state = kwargs.get("initial_value0")  # packed state dict or None on first pass
        if state is not None:
            count = state["count"]
            total_frames_val = state["total_frames"]
            remaining = state["remaining"]
            accum = state["accum"]
            previous_value = state["previous_value"]
            open_node_id = state["open_node_id"]
            blend_overlap = state.get("blend_overlap", 0)
        else:
            selected_mode = mode.get("mode", "iterations")
            count = mode.get("iterations", 4) if selected_mode == "iterations" else 0
            total_frames_val = mode.get("total_frames", 100) if selected_mode == "total_frames" else 0
            remaining = count
            accum = None
            previous_value = initial_value
            open_node_id = unique_id
            blend_overlap = 0

        accumulated_count = _accum_count(accum, blend_overlap)
        # In total_frames mode, count=0 and remaining goes negative each iteration.
        # The math still produces correct 1-based iteration: 0-0+1=1, 0-(-1)+1=2, etc.
        current_iteration = count - remaining + 1
        if total_frames_val > 0:
            comfy.utils.ProgressBar(total_frames_val, node_id=open_node_id).update_absolute(accumulated_count)
        elif count > 0:
            comfy.utils.ProgressBar(count, node_id=open_node_id).update_absolute(count - remaining)
        loop_state = {"remaining": remaining, "accum": accum, "previous_value": previous_value, "count": count, "open_node_id": open_node_id, "total_frames": total_frames_val, "blend_overlap": blend_overlap}
        return io.NodeOutput(loop_state, previous_value, accumulated_count, current_iteration)


def _overlap_option(name, tooltip):
    """DynamicCombo option with an overlap_frames count input."""
    return io.DynamicCombo.Option(name, [
        io.Int.Input("overlap_frames", default=8, min=1, tooltip=tooltip),
    ])


class TensorLoopClose(io.ComfyNode):
    """
    Closes the loop started by TensorLoopOpen.
    Connect:
      - flow_control from TensorLoopOpen
      - processed: the output generated this iteration
    Supports IMAGE, MASK, and LATENT types.
    """
    MATCHTYPE = io.MatchType.Template("data", allowed_types=[io.Image, io.Mask, io.Latent])

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TensorLoopClose",
            display_name="Tensor Loop Close",
            category="looping/accumulation",
            inputs=[
                io.FlowControl.Input("flow_control", raw_link=True),
                io.MatchType.Input("processed", template=cls.MATCHTYPE, raw_link=True, tooltip="Output generated this iteration."),
                io.Boolean.Input("accumulate", default=True,
                                 tooltip="When enabled, collects all iterations into a batch. When disabled, only outputs the final iteration's result."),
                io.DynamicCombo.Input("overlap", tooltip="How to handle duplicate frames where consecutive iterations overlap (e.g. context frames a video model re-generates).\n"
                                                         "- disabled: keep every frame as-is\n"
                                                         "- start/end: cut the duplicates from the start or end of each iteration's output\n"
                                                         "- fade_linear/fade_smooth: crossfade the end of each iteration into the start of the next", options=[
                    io.DynamicCombo.Option("disabled", []),
                    _overlap_option("start", "Number of frames to cut from the START of each iteration's output. Use when the model re-generates its context frames at the start of its output (most common for video continuation). The first generation is always kept whole; only iterations 2+ are trimmed."),
                    _overlap_option("end", "Number of frames to cut from the END of each iteration's output. Use when the model generates look-ahead frames at the end of its output that the next iteration re-generates. The trimmed tail of the final iteration is re-appended."),
                    _overlap_option("fade_linear", "Number of overlapping frames to crossfade: the end of each iteration is blended into the start of the next with a linear ramp. The first generation is kept whole."),
                    _overlap_option("fade_smooth", "Number of overlapping frames to crossfade: the end of each iteration is blended into the start of the next with a smoothstep (ease in/out) ramp. The first generation is kept whole."),
                ]),
                io.Boolean.Input("stop", optional=True, default=False, raw_link=True, force_input=True,
                                 tooltip="Optional early stop signal from inside the loop body. When True, the loop stops after the current iteration regardless of remaining iterations or total_frames target."),
            ],
            outputs=[
                io.MatchType.Output(cls.MATCHTYPE, id="output", tooltip="Accumulated batch or final iteration result, depending on 'accumulate' setting."),
            ],
            enable_expand=True,
        )


    @classmethod
    def execute(cls, flow_control, processed, accumulate=True, overlap=None, stop=False) -> io.NodeOutput:
        graph = GraphBuilder()
        open_id = flow_control[0]
        unpack = graph.node("_ImageAccumStateUnpack", loop_state=[open_id, 0])
        # unpack: 0=remaining, 1=accum, 2=previous_value, 3=accumulated_count, 4=count, 5=open_node_id, 6=total_frames
        sub = graph.node("_IntOperations", operation="subtract", a=unpack.out(0), b=1)

        overlap_mode = "disabled"
        overlap_frames = 0
        if isinstance(overlap, dict):
            overlap_mode = overlap.get("overlap", "disabled")
            overlap_frames = overlap.get("overlap_frames", 0)

        accum_out = unpack.out(1)
        if accumulate:
            to_accum = processed
            if overlap_frames > 0 and overlap_mode != "disabled":
                # The first generation has no preceding chunk, so it is kept whole; only the seams
                # of iterations 2+ are trimmed. start_trim is 0 on iteration 1 (a _BatchOps no-op).
                if overlap_mode == "start":
                    iter_index = graph.node("_IntOperations", a=unpack.out(4), b=unpack.out(0), operation="subtract")
                    not_first = graph.node("_IntOperations", a=iter_index.out(0), b=0, operation=">")
                    start_trim = graph.node("_IntOperations", a=not_first.out(0), b=overlap_frames, operation="multiply")
                    to_accum = graph.node("_BatchOps", batch=processed, operation="trim_start", amount=start_trim.out(0)).out(0)
                elif overlap_mode == "end":
                    to_accum = graph.node("_BatchOps", batch=processed, operation="trim_end", amount=overlap_frames).out(0)
                # fade modes blend the seams post-loop

            accum_out = graph.node("_AccumulateNode", to_add=to_accum, accumulation=accum_out).out(0)

        # Disable total_frames when not accumulating to avoid infinite loops
        blend_overlap = overlap_frames if overlap_mode in ("fade_linear", "fade_smooth") else 0
        pack = graph.node("_ImageAccumStatePack",
            remaining=sub.out(0),
            accum=accum_out,
            previous_value=processed,
            count=unpack.out(4),
            open_node_id=unpack.out(5),
            total_frames=unpack.out(6) if accumulate else 0,
            prev_accumulated_count=unpack.out(3),
            blend_overlap=blend_overlap if accumulate else 0,
        )

        # Optional early stop from loop body
        if is_link(stop):
            condition = graph.node("_ConditionalSelect",
                condition=stop,
                value_if_true=False,
                value_if_false=pack.out(1),
            ).out(0)
        else:
            condition = pack.out(1)

        while_close = graph.node(
            "_WhileLoopClose",
            flow_control=flow_control,
            condition=condition,
            initial_value0=pack.out(0),
        )

        final_unpack = graph.node("_ImageAccumStateUnpack", loop_state=while_close.out(0))
        if accumulate:
            if overlap_mode in ("fade_linear", "fade_smooth") and overlap_frames > 0:
                result = graph.node("_AccumulationToImageBatch",
                    accumulation=final_unpack.out(1),
                    overlap_frames=overlap_frames,
                    overlap_mode=overlap_mode,
                ).out(0)
            else:
                result = graph.node("_AccumulationToImageBatch", accumulation=final_unpack.out(1)).out(0)

            # End mode: re-append the tail that was trimmed from the last iteration
            if overlap_mode == "end" and overlap_frames > 0:
                tail = graph.node("_BatchOps", batch=final_unpack.out(2), operation="keep_end", amount=overlap_frames).out(0)
                result = graph.node("_BatchOps", batch=result, operation="concat", batch_b=tail).out(0)

            result = graph.node("_BatchOps", batch=result, operation="max_count", amount=final_unpack.out(6)).out(0)
        else:
            result = final_unpack.out(2)

        # Bypass: when iterations==0 AND total_frames==0, return initial_value directly
        count_is_zero = graph.node("_IntOperations", a=unpack.out(4), b=0, operation="==")
        tf_is_zero = graph.node("_IntOperations", a=unpack.out(6), b=0, operation="==")
        both_zero = graph.node("_IntOperations", a=count_is_zero.out(0), b=tf_is_zero.out(0), operation="multiply")
        result = graph.node("_ConditionalSelect",
            condition=both_zero.out(1),
            value_if_true=unpack.out(2),
            value_if_false=result,
        ).out(0)

        return io.NodeOutput(result, expand=graph.finalize())

# Internal helper nodes — dev only, hidden from the node menu.

class _AccumulateNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_AccumulateNode",
            display_name="Accumulate",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.AnyType.Input("to_add"),
                io.Accumulation.Input("accumulation", optional=True),
            ],
            outputs=[
                io.Accumulation.Output(),
            ],
        )

    @classmethod
    def execute(cls, to_add, accumulation=None) -> io.NodeOutput:
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return io.NodeOutput({"accum": value})

class _WhileLoopOpen(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_WhileLoopOpen",
            display_name="While Loop Open",
            category="looping",
            is_dev_only=True,
            inputs=[
                io.Boolean.Input("condition", default=True),
                *[io.AnyType.Input(f"initial_value{i}", optional=True) for i in range(NUM_FLOW_SOCKETS)],
            ],
            outputs=[
                io.FlowControl.Output("flow_control", display_name="FLOW_CONTROL"),
                *[io.AnyType.Output(f"value{i}") for i in range(NUM_FLOW_SOCKETS)],
            ],
            accept_all_inputs=True,
        )

    @classmethod
    def execute(cls, condition: bool, **kwargs) -> io.NodeOutput:
        values = [kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)]
        return io.NodeOutput("stub", *values)

class _WhileLoopClose(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_WhileLoopClose",
            display_name="While Loop Close",
            category="looping",
            is_dev_only=True,
            inputs=[
                io.FlowControl.Input("flow_control", raw_link=True),
                io.Boolean.Input("condition", force_input=True),
                *[io.AnyType.Input(f"initial_value{i}", optional=True) for i in range(NUM_FLOW_SOCKETS)],
            ],
            outputs=[
                *[io.AnyType.Output(f"value{i}") for i in range(NUM_FLOW_SOCKETS)],
            ],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
            enable_expand=True,
            accept_all_inputs=True,
        )

    @staticmethod
    def _explore_dependencies(node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    _WhileLoopClose._explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    @staticmethod
    def _collect_contained(node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                _WhileLoopClose._collect_contained(child_id, upstream, contained)

    @classmethod
    def execute(cls, flow_control, condition: bool, **kwargs) -> io.NodeOutput:
        dynprompt = cls.hidden.dynprompt
        unique_id = cls.hidden.unique_id
        values = [kwargs.get(f"initial_value{i}", None) for i in range(NUM_FLOW_SOCKETS)]

        if not condition: # Done with the loop — return current values
            return io.NodeOutput(*values)

        # Build the graph expansion for the next loop iteration
        upstream = {}
        cls._explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        open_node = flow_control[0]
        cls._collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        # Use "Recurse" for this node's clone to avoid exponential name growth
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
            new_open.set_input(f"initial_value{i}", values[i])

        my_clone = graph.lookup_node("Recurse")
        result = tuple(my_clone.out(x) for x in range(NUM_FLOW_SOCKETS))
        return io.NodeOutput(*result, expand=graph.finalize())


class _IntOperations(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_IntOperations",
            display_name="Int Operations",
            category="looping/logic",
            is_dev_only=True,
            inputs=[
                io.Int.Input("a", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Int.Input("b", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Combo.Input("operation", options=[
                    "add", "subtract", "multiply", "divide", "modulo", "power",
                    "==", "!=", "<", ">", "<=", ">=",
                ]),
            ],
            outputs=[
                io.Int.Output(),
                io.Boolean.Output(),
            ],
        )

    OPS = {
        "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b, "divide": lambda a, b: a // b if b else 0,
        "modulo": lambda a, b: a % b if b else 0, "power": lambda a, b: a ** b,
        "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
        "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
    }

    @classmethod
    def execute(cls, a: int, b: int, operation: str) -> io.NodeOutput:
        result = cls.OPS[operation](a, b)
        return io.NodeOutput(int(result), bool(result))

def _get_frame_count(x, dim):
    """Return the size along the given dimension."""
    if isinstance(x, dict) and "samples" in x:
        s = x["samples"]
        return s.tensors[0].shape[dim] if _is_nested(s) else s.shape[dim]
    return x.shape[dim]

def _get_cat_dim(x):
    """Return the concatenation dimension for a tensor: dim 2 for 5D video, dim 0 for everything else."""
    if isinstance(x, dict):
        s = x["samples"]
        if _is_nested(s):
            return 2  # NestedTensor sub-tensors use temporal dim
        return 2 if s.ndim == 5 else 0
    return 0  # IMAGE/MASK: batch dim

def _slice_tensor(x, dim, start=None, end=None):
    """Slice a tensor, latent dict, or NestedTensor along the given dim."""
    if isinstance(x, dict) and "samples" in x:
        result = x.copy()
        samples = x["samples"]
        if _is_nested(samples):
            sliced = []
            for t in samples.tensors:
                sliced.append(t[(slice(None),) * dim + (slice(start, end),)])
            result["samples"] = NestedTensor(sliced)
            mask = x.get("noise_mask")
            if mask is not None and _is_nested(mask):
                mask_sliced = []
                for t in mask.tensors:
                    mask_sliced.append(t[(slice(None),) * dim + (slice(start, end),)])
                result["noise_mask"] = NestedTensor(mask_sliced)
        else:
            result["samples"] = samples[(slice(None),) * dim + (slice(start, end),)]
            mask = x.get("noise_mask")
            if mask is not None and mask.ndim == samples.ndim:
                result["noise_mask"] = mask[(slice(None),) * dim + (slice(start, end),)]
        return result
    else:
        return x[(slice(None),) * dim + (slice(start, end),)]

def _concat_tensor(a, b, dim):
    """Concatenate two tensors, latent dicts, or NestedTensors along the given dim."""
    if isinstance(a, dict) and "samples" in a:
        result = a.copy()
        sa, sb = a["samples"], b["samples"]
        if _is_nested(sa):
            result["samples"] = NestedTensor([torch.cat([ta, tb], dim=dim) for ta, tb in zip(sa.tensors, sb.tensors)])
            ma, mb = a.get("noise_mask"), b.get("noise_mask")
            if ma is not None and mb is not None and _is_nested(ma) and _is_nested(mb):
                result["noise_mask"] = NestedTensor([torch.cat([ta, tb], dim=dim) for ta, tb in zip(ma.tensors, mb.tensors)])
            elif ma is not None and _is_nested(ma):
                # Per-frame mask can't cover the concatenated frames — drop it
                result.pop("noise_mask", None)
        else:
            result["samples"] = torch.cat([sa, sb], dim=dim)
            ma, mb = a.get("noise_mask"), b.get("noise_mask")
            if ma is not None and mb is not None and ma.ndim == sa.ndim and mb.ndim == sb.ndim:
                result["noise_mask"] = torch.cat([ma, mb], dim=dim)
            elif ma is not None and ma.ndim == sa.ndim:
                # Per-frame mask can't cover the concatenated frames — drop it
                result.pop("noise_mask", None)
        return result
    else:
        return torch.cat([a, b], dim=dim)

def _blend_overlap(items, overlap_frames, mode):
    """Concatenate items with crossfade blending in the overlap regions."""
    if not items:
        return None
    dim = _get_cat_dim(items[0])
    # Clamp overlap to the smallest item size to avoid slicing beyond bounds
    min_frames = min(_get_frame_count(item, dim) for item in items)
    overlap_frames = min(overlap_frames, min_frames - 1) if min_frames > 1 else 0
    if overlap_frames <= 0:
        # Nothing to blend — fall through to simple concat
        result = items[0]
        for item in items[1:]:
            result = _concat_tensor(result, item, dim)
        return result
    t = torch.linspace(0, 1, overlap_frames)
    if mode == "fade_smooth":
        t = t * t * (3 - 2 * t)

    result = items[0]
    for i in range(1, len(items)):
        prev_tail = _slice_tensor(result, dim, start=-overlap_frames)
        curr_head = _slice_tensor(items[i], dim, end=overlap_frames)
        curr_rest = _slice_tensor(items[i], dim, start=overlap_frames)
        result_base = _slice_tensor(result, dim, end=-overlap_frames)

        # Lerp: blended = prev_tail * (1-w) + curr_head * w
        if isinstance(prev_tail, dict):
            blended = prev_tail.copy()
            ps, cs = prev_tail["samples"], curr_head["samples"]
            if _is_nested(ps):
                blended_tensors = []
                for pt, ch in zip(ps.tensors, cs.tensors):
                    w = t.to(pt.device).reshape([1] * dim + [overlap_frames] + [1] * (pt.ndim - dim - 1))
                    blended_tensors.append(pt * (1 - w) + ch * w)
                blended["samples"] = NestedTensor(blended_tensors)
            else:
                w = t.to(ps.device).reshape([1] * dim + [overlap_frames] + [1] * (ps.ndim - dim - 1))
                blended["samples"] = ps * (1 - w) + cs * w
        else:
            w = t.to(prev_tail.device).reshape([1] * dim + [overlap_frames] + [1] * (prev_tail.ndim - dim - 1))
            blended = prev_tail * (1 - w) + curr_head * w

        result = _concat_tensor(_concat_tensor(result_base, blended, dim), curr_rest, dim)
    return result


class _AccumulationToImageBatch(io.ComfyNode):
    """Concatenates an ACCUMULATION of IMAGE/MASK tensors or LATENT dicts into a single batch."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_AccumulationToImageBatch",
            display_name="Accumulation to Batch",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.Accumulation.Input("accumulation"),
                io.Int.Input("overlap_frames", default=0, min=0),
                io.Combo.Input("overlap_mode", options=["disabled", "fade_linear", "fade_smooth"], default="disabled"),
            ],
            outputs=[io.AnyType.Output("result")],
        )

    @classmethod
    def execute(cls, accumulation, overlap_frames=0, overlap_mode="disabled") -> io.NodeOutput:
        items = accumulation["accum"]
        if not items:
            return io.NodeOutput(None)

        # Fade modes: blend overlap regions between consecutive items
        if overlap_mode in ("fade_linear", "fade_smooth") and overlap_frames > 0 and len(items) > 1:
            return io.NodeOutput(_blend_overlap(items, overlap_frames, overlap_mode))

        # Standard concatenation (no overlap or start/end trim was handled per-iteration)
        if isinstance(items[0], dict):
            samples = items[0]["samples"]
            if _is_nested(samples):
                num_sub = len(samples.tensors)
                catted = []
                for i in range(num_sub):
                    catted.append(torch.cat([item["samples"].tensors[i] for item in items], dim=2))
                result = items[0].copy()
                result["samples"] = NestedTensor(catted)
                result.pop("noise_mask", None)
                # Only keep masks when every item has one, otherwise the mask wouldn't cover all frames
                masks = [item.get("noise_mask") for item in items]
                if all(m is not None and _is_nested(m) for m in masks):
                    mask_catted = []
                    for i in range(len(masks[0].tensors)):
                        mask_catted.append(torch.cat([m.tensors[i] for m in masks], dim=2))
                    result["noise_mask"] = NestedTensor(mask_catted)
                return io.NodeOutput(result)
            elif samples.ndim == 5:
                result = items[0].copy()
                result["samples"] = torch.cat([item["samples"] for item in items], dim=2)
                result.pop("noise_mask", None)
                masks = [item.get("noise_mask") for item in items]
                if all(m is not None and m.ndim == 5 for m in masks):
                    result["noise_mask"] = torch.cat(masks, dim=2)
                return io.NodeOutput(result)
            else:
                # Image latent — batch along dim 0
                result = items[0].copy()
                result["samples"] = torch.cat([item["samples"] for item in items], dim=0)
                result.pop("noise_mask", None)
                masks = [item.get("noise_mask") for item in items]
                if all(m is not None and m.ndim == samples.ndim for m in masks):
                    result["noise_mask"] = torch.cat(masks, dim=0)
                return io.NodeOutput(result)
        else:
            return io.NodeOutput(torch.cat(items, dim=0))


class _ConditionalSelect(io.ComfyNode):
    """Returns value_if_true when condition is True, value_if_false otherwise.
    Uses lazy evaluation so only the selected branch is resolved."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_ConditionalSelect",
            display_name="Conditional Select",
            category="looping/logic",
            is_dev_only=True,
            inputs=[
                io.Boolean.Input("condition"),
                io.AnyType.Input("value_if_true", lazy=True),
                io.AnyType.Input("value_if_false", lazy=True),
            ],
            outputs=[io.AnyType.Output("result")],
        )

    @classmethod
    def check_lazy_status(cls, condition, value_if_true=None, value_if_false=None) -> list[str]:
        # Unevaluated lazy inputs arrive as None
        if condition and value_if_true is None:
            return ["value_if_true"]
        if not condition and value_if_false is None:
            return ["value_if_false"]
        return []

    @classmethod
    def execute(cls, condition, **kwargs) -> io.NodeOutput:
        selected = kwargs.get("value_if_true") if condition else kwargs.get("value_if_false")
        return io.NodeOutput(selected)


class _ImageAccumStatePack(io.ComfyNode):
    """Packs loop state into a single dict for TensorLoopOpen's initial_value0."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_ImageAccumStatePack",
            display_name="Image Accum State Pack",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.AnyType.Input("remaining"),
                io.Accumulation.Input("accum", optional=True),
                io.AnyType.Input("previous_value"),
                io.AnyType.Input("count"),
                io.AnyType.Input("open_node_id"),
                io.AnyType.Input("total_frames"),
                io.Int.Input("prev_accumulated_count", default=0),
                io.Int.Input("blend_overlap", default=0),
            ],
            outputs=[
                io.AnyType.Output("loop_state"),
                io.Boolean.Output("should_continue"),
            ],
        )

    @classmethod
    def execute(cls, remaining, accum, previous_value, count, open_node_id, total_frames, prev_accumulated_count=0, blend_overlap=0) -> io.NodeOutput:
        # Effective count: fade modes consume blend_overlap frames per seam in the post-loop blend
        accumulated_count = _accum_count(accum, blend_overlap)

        if total_frames > 0:
            should_continue = accumulated_count < total_frames
            # Bail if the last iteration made no progress — the target is unreachable
            if accumulated_count <= prev_accumulated_count:
                should_continue = False
            comfy.utils.ProgressBar(total_frames, node_id=open_node_id).update_absolute(accumulated_count)
        else:
            should_continue = remaining > 0
            current_iteration = count - remaining
            comfy.utils.ProgressBar(count, node_id=open_node_id).update_absolute(current_iteration)

        return io.NodeOutput(
            {"remaining": remaining, "accum": accum, "previous_value": previous_value, "count": count, "open_node_id": open_node_id, "total_frames": total_frames, "blend_overlap": blend_overlap},
            should_continue,
        )


class _ImageAccumStateUnpack(io.ComfyNode):
    """Unpacks loop_state from TensorLoopOpen."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_ImageAccumStateUnpack",
            display_name="Image Accum State Unpack",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[io.AnyType.Input("loop_state")],
            outputs=[
                io.Int.Output("remaining"),
                io.Accumulation.Output("accumulation"),
                io.AnyType.Output("previous_value"),
                io.Int.Output("accumulated_count"),
                io.Int.Output("count"),
                io.AnyType.Output("open_node_id"),
                io.Int.Output("total_frames"),
            ],
        )

    @classmethod
    def execute(cls, loop_state) -> io.NodeOutput:
        remaining = loop_state["remaining"]
        accum = loop_state["accum"]
        previous_value = loop_state["previous_value"]
        count = loop_state.get("count", 0)
        open_node_id = loop_state.get("open_node_id")
        total_frames = loop_state.get("total_frames", 0)
        accumulated_count = _accum_count(accum, loop_state.get("blend_overlap", 0))
        return io.NodeOutput(remaining, accum, previous_value, accumulated_count, count, open_node_id, total_frames)



class _BatchOps(io.ComfyNode):
    """Batch slicing, trimming, and concatenation operations."""
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="_BatchOps",
            display_name="Batch Ops",
            category="looping/accumulation",
            is_dev_only=True,
            inputs=[
                io.AnyType.Input("batch"),
                io.Combo.Input("operation", options=["max_count", "trim_start", "trim_end", "keep_end", "concat"]),
                io.Int.Input("amount", default=0, min=0),
                io.AnyType.Input("batch_b", optional=True),
            ],
            outputs=[io.AnyType.Output("output")],
        )

    @classmethod
    def execute(cls, batch, operation, amount=0, batch_b=None) -> io.NodeOutput:
        dim = _get_cat_dim(batch)
        if operation == "concat" and batch_b is not None:
            return io.NodeOutput(_concat_tensor(batch, batch_b, dim))
        if amount <= 0:
            return io.NodeOutput(batch)
        # Clamp to avoid slicing beyond the tensor size
        total = _get_frame_count(batch, dim)
        amount = min(amount, total)
        if operation == "max_count":
            return io.NodeOutput(_slice_tensor(batch, dim, end=amount))
        elif operation == "trim_start":
            return io.NodeOutput(_slice_tensor(batch, dim, start=amount))
        elif operation == "trim_end":
            return io.NodeOutput(_slice_tensor(batch, dim, end=-amount) if amount < total else batch)
        elif operation == "keep_end":
            return io.NodeOutput(_slice_tensor(batch, dim, start=-amount))
        return io.NodeOutput(batch)


class LoopExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TensorLoopOpen,
            TensorLoopClose,
            _WhileLoopOpen,
            _WhileLoopClose,
            _AccumulateNode,
            _IntOperations,
            _AccumulationToImageBatch,
            _ConditionalSelect,
            _ImageAccumStateUnpack,
            _ImageAccumStatePack,
            _BatchOps,
        ]

def comfy_entrypoint():
    return LoopExtension()

"""Convert a frontend-format workflow JSON into the API prompt format.

The ComfyUI frontend stores graphs as litegraph workflow JSON (the format
emitted by `Save (API Format)`'s sibling, `Save`). The backend's `/prompt`
endpoint expects a different, flatter shape called the "prompt format" — the
one persisted into PNG EXIF and emitted by `Save (API Format)`.

Conversion is normally done in the browser by `ComfyApp.graphToPrompt()` in
the frontend repo. That means external automation (Python scripts, batch
runners, queue-management tools) can't easily get from one to the other
without spinning up a browser. This module fills that gap server-side.

Closes #1112.

## Coverage

Handles the common case: standard nodes with a mix of widget-typed and
link-typed inputs, including widget-to-input conversions, BYPASS / NEVER
node modes, and the implicit `control_after_generate` widget that follows
INT seed widgets.

Intentionally **does not** resolve:
  - PrimitiveNode value propagation — should be done in the frontend
  - Reroute pass-through — should be done in the frontend
  - Group-node expansion — only top-level nodes are emitted
  - Custom-node missing from the install — silently skipped, not faked

These are the same ones the frontend handles before calling
graphToPrompt; if a workflow relies on them, run it through the frontend
once to bake them out, save, and convert that.
"""
from __future__ import annotations

from typing import Any

import nodes


# Frontend-only node types that don't exist in NODE_CLASS_MAPPINGS and
# should never make it into the prompt format.
_FRONTEND_ONLY_NODES = frozenset({
    "Reroute", "PrimitiveNode", "Note", "MarkdownNote",
})

# Widget-eligible input types — anything else has to come from a link.
_WIDGET_PRIMITIVE_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})

# Litegraph node-mode constants.
_MODE_NEVER = 2
_MODE_BYPASS = 4


def workflow_to_prompt(workflow: dict) -> dict:
    """Convert a workflow-format dict into prompt-format dict.

    Returns an empty dict if `workflow` has no nodes. Does not raise on
    individual node errors — best-effort: nodes that can't be converted
    are skipped and the rest of the graph is emitted. Run the result
    through the existing /prompt validator to surface structural issues.
    """
    nodes_list = workflow.get("nodes") or []
    links = workflow.get("links") or []

    # Build a (target_node_id, target_slot_index) -> [src_id_str, src_slot] map.
    inbound: dict[tuple[int, int], list] = {}
    for link in links:
        # Link tuples are [link_id, src_id, src_slot, tgt_id, tgt_slot, type].
        # Newer frontend versions sometimes use dict form; tolerate both.
        if isinstance(link, dict):
            src_id = link.get("origin_id")
            src_slot = link.get("origin_slot")
            tgt_id = link.get("target_id")
            tgt_slot = link.get("target_slot")
        else:
            if len(link) < 5:
                continue
            _link_id, src_id, src_slot, tgt_id, tgt_slot = link[:5]
        if None in (src_id, src_slot, tgt_id, tgt_slot):
            continue
        inbound[(int(tgt_id), int(tgt_slot))] = [str(src_id), int(src_slot)]

    prompt: dict[str, dict] = {}
    for node in nodes_list:
        node_id = node.get("id")
        class_type = node.get("type")
        if node_id is None or not class_type:
            continue
        if class_type in _FRONTEND_ONLY_NODES:
            continue
        # Skip muted/bypassed — same semantics as the frontend (these don't
        # ship to the executor at all).
        if node.get("mode") in (_MODE_NEVER, _MODE_BYPASS):
            continue
        cls = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if cls is None:
            # Custom node not installed in this comfy instance. Skipping is
            # safer than emitting a class_type that won't validate.
            continue

        try:
            inp_types = cls.INPUT_TYPES()
        except Exception:
            # Some custom nodes' INPUT_TYPES throw without their runtime
            # context. Skip rather than crash the whole conversion.
            continue

        node_inputs = node.get("inputs") or []
        link_input_names = {inp.get("name") for inp in node_inputs if inp.get("name")}
        slot_idx_by_name = {
            inp.get("name"): i
            for i, inp in enumerate(node_inputs)
            if inp.get("name")
        }

        widget_values = list(node.get("widgets_values") or [])
        widget_idx = 0

        prompt_inputs: dict[str, Any] = {}
        ordered_inputs = list(inp_types.get("required", {}).items()) + \
                         list(inp_types.get("optional", {}).items())

        for input_name, input_def in ordered_inputs:
            input_type = input_def[0] if isinstance(input_def, (list, tuple)) and input_def else input_def

            if input_name in link_input_names:
                slot_idx = slot_idx_by_name[input_name]
                key = (int(node_id), slot_idx)
                if key in inbound:
                    prompt_inputs[input_name] = inbound[key]
                # else: unconnected link slot (e.g. converted-but-unwired);
                # leave absent so /prompt validation flags if required.
                continue

            # Widget input. COMBO inputs are encoded as a list of choices.
            is_combo = isinstance(input_type, list)
            is_widgety = is_combo or (
                isinstance(input_type, str) and input_type in _WIDGET_PRIMITIVE_TYPES
            )
            if not is_widgety:
                # Non-primitive type with no link wired (e.g. unconnected
                # MODEL input). Nothing to emit.
                continue
            if widget_idx >= len(widget_values):
                continue
            prompt_inputs[input_name] = widget_values[widget_idx]
            widget_idx += 1

            # control_after_generate sneaks an extra widget value in after
            # any INT seed widget. The frontend appends it implicitly so the
            # workflow JSON contains one extra entry per seed slot.
            if input_type == "INT" and input_name in ("seed", "noise_seed"):
                if widget_idx < len(widget_values):
                    widget_idx += 1

        prompt[str(node_id)] = {
            "class_type": class_type,
            "inputs": prompt_inputs,
        }

    return prompt

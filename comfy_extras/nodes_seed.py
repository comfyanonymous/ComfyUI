import logging
import time
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io
from server import PromptServer

log = logging.getLogger(__name__)

def _build_adjacency(prompt: dict) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = {nid: set() for nid in prompt}
    for nid, data in prompt.items():
        for val in data.get("inputs", {}).values():
            if isinstance(val, list) and len(val) == 2:
                parent = str(val[0])
                if parent in adj:
                    adj[nid].add(parent)
                    adj[parent].add(nid)
    return adj


def _connected_component(start: str, adj: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(adj.get(node, set()) - visited)
    return visited


def _on_prompt(json_data: dict) -> dict:
    prompt = json_data.get("prompt")
    if not isinstance(prompt, dict):
        return json_data

    # Fast scan — build nothing if GlobalSeedNode isn't in the workflow.
    seed_node_ids = [
        nid for nid, info in prompt.items()
        if info.get("class_type") == "GlobalSeedNode"
    ]
    if not seed_node_ids:
        return json_data

    # Only build the adjacency map if we actually have work to do.
    adj = _build_adjacency(prompt)
    processed: set[str] = set()

    for nid in seed_node_ids:
        inputs = prompt[nid].get("inputs", {})
        master = inputs.get("seed", 0)
        mode = inputs.get("mode", "placeholders only")
        is_global = bool(inputs.get("global", True))
        is_wired = isinstance(inputs.get("seed"), list)
        # this is safety for the case the node seed is connected,
        # and the next lines of code can only work before node execution
        if is_wired and is_global:
            log.warning("GlobalSeedNode [%s]: seed input is wired in global mode, skipping.", nid)
            continue

        master = int(master)
        if not is_global:
            targets = _connected_component(nid, adj)
        else:
            targets = set(prompt.keys())

        for tid in targets:
            if tid in processed:
                continue
            node_inputs = prompt[tid].get("inputs", {})
            if "seed" not in node_inputs:
                continue
            val = node_inputs["seed"]
            if isinstance(val, list):  # live wire, skip
                continue
            if mode == "overwrite all" or val == 0:
                node_inputs["seed"] = master
                log.debug("GlobalSeedNode [%s]: %s.seed → %d", nid, tid, master)
                processed.add(tid)

    return json_data


class GlobalSeedNode(io.ComfyNode):
    @classmethod
    @override
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GlobalSeedNode",
            display_name="Global Seed",
            category="utils/seed",
            search_aliases=["seed", "random", "global seed", "master seed"],
            inputs=[
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Master seed value supplied to all reachable nodes.",
                ),
                io.Combo.Input(
                    "mode",
                    options=["placeholders only", "overwrite all"],
                    default="placeholders only",
                    tooltip=(
                        "'placeholders only' only touches nodes whose seed is 0. "
                        "'overwrite all' replaces every static seed in scope."
                    ),
                ),
                io.Boolean.Input(
                    "global",
                    default=True,
                    label_on="global",
                    label_off="local",
                    tooltip=(
                        "When global, all seed inputs in the workflow are set to this value. "
                        "When local, only nodes connected to this node are affected."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="seed"),
            ],
        )

    @classmethod
    @override
    def fingerprint_inputs(cls, seed: int, mode: str, **kwargs) -> str:
        return str(time.time())

    @classmethod
    @override
    def execute(cls, seed: int, mode: str, **kwargs) -> io.NodeOutput:
        return io.NodeOutput(seed)

class SeedExtension(ComfyExtension):
    @override
    async def on_load(self) -> None:
        PromptServer.instance.add_on_prompt_handler(_on_prompt)

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GlobalSeedNode]


async def comfy_entrypoint() -> SeedExtension:
    return SeedExtension()

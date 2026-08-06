from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

COMFY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = COMFY_ROOT / "user/default/workflows/MiniMax-H3_FL2VA_RTX3060_TURBO6_FFN4.json"
LORA = COMFY_ROOT.parents[1] / "Models/Lora/minimax_h3_turbo_4step_ckpt500.safetensors"
OBJECT_INFO_URL = "http://127.0.0.1:8188/object_info"


def main() -> int:
    workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    subgraph = workflow["definitions"]["subgraphs"][0]
    workflow_node_types = {node["type"] for node in subgraph["nodes"]}

    with urllib.request.urlopen(OBJECT_INFO_URL, timeout=30) as response:
        object_info = json.load(response)

    missing_node_types = sorted(workflow_node_types - object_info.keys())
    required_custom_nodes = {
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3TurboSampler",
        "MiniMaxH3ChunkFeedForward",
    }
    missing_custom_nodes = sorted(required_custom_nodes - object_info.keys())

    print(f"server={OBJECT_INFO_URL}")
    print(f"workflow={WORKFLOW}")
    print(f"workflow_nodes={len(subgraph['nodes'])}")
    print(f"workflow_links={len(subgraph['links'])}")
    print(f"missing_node_types={missing_node_types}")
    print(f"missing_custom_nodes={missing_custom_nodes}")
    print(f"lora_exists={LORA.is_file()}")
    print(f"lora_bytes={LORA.stat().st_size if LORA.is_file() else 0}")

    return 1 if missing_node_types or missing_custom_nodes or not LORA.is_file() else 0


if __name__ == "__main__":
    sys.exit(main())

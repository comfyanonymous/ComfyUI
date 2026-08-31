"""Load a safetensors index without concatenating files on disk."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_sharded_state_dict(index_path: str | Path) -> dict[str, torch.Tensor]:
    """Load all tensors referenced by a Hugging Face safetensors index.

    ``safetensors.torch.load_file`` uses memory mapping. Keeping the returned
    tensors in one dictionary therefore avoids an additional host-memory copy
    while presenting the normal state-dict interface expected by ComfyUI.
    """

    path = Path(index_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid or empty weight_map in {path}")

    by_shard: dict[str, list[str]] = {}
    for key, shard_name in weight_map.items():
        if not isinstance(key, str) or not isinstance(shard_name, str):
            raise ValueError(f"non-string weight map entry in {path}")
        by_shard.setdefault(shard_name, []).append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in sorted(by_shard):
        shard_path = (path.parent / shard_name).resolve()
        if shard_path.parent != path.parent:
            raise ValueError(f"shard escapes index directory: {shard_name}")
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        shard = load_file(str(shard_path), device="cpu")
        expected = set(by_shard[shard_name])
        missing = expected - set(shard)
        if missing:
            raise ValueError(
                f"{shard_path} is missing indexed tensors: {sorted(missing)[:8]}"
            )
        for key in expected:
            if key in state_dict:
                raise ValueError(f"duplicate tensor across shards: {key}")
            state_dict[key] = shard[key]

    if set(state_dict) != set(weight_map):
        raise AssertionError("loaded state dict does not match index")
    return state_dict


def load_checkpoint_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load either one safetensors file or a Hugging Face sharded index."""

    checkpoint = Path(path).resolve()
    if checkpoint.name.endswith(".safetensors.index.json"):
        return load_sharded_state_dict(checkpoint)
    if checkpoint.suffix == ".safetensors":
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return load_file(str(checkpoint), device="cpu")
    raise ValueError(
        f"expected .safetensors or .safetensors.index.json, got {checkpoint}"
    )


def component_checkpoint(root: str | Path, component: str) -> Path:
    """Resolve a component without imposing sharded-only storage."""

    component_dir = Path(root).resolve() / component
    index_path = component_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return index_path
    model_path = component_dir / "model.safetensors"
    if model_path.is_file():
        return model_path
    candidates = sorted(
        path
        for path in component_dir.glob("*.safetensors")
        if not (path.name.startswith("model-") and "-of-" in path.name)
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(index_path)
    raise ValueError(
        f"{component_dir} contains multiple single-file checkpoints: {[path.name for path in candidates]}"
    )

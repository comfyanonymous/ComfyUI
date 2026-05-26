import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FILES = [
    ROOT / "comfy/ldm/seedvr/vae.py",
    ROOT / "comfy/sd.py",
    ROOT / "comfy_extras/nodes_seedvr.py",
]
FORBIDDEN_ATTRS = {"original_image_video", "img_dims", "tiled_args"}
FORBIDDEN_KEYS = {
    "sampler_metadata",
    "latent_sidecar_metadata",
    "saved_latent_metadata",
    "workflow_hidden_state",
}
FORBIDDEN_GETSET_KEYS = {"original_image_video", "img_dims", "tiled_args"}


def test_seedvr2_decode_paths_do_not_use_hidden_vae_object_state():
    for path in FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
                pytest.fail(f"{path}: forbidden VAE object state attr {node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"getattr", "setattr", "delattr"} and len(node.args) >= 2:
                    key = node.args[1]
                    if isinstance(key, ast.Constant) and key.value in FORBIDDEN_GETSET_KEYS:
                        pytest.fail(f"{path}: forbidden VAE object state access {key.value}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in FORBIDDEN_ATTRS or node.value in FORBIDDEN_KEYS:
                    pytest.fail(f"{path}: forbidden hidden-state string {node.value}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

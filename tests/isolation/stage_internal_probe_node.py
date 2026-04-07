from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
PROBE_SOURCE_ROOT = COMFYUI_ROOT / "tests" / "isolation" / "internal_probe_node"
PROBE_NODE_NAME = "InternalIsolationProbeNode"

PYPROJECT_CONTENT = """[project]
name = "InternalIsolationProbeNode"
version = "0.0.1"

[tool.comfy.isolation]
can_isolate = true
share_torch = true
"""


def _probe_target_root(comfy_root: Path) -> Path:
    return Path(comfy_root) / "custom_nodes" / PROBE_NODE_NAME


def stage_probe_node(comfy_root: Path) -> Path:
    if not PROBE_SOURCE_ROOT.is_dir():
        raise RuntimeError(f"Missing probe source directory: {PROBE_SOURCE_ROOT}")

    target_root = _probe_target_root(comfy_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in PROBE_SOURCE_ROOT.iterdir():
        destination_path = target_root / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)

    (target_root / "pyproject.toml").write_text(PYPROJECT_CONTENT, encoding="utf-8")
    return target_root


@contextmanager
def staged_probe_node() -> Iterator[Path]:
    staging_root = Path(tempfile.mkdtemp(prefix="comfyui_internal_probe_"))
    try:
        yield stage_probe_node(staging_root)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage the internal isolation probe node under an explicit ComfyUI root."
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="Explicit ComfyUI root to stage under. Caller owns cleanup.",
    )
    args = parser.parse_args()

    staged = stage_probe_node(args.target_root)
    sys.stdout.write(f"{staged}\n")

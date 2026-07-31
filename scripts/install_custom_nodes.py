#!/usr/bin/env python3
"""Install pinned ComfyUI custom nodes from a YAML manifest."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml


COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def run(*command: str, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def repository_name(repository: str) -> str:
    path = urlparse(repository).path.rstrip("/")
    name = Path(path).name
    if name.endswith(".git"):
        name = name[:-4]
    if not name or name in {".", ".."}:
        raise ValueError(f"Cannot determine a directory name from repository: {repository}")
    return name


def read_manifest(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("nodes"), list):
        raise ValueError("Manifest must contain a 'nodes' list")

    nodes: list[tuple[str, str]] = []
    for index, node in enumerate(manifest["nodes"], start=1):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a mapping")

        repository = node.get("repo")
        commit = node.get("commit")
        if not isinstance(repository, str) or not repository:
            raise ValueError(f"Node {index} has no valid repository URL")
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            raise ValueError(f"Node {index} must have a full 40-character Git commit")
        nodes.append((repository, commit.lower()))

    return nodes


def install_node(repository: str, commit: str, destination: Path) -> None:
    node_directory = destination / repository_name(repository)
    if node_directory.exists():
        raise FileExistsError(f"Custom-node directory already exists: {node_directory}")

    print(f"Installing {repository} at {commit}", flush=True)
    run("git", "clone", "--filter=blob:none", "--no-checkout", repository, str(node_directory))
    run("git", "checkout", "--detach", commit, cwd=node_directory)

    requirements = node_directory / "requirements.txt"
    if requirements.is_file():
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements),
            cwd=node_directory,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    try:
        nodes = read_manifest(args.manifest)
        args.destination.mkdir(parents=True, exist_ok=True)
        for repository, commit in nodes:
            install_node(repository, commit, args.destination)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

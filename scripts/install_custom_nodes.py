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


def read_manifest(path: Path) -> list[tuple[str, str, list[str], list[Path]]]:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = yaml.safe_load(manifest_file)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("nodes"), list):
        raise ValueError("Manifest must contain a 'nodes' list")

    manifest_directory = path.resolve().parent
    nodes: list[tuple[str, str, list[str], list[Path]]] = []
    for index, node in enumerate(manifest["nodes"], start=1):
        if not isinstance(node, dict):
            raise ValueError(f"Node {index} must be a mapping")

        repository = node.get("repo")
        commit = node.get("commit")
        packages = node.get("pip", [])
        patch_names = node.get("patches", [])
        if not isinstance(repository, str) or not repository:
            raise ValueError(f"Node {index} has no valid repository URL")
        if not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit):
            raise ValueError(f"Node {index} must have a full 40-character Git commit")
        if not isinstance(packages, list) or not all(
            isinstance(package, str) and package for package in packages
        ):
            raise ValueError(f"Node {index} 'pip' value must be a list of packages")
        if not isinstance(patch_names, list) or not all(
            isinstance(patch_name, str) and patch_name for patch_name in patch_names
        ):
            raise ValueError(f"Node {index} 'patches' value must be a list of paths")

        patches = [(manifest_directory / patch_name).resolve() for patch_name in patch_names]
        if any(manifest_directory not in patch.parents for patch in patches):
            raise ValueError(f"Node {index} patch path must stay inside the manifest directory")
        nodes.append((repository, commit.lower(), packages, patches))

    return nodes


def install_node(
    repository: str,
    commit: str,
    packages: list[str],
    patches: list[Path],
    destination: Path,
) -> None:
    node_directory = destination / repository_name(repository)
    if node_directory.exists():
        raise FileExistsError(f"Custom-node directory already exists: {node_directory}")

    print(f"Installing {repository} at {commit}", flush=True)
    run("git", "clone", "--filter=blob:none", "--no-checkout", repository, str(node_directory))
    run("git", "checkout", "--detach", commit, cwd=node_directory)

    for patch in patches:
        if not patch.is_file():
            raise FileNotFoundError(f"Custom-node patch does not exist: {patch}")
        run("git", "apply", "--check", str(patch), cwd=node_directory)
        run("git", "apply", str(patch), cwd=node_directory)

    requirements = (node_directory / "requirements.txt").resolve()
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

    if packages:
        run(sys.executable, "-m", "pip", "install", *packages, cwd=node_directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    try:
        nodes = read_manifest(args.manifest)
        args.destination.mkdir(parents=True, exist_ok=True)
        for repository, commit, packages, patches in nodes:
            install_node(repository, commit, packages, patches, args.destination)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

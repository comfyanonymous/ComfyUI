import configparser
import importlib.metadata
import logging
import os
import platform
import re
import urllib.parse
from datetime import datetime, timezone

import folder_paths
from comfyui_version import __version__

HEADLESS_MARKER = "used_headlessly"


def headless_marker_path() -> str:
    return os.path.join(folder_paths.get_user_directory(), HEADLESS_MARKER)


def used_headlessly() -> bool:
    return os.path.exists(headless_marker_path())


def mark_used_headlessly() -> None:
    """Record, once and permanently, that a prompt was queued without a client_id."""
    path = headless_marker_path()
    if os.path.exists(path):
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()
    except OSError as e:
        logging.warning("Could not write headless marker %s: %s", path, e)


# Installation snapshot in Comfy-Desktop's "comfyui-desktop-2-snapshot" format.
# Local file reads only (no git binary, no network); bad or unreadable data drops
# the affected field or entry instead of raising.

_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
# tomllib is 3.11+ and ComfyUI supports 3.10, so name/version are read from the [project] table text.
_PROJECT_FIELD = re.compile(r"""^\s*(name|version)\s*=\s*(?:"([^"\n]*)"|'([^'\n]*)')""", re.MULTILINE)


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _pyproject_fields(path: str) -> dict[str, str]:
    """First name/version in [project]: from its header line up to the next line starting with `[`."""
    table = []
    inside = False
    for line in (_read_text(path) or "").splitlines():
        if inside and line.startswith("["):
            break
        if inside:
            table.append(line)
        elif line.strip() == "[project]":
            inside = True
    fields = {}
    for m in _PROJECT_FIELD.finditer("\n".join(table)):
        fields.setdefault(m.group(1), m.group(2) if m.group(2) is not None else m.group(3))
    return fields


def _git_dir(path: str) -> str | None:
    dot_git = os.path.join(path, ".git")
    if os.path.isdir(dot_git):
        return dot_git
    m = re.search(r"^gitdir:\s*(.+?)\s*$", _read_text(dot_git) or "", re.MULTILINE)
    return os.path.normpath(os.path.join(path, m.group(1))) if m else None


def _git_head(gitdir: str) -> str | None:
    head = (_read_text(os.path.join(gitdir, "HEAD")) or "").strip()
    if not head.startswith("ref:"):
        return head if _SHA.fullmatch(head) else None
    ref = head[4:].strip()
    parts = ref.split("/")
    if parts[0] != "refs" or ".." in parts:
        return None
    # simplification: a linked worktree's refs live under its `commondir`, which is not followed.
    sha = (_read_text(os.path.join(gitdir, *parts)) or "").strip()
    if _SHA.fullmatch(sha):
        return sha
    for line in (_read_text(os.path.join(gitdir, "packed-refs")) or "").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == ref and _SHA.fullmatch(fields[0]):
            return fields[0]
    return None


def _strip_userinfo(url: str) -> str | None:
    """http(s) URLs keep scheme/host/port/path only; other forms such as git@host:org/repo pass unchanged."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https"):
        return url
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc.rpartition("@")[2], parts.path, "", ""))


def _git_origin_url(gitdir: str) -> str | None:
    config = configparser.ConfigParser(strict=False, interpolation=None)
    try:
        config.read(os.path.join(gitdir, "config"), encoding="utf-8")
        url = config.get('remote "origin"', "url", fallback="").strip()
    except (configparser.Error, UnicodeDecodeError):
        return None
    return _strip_userinfo(url) if url else None


# Shipped inside the ComfyUI checkout itself, so not an installed node (Desktop CORE_FILE_NODES).
_CORE_FILE_NODES = {"websocket_image_save.py"}


def _scan_dir(path: str, name: str, enabled: bool) -> dict | None:
    try:
        entries = os.listdir(path)
    except OSError:  # unreadable or vanished directory
        return None
    if ".tracking" in entries:
        fields = _pyproject_fields(os.path.join(path, "pyproject.toml"))
        node = {"id": fields.get("name") or name, "type": "cnr", "dirName": name, "enabled": enabled}
        if fields.get("version"):
            node["version"] = fields["version"]
        return node
    node = {"id": name, "type": "git", "dirName": name, "enabled": enabled}
    gitdir = _git_dir(path) if ".git" in entries else None
    if gitdir is not None:
        url, commit = _git_origin_url(gitdir), _git_head(gitdir)
        if url:
            node["url"] = url
        if commit:
            node["commit"] = commit
        return node
    # An unmanaged directory is a node only with a pyproject.toml or top-level Python, so stray folders are skipped.
    if "pyproject.toml" in entries or any(e.endswith(".py") for e in entries):
        return node
    return None


def scan_custom_nodes(dirs: list[str]) -> list[dict]:
    """Desktop ScannedNode entries for each custom_nodes dir and the directories in its .disabled/ subfolder (enabled False)."""
    nodes = []
    for base in dirs:
        for root, enabled in ((base, True), (os.path.join(base, ".disabled"), False)):
            try:
                names = sorted(os.listdir(root))
            except OSError:
                continue
            for name in names:
                path = os.path.join(root, name)
                if name.startswith(".") or name == "__pycache__":
                    continue
                if os.path.isdir(path):
                    node = _scan_dir(path, name, enabled)
                    if node is not None:
                        nodes.append(node)
                elif enabled and name.endswith(".py") and name != "__init__.py" and name not in _CORE_FILE_NODES:
                    nodes.append({"id": name[:-3], "type": "file", "dirName": name, "enabled": True})
    return nodes


def count_custom_nodes() -> int:
    return sum(1 for n in scan_custom_nodes(folder_paths.get_folder_paths("custom_nodes")) if n["enabled"])


def pip_packages() -> dict[str, str]:
    packages = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            packages[name] = dist.version
    return packages


MODEL_EXTENSIONS = set(folder_paths.supported_pt_extensions) | {".gguf", ".onnx"}


def list_models() -> list[dict]:
    """Model files as folder name + relative path. Identical relative paths under two roots of one folder collapse to one entry."""
    return [
        {"directory": folder, "filename": filename}
        for folder in list(folder_paths.folder_names_and_paths)
        if folder not in ("custom_nodes", "configs")
        for filename in folder_paths.get_filename_list(folder)
        if os.path.splitext(filename)[1].lower() in MODEL_EXTENSIONS
    ]


def build_snapshot() -> dict:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    tag = "v" + __version__
    return {
        "type": "comfyui-desktop-2-snapshot",
        "version": 2,
        "exportedAt": now,
        "installationName": os.path.basename(os.path.abspath(folder_paths.base_path)),
        "snapshots": [{
            "version": 2,
            "createdAt": now,
            "trigger": "manual",
            "label": None,
            "comfyui": {"ref": tag, "commit": None, "releaseTag": tag, "variant": "comfyui-core", "baseTag": tag},
            "customNodes": scan_custom_nodes(folder_paths.get_folder_paths("custom_nodes")),
            "pipPackages": pip_packages(),
            "pythonVersion": platform.python_version(),
        }],
    }


def collect() -> tuple[dict, list[dict]]:
    return build_snapshot(), list_models()

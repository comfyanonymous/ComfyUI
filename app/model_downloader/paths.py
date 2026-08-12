"""Path resolution for model downloads.

Model identifiers used across the download API are *relative destination
paths* of the form ``<directory>/<filename>`` (e.g. ``loras/my_lora.safetensors``).
This module turns one of those identifiers into an absolute on-disk path
under one of ComfyUI's registered model folders, while rejecting unknown
folders, path traversal, and other ill-formed inputs.
"""

import os
import re
from typing import Optional, Tuple

import folder_paths


# Constrain components so a model_id can never escape its target directory.
# - directory: a single path segment of safe chars
# - filename: a single path segment of safe chars, must end with a model extension
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Destination filename must name a model file (same set as the URL allowlist),
# so a download can't land as e.g. ``foo.txt`` that ComfyUI won't recognise.
_MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pth", ".pt")

# Distinctive temp suffix so the startup orphan-sweep only removes files THIS
# subsystem created — never unrelated ``*.tmp`` files in the model dirs.
_TMP_SUFFIX = ".comfy-download.tmp"


class InvalidModelId(ValueError):
    """Raised when a model_id is syntactically invalid or refers to an
    unknown model folder."""


def parse_model_id(model_id: str) -> Tuple[str, str]:
    """Split ``<directory>/<filename>`` and validate both components.

    Returns ``(directory, filename)``. Raises ``InvalidModelId`` on
    malformed input. Does NOT touch the filesystem.
    """
    if not isinstance(model_id, str) or "/" not in model_id:
        raise InvalidModelId(f"model_id must be '<directory>/<filename>', got {model_id!r}")
    directory, _, filename = model_id.partition("/")
    if "/" in filename or not directory or not filename:
        raise InvalidModelId(f"model_id must be exactly one '/' separator, got {model_id!r}")
    if not _SEGMENT_RE.match(directory):
        raise InvalidModelId(f"invalid directory segment {directory!r}")
    if not _SEGMENT_RE.match(filename):
        raise InvalidModelId(f"invalid filename segment {filename!r}")
    if not filename.endswith(_MODEL_EXTENSIONS):
        raise InvalidModelId(
            f"filename must end with a model extension {_MODEL_EXTENSIONS}, got {filename!r}"
        )
    if directory not in folder_paths.folder_names_and_paths:
        raise InvalidModelId(f"unknown model folder {directory!r}")
    return directory, filename


def resolve_existing(model_id: str) -> Optional[str]:
    """Return the absolute path of an installed model, or None if missing.

    Honours ``extra_model_paths.yaml`` transparently via
    ``folder_paths.get_full_path``.
    """
    directory, filename = parse_model_id(model_id)
    return folder_paths.get_full_path(directory, filename)


def resolve_destination(model_id: str, epoch: int = 0) -> Tuple[str, str]:
    """Return ``(final_path, tmp_path)`` for a download.

    Downloads land at the first registered path for the model's directory
    (the "primary" location). The temp sibling is the write target, atomically
    renamed onto ``final_path`` on success.

    ``tmp_path`` embeds the session ``epoch`` so a cancel+retry of the same
    model never shares a temp path between the old (cancelling) worker and the
    new attempt — otherwise the old worker's rollback could delete the new
    worker's in-progress file. The distinctive suffix scopes the orphan sweep.
    """
    directory, filename = parse_model_id(model_id)
    roots = folder_paths.get_folder_paths(directory)
    if not roots:
        raise InvalidModelId(f"no on-disk path registered for folder {directory!r}")
    root = roots[0]
    final_path = os.path.join(root, filename)
    tmp_path = f"{final_path}.{epoch}{_TMP_SUFFIX}"
    return final_path, tmp_path


def iter_all_tmp_paths():
    """Yield this subsystem's temp files under every registered model folder.

    Matches only our distinctive ``_TMP_SUFFIX`` (not every ``*.tmp``) so the
    startup orphan-sweep can't delete temp files created by other tools.
    """
    seen_roots: set[str] = set()
    for directory in folder_paths.folder_names_and_paths.keys():
        for root in folder_paths.get_folder_paths(directory):
            if root in seen_roots or not os.path.isdir(root):
                continue
            seen_roots.add(root)
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and entry.name.endswith(_TMP_SUFFIX):
                        yield entry.path
            except OSError:
                # Folder might be unreadable / missing on certain mounts —
                # not fatal, just skip it.
                continue

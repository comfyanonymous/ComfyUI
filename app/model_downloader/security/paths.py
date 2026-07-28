"""Path resolution + traversal safety for downloads.

A ``model_id`` is a *relative destination path* of the form
``<directory>/<filename>`` (e.g. ``loras/my_lora.safetensors``). This module
turns one into an absolute on-disk path under one of ComfyUI's registered
model folders, rejecting unknown folders, path traversal, and symlink escape.
This is the only thing that composes destination paths, so the engine never
touches user-supplied path strings directly.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Optional

import folder_paths

from app.model_downloader.constants import TMP_SUFFIX
from app.model_downloader.security.allowlist import ALLOWED_MODEL_EXTENSIONS

# A model_id component is a single path segment of safe characters — no slashes,
# no "..", no leading dots that could escape the target directory.
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class InvalidModelId(ValueError):
    """Raised when a model_id is malformed or names an unknown model folder."""


def parse_model_id(model_id: str, allow_any_extension: bool = False) -> tuple[str, str]:
    """Split ``<directory>/<filename>`` and validate both components.

    Returns ``(directory, filename)``. Does not touch the filesystem.
    """
    if not isinstance(model_id, str) or "/" not in model_id:
        raise InvalidModelId(
            f"model_id must be '<directory>/<filename>', got {model_id!r}"
        )
    directory, _, filename = model_id.partition("/")
    if "/" in filename or not directory or not filename:
        raise InvalidModelId(
            f"model_id must have exactly one '/' separator, got {model_id!r}"
        )
    if not _SEGMENT_RE.match(directory):
        raise InvalidModelId(f"invalid directory segment {directory!r}")
    if not _SEGMENT_RE.match(filename):
        raise InvalidModelId(f"invalid filename segment {filename!r}")
    if not allow_any_extension and not filename.lower().endswith(
        ALLOWED_MODEL_EXTENSIONS
    ):
        raise InvalidModelId(
            f"filename must end with a known model extension "
            f"{ALLOWED_MODEL_EXTENSIONS}, got {filename!r}"
        )
    if directory not in folder_paths.folder_names_and_paths:
        raise InvalidModelId(f"unknown model folder {directory!r}")
    return directory, filename


def apply_extension(model_id: str, ext: str) -> str:
    """Return ``model_id`` with its filename forced to end in ``ext``.

    ``ext`` includes the leading dot (e.g. ``".safetensors"``). If the filename
    already ends in a *known model extension* it is replaced; otherwise ``ext``
    is appended (so ``loras/mymodel`` -> ``loras/mymodel.safetensors`` and
    ``loras/mymodel.ckpt`` -> ``loras/mymodel.safetensors``). A filename with a
    non-model suffix (``my.model.v2``) is treated as an extensionless stem and
    ``ext`` is appended. The directory part is left untouched; validation is
    still the caller's job via :func:`parse_model_id`.
    """
    directory, sep, filename = model_id.partition("/")
    if not sep:
        return model_id  # malformed; parse_model_id will reject it
    low = filename.lower()
    for known in ALLOWED_MODEL_EXTENSIONS:
        if low.endswith(known):
            filename = filename[: -len(known)]
            break
    return f"{directory}{sep}{filename}{ext}"


def resolve_existing(model_id: str, allow_any_extension: bool = False) -> Optional[str]:
    """Return the absolute path of an installed model, or None if missing.

    Honours ``extra_model_paths.yaml`` transparently via ``get_full_path``.
    """
    directory, filename = parse_model_id(model_id, allow_any_extension)
    return folder_paths.get_full_path(directory, filename)


def resolve_destination(
    model_id: str, allow_any_extension: bool = False
) -> tuple[str, str]:
    """Return ``(final_path, temp_path)`` for a download.

    Downloads land at the first registered path for the model's directory
    (the "primary" location). ``temp_path`` is a sibling ``.part`` file that
    is atomically renamed onto ``final_path`` on success. The result is
    asserted to stay within the registered root (defence in depth on top of
    the segment regex).
    """
    directory, filename = parse_model_id(model_id, allow_any_extension)
    roots = folder_paths.get_folder_paths(directory)
    if not roots:
        raise InvalidModelId(f"no on-disk path registered for folder {directory!r}")
    root = os.path.realpath(roots[0])
    final_path = os.path.realpath(os.path.join(root, filename))
    if final_path != root and not final_path.startswith(root + os.sep):
        raise InvalidModelId(f"resolved path escapes model root: {model_id!r}")
    temp_path = f"{final_path}{TMP_SUFFIX}"
    return final_path, temp_path


def iter_all_tmp_paths() -> Iterator[str]:
    """Yield this subsystem's temp files under every registered model folder.

    Matches only the distinctive ``TMP_SUFFIX`` so the startup orphan sweep
    can never delete temp files created by other tools.
    """
    seen_roots: set[str] = set()
    for directory in list(folder_paths.folder_names_and_paths.keys()):
        for root in folder_paths.get_folder_paths(directory):
            if root in seen_roots or not os.path.isdir(root):
                continue
            seen_roots.add(root)
            try:
                for entry in os.scandir(root):
                    if entry.is_file() and entry.name.endswith(TMP_SUFFIX):
                        yield entry.path
            except OSError:
                continue

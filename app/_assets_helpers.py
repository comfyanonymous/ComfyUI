import os
from pathlib import Path
from typing import Optional, Literal, Sequence

import folder_paths


def get_comfy_models_folders() -> list[tuple[str, list[str]]]:
    """Build a list of (folder_name, base_paths[]) categories that are configured for model locations.

    We trust `folder_paths.folder_names_and_paths` and include a category if
    *any* of its base paths lies under the Comfy `models_dir`.
    """
    targets: list[tuple[str, list[str]]] = []
    models_root = os.path.abspath(folder_paths.models_dir)
    for name, (paths, _exts) in folder_paths.folder_names_and_paths.items():
        if any(os.path.abspath(p).startswith(models_root + os.sep) for p in paths):
            targets.append((name, paths))
    return targets


def get_relative_to_root_category_path_of_asset(file_path: str) -> tuple[Literal["input", "output", "models"], str]:
    """Given an absolute or relative file path, determine which root category the path belongs to:
      - 'input' if the file resides under `folder_paths.get_input_directory()`
      - 'output' if the file resides under `folder_paths.get_output_directory()`
      - 'models' if the file resides under any base path of categories returned by `get_comfy_models_folders()`

    Returns:
        (root_category, relative_path_inside_that_root)
        For 'models', the relative path is prefixed with the category name:
            e.g. ('models', 'vae/test/sub/ae.safetensors')

    Raises:
        ValueError: if the path does not belong to input, output, or configured model bases.
    """
    fp_abs = os.path.abspath(file_path)

    def _is_within(child: str, parent: str) -> bool:
        try:
            return os.path.commonpath([child, parent]) == parent
        except Exception:
            return False

    def _rel(child: str, parent: str) -> str:
        return os.path.relpath(os.path.join(os.sep, os.path.relpath(child, parent)), os.sep)

    # 1) input
    input_base = os.path.abspath(folder_paths.get_input_directory())
    if _is_within(fp_abs, input_base):
        return "input", _rel(fp_abs, input_base)

    # 2) output
    output_base = os.path.abspath(folder_paths.get_output_directory())
    if _is_within(fp_abs, output_base):
        return "output", _rel(fp_abs, output_base)

    # 3) models (check deepest matching base to avoid ambiguity)
    best: Optional[tuple[int, str, str]] = None  # (base_len, bucket, rel_inside_bucket)
    for bucket, bases in get_comfy_models_folders():
        for b in bases:
            base_abs = os.path.abspath(b)
            if not _is_within(fp_abs, base_abs):
                continue
            cand = (len(base_abs), bucket, _rel(fp_abs, base_abs))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is not None:
        _, bucket, rel_inside = best
        combined = os.path.join(bucket, rel_inside)
        return "models", os.path.relpath(os.path.join(os.sep, combined), os.sep)

    raise ValueError(f"Path is not within input, output, or configured model bases: {file_path}")


def get_name_and_tags_from_asset_path(file_path: str) -> tuple[str, list[str]]:
    """Return a tuple (name, tags) derived from a filesystem path.

    Semantics:
      - Root category is determined by `get_relative_to_root_category_path_of_asset`.
      - The returned `name` is the base filename with extension from the relative path.
      - The returned `tags` are:
            [root_category] + parent folders of the relative path (in order)
        For 'models', this means:
            file '/.../ModelsDir/vae/test_tag/ae.safetensors'
            -> root_category='models', some_path='vae/test_tag/ae.safetensors'
            -> name='ae.safetensors', tags=['models', 'vae', 'test_tag']

    Raises:
        ValueError: if the path does not belong to input, output, or configured model bases.
    """
    root_category, some_path = get_relative_to_root_category_path_of_asset(file_path)
    p = Path(some_path)
    parent_parts = [part for part in p.parent.parts if part not in (".", "..", p.anchor)]
    return p.name, normalize_tags([root_category, *parent_parts])


def normalize_tags(tags: Optional[Sequence[str]]) -> list[str]:
    return [t.strip().lower() for t in (tags or []) if (t or "").strip()]

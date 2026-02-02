import contextlib
import os
from decimal import Decimal
from aiohttp import web
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Any

import folder_paths


RootType = Literal["models", "input", "output"]
ALLOWED_ROOTS: tuple[RootType, ...] = ("models", "input", "output")

def get_query_dict(request: web.Request) -> dict[str, Any]:
    """
    Gets a dictionary of query parameters from the request.

    'request.query' is a MultiMapping[str], needs to be converted to a dictionary to be validated by Pydantic.
    """
    query_dict = {
        key: request.query.getall(key) if len(request.query.getall(key)) > 1 else request.query.get(key)
        for key in request.query.keys()
    }
    return query_dict

def list_tree(base_dir: str) -> list[str]:
    out: list[str] = []
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        return out
    for dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
        for name in filenames:
            out.append(os.path.abspath(os.path.join(dirpath, name)))
    return out

def prefixes_for_root(root: RootType) -> list[str]:
    if root == "models":
        bases: list[str] = []
        for _bucket, paths in get_comfy_models_folders():
            bases.extend(paths)
        return [os.path.abspath(p) for p in bases]
    if root == "input":
        return [os.path.abspath(folder_paths.get_input_directory())]
    if root == "output":
        return [os.path.abspath(folder_paths.get_output_directory())]
    return []

def escape_like_prefix(s: str, escape: str = "!") -> tuple[str, str]:
    """Escapes %, _ and the escape char itself in a LIKE prefix.
    Returns (escaped_prefix, escape_char). Caller should append '%' and pass escape=escape_char to .like().
    """
    s = s.replace(escape, escape + escape)  # escape the escape char first
    s = s.replace("%", escape + "%").replace("_", escape + "_")  # escape LIKE wildcards
    return s, escape

def fast_asset_file_check(
    *,
    mtime_db: int | None,
    size_db: int | None,
    stat_result: os.stat_result,
) -> bool:
    if mtime_db is None:
        return False
    actual_mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
    if int(mtime_db) != int(actual_mtime_ns):
        return False
    sz = int(size_db or 0)
    if sz > 0:
        return int(stat_result.st_size) == sz
    return True

def utcnow() -> datetime:
    """Naive UTC timestamp (no tzinfo). We always treat DB datetimes as UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

def get_comfy_models_folders() -> list[tuple[str, list[str]]]:
    """Build a list of (folder_name, base_paths[]) categories that are configured for model locations.

    We trust `folder_paths.folder_names_and_paths` and include a category if
    *any* of its base paths lies under the Comfy `models_dir`.
    """
    targets: list[tuple[str, list[str]]] = []
    models_root = os.path.abspath(folder_paths.models_dir)
    for name, values in folder_paths.folder_names_and_paths.items():
        paths, _exts = values[0], values[1]  # NOTE: this prevents nodepacks that hackily edit folder_... from breaking ComfyUI
        if any(os.path.abspath(p).startswith(models_root + os.sep) for p in paths):
            targets.append((name, paths))
    return targets

def resolve_destination_from_tags(tags: list[str]) -> tuple[str, list[str]]:
    """Validates and maps tags -> (base_dir, subdirs_for_fs)"""
    root = tags[0]
    if root == "models":
        if len(tags) < 2:
            raise ValueError("at least two tags required for model asset")
        try:
            bases = folder_paths.folder_names_and_paths[tags[1]][0]
        except KeyError:
            raise ValueError(f"unknown model category '{tags[1]}'")
        if not bases:
            raise ValueError(f"no base path configured for category '{tags[1]}'")
        base_dir = os.path.abspath(bases[0])
        raw_subdirs = tags[2:]
    else:
        base_dir = os.path.abspath(
            folder_paths.get_input_directory() if root == "input" else folder_paths.get_output_directory()
        )
        raw_subdirs = tags[1:]
    for i in raw_subdirs:
        if i in (".", ".."):
            raise ValueError("invalid path component in tags")

    return base_dir, raw_subdirs if raw_subdirs else []

def ensure_within_base(candidate: str, base: str) -> None:
    cand_abs = os.path.abspath(candidate)
    base_abs = os.path.abspath(base)
    try:
        if os.path.commonpath([cand_abs, base_abs]) != base_abs:
            raise ValueError("destination escapes base directory")
    except Exception:
        raise ValueError("invalid destination path")

def compute_relative_filename(file_path: str) -> str | None:
    """
    Return the model's path relative to the last well-known folder (the model category),
    using forward slashes, eg:
      /.../models/checkpoints/flux/123/flux.safetensors -> "flux/123/flux.safetensors"
      /.../models/text_encoders/clip_g.safetensors -> "clip_g.safetensors"

    For non-model paths, returns None.
    NOTE: this is a temporary helper, used only for initializing metadata["filename"] field.
    """
    try:
        root_category, rel_path = get_relative_to_root_category_path_of_asset(file_path)
    except ValueError:
        return None

    p = Path(rel_path)
    parts = [seg for seg in p.parts if seg not in (".", "..", p.anchor)]
    if not parts:
        return None

    if root_category == "models":
        # parts[0] is the category ("checkpoints", "vae", etc) – drop it
        inside = parts[1:] if len(parts) > 1 else [parts[0]]
        return "/".join(inside)
    return "/".join(parts)  # input/output: keep all parts

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
    best: tuple[int, str, str] | None = None  # (base_len, bucket, rel_inside_bucket)
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
    return p.name, list(dict.fromkeys(normalize_tags([root_category, *parent_parts])))

def normalize_tags(tags: list[str] | None) -> list[str]:
    """
    Normalize a list of tags by:
      - Stripping whitespace and converting to lowercase.
      - Removing duplicates.
    """
    return [t.strip().lower() for t in (tags or []) if (t or "").strip()]

def collect_models_files() -> list[str]:
    out: list[str] = []
    for folder_name, bases in get_comfy_models_folders():
        rel_files = folder_paths.get_filename_list(folder_name) or []
        for rel_path in rel_files:
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)
            allowed = False
            for b in bases:
                base_abs = os.path.abspath(b)
                with contextlib.suppress(Exception):
                    if os.path.commonpath([abs_path, base_abs]) == base_abs:
                        allowed = True
                        break
            if allowed:
                out.append(abs_path)
    return out

def is_scalar(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False

def project_kv(key: str, value):
    """
    Turn a metadata key/value into typed projection rows.
    Returns list[dict] with keys:
      key, ordinal, and one of val_str / val_num / val_bool / val_json (others None)
    """
    rows: list[dict] = []

    def _null_row(ordinal: int) -> dict:
        return {
            "key": key, "ordinal": ordinal,
            "val_str": None, "val_num": None, "val_bool": None, "val_json": None
        }

    if value is None:
        rows.append(_null_row(0))
        return rows

    if is_scalar(value):
        if isinstance(value, bool):
            rows.append({"key": key, "ordinal": 0, "val_bool": bool(value)})
        elif isinstance(value, (int, float, Decimal)):
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            rows.append({"key": key, "ordinal": 0, "val_num": num})
        elif isinstance(value, str):
            rows.append({"key": key, "ordinal": 0, "val_str": value})
        else:
            rows.append({"key": key, "ordinal": 0, "val_json": value})
        return rows

    if isinstance(value, list):
        if all(is_scalar(x) for x in value):
            for i, x in enumerate(value):
                if x is None:
                    rows.append(_null_row(i))
                elif isinstance(x, bool):
                    rows.append({"key": key, "ordinal": i, "val_bool": bool(x)})
                elif isinstance(x, (int, float, Decimal)):
                    num = x if isinstance(x, Decimal) else Decimal(str(x))
                    rows.append({"key": key, "ordinal": i, "val_num": num})
                elif isinstance(x, str):
                    rows.append({"key": key, "ordinal": i, "val_str": x})
                else:
                    rows.append({"key": key, "ordinal": i, "val_json": x})
            return rows
        for i, x in enumerate(value):
            rows.append({"key": key, "ordinal": i, "val_json": x})
        return rows

    rows.append({"key": key, "ordinal": 0, "val_json": value})
    return rows

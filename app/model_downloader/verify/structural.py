"""Cheap structural validation, no full read.

For ``.safetensors``/``.sft`` we parse the header (first few KB): it carries
the tensor table and the byte length of the data region. We assert
``file_size == 8 + header_len + data_region_len``. This detects truncation
and most corruption for free, before any crypto hashing. Other extensions
have no cheap structural check and pass through.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Optional

_SAFETENSORS_EXTS = (".safetensors", ".sft")
# A sane upper bound so a corrupt header length can't make us read gigabytes.
_MAX_HEADER_BYTES = 100 * 1024 * 1024


class StructuralError(Exception):
    """The file failed its structural integrity check."""


def validate(path: str, name_hint: Optional[str] = None) -> None:
    """Validate the file at ``path``. Raises :class:`StructuralError` on failure.

    The file format is detected from ``name_hint`` when provided, otherwise from
    ``path``. Callers that download into a temp file with an opaque suffix (e.g.
    ``*.comfy-download.part``) must pass the final destination name as
    ``name_hint`` so the format check is not silently skipped.
    """
    lower = (name_hint or path).lower()
    if lower.endswith(_SAFETENSORS_EXTS):
        _validate_safetensors(path)
    # No structural check for other formats; the size + (optional) checksum
    # gates in the engine cover those.


def _validate_safetensors(path: str) -> None:
    file_size = os.path.getsize(path)
    if file_size < 8:
        raise StructuralError(f"file too small to be safetensors ({file_size} bytes)")
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            raise StructuralError(f"implausible safetensors header length {header_len}")
        if 8 + header_len > file_size:
            raise StructuralError("safetensors header extends past end of file")
        try:
            header = json.loads(f.read(header_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise StructuralError(f"safetensors header is not valid JSON: {e}") from e

    if not isinstance(header, dict):
        raise StructuralError("safetensors header is not a JSON object")

    data_len = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict) or "data_offsets" not in entry:
            raise StructuralError(f"tensor {name!r} missing data_offsets")
        offsets = entry["data_offsets"]
        if not (isinstance(offsets, list) and len(offsets) == 2):
            raise StructuralError(f"tensor {name!r} has malformed data_offsets")
        begin, end = offsets
        # bool is an int subclass; reject it explicitly to avoid True/False offsets.
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or isinstance(begin, bool)
            or isinstance(end, bool)
            or begin < 0
            or end < begin
        ):
            raise StructuralError(f"tensor {name!r} has malformed data_offsets")
        data_len = max(data_len, end)

    expected = 8 + header_len + data_len
    if file_size != expected:
        raise StructuralError(
            f"size mismatch: file is {file_size} bytes, header implies {expected} "
            f"(8 + {header_len} header + {data_len} data)"
        )

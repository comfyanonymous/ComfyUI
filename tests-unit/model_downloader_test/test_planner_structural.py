"""Unit tests for the segment planner and structural safetensors validation."""

from __future__ import annotations

import json
import struct

import pytest

from app.model_downloader.engine.planner import (
    effective_segment_count,
    plan_segments,
)
from app.model_downloader.verify import structural


# ----- planner -----


def test_plan_segments_covers_full_range_contiguously():
    total = 1000
    plans = plan_segments(total, 4)
    assert len(plans) == 4
    assert plans[0].start == 0
    assert plans[-1].end == total - 1
    # contiguous, no gaps/overlaps
    for a, b in zip(plans, plans[1:]):
        assert b.start == a.end + 1
    assert sum(p.length for p in plans) == total


def test_effective_segment_count_falls_back_to_single():
    # No range support -> single
    assert effective_segment_count(10_000_000, False, 8) == 1
    # Unknown size -> single
    assert effective_segment_count(None, True, 8) == 1
    # Tiny file -> fewer segments than configured
    assert effective_segment_count(1024, True, 8) == 1
    # Large file with range support -> configured count
    assert effective_segment_count(1_000_000_000, True, 8) == 8


# ----- structural -----


def _make_safetensors(tensor_data_len: int, *, corrupt_size: bool = False) -> bytes:
    header = {"t": {"dtype": "F32", "shape": [tensor_data_len], "data_offsets": [0, tensor_data_len]}}
    header_bytes = json.dumps(header).encode("utf-8")
    body = b"\x00" * tensor_data_len
    if corrupt_size:
        body = body[:-1]  # truncate one byte
    return struct.pack("<Q", len(header_bytes)) + header_bytes + body


def test_structural_valid_safetensors(tmp_path):
    p = tmp_path / "ok.safetensors"
    p.write_bytes(_make_safetensors(256))
    structural.validate(str(p))  # no raise


def test_structural_detects_truncation(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(_make_safetensors(256, corrupt_size=True))
    with pytest.raises(structural.StructuralError):
        structural.validate(str(p))


def test_structural_skips_unknown_extension(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"anything")
    structural.validate(str(p))  # no structural check, no raise


def test_structural_detects_truncation_via_name_hint(tmp_path):
    # The downloader validates the opaque temp file (a ``.part`` path) but keys
    # the format check off the final destination name via ``name_hint``, so
    # truncation must still be detected instead of silently skipped.
    p = tmp_path / "bad.comfy-download.part"
    p.write_bytes(_make_safetensors(256, corrupt_size=True))
    with pytest.raises(structural.StructuralError):
        structural.validate(str(p), name_hint="model.safetensors")

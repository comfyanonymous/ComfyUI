import struct
import json

import pytest

from comfy.utils import load_torch_file


def _write_safetensors(path, payload: bytes) -> str:
    path.write_bytes(payload)
    return str(path)


def test_load_torch_file_header_too_large(tmp_path):
    """A header length prefix too large to be a real safetensors header
    (e.g. a .ckpt or .pt renamed to .safetensors) raises the friendly
    'corrupt or invalid' message instead of the raw SafetensorError."""
    payload = struct.pack("<Q", 2**40) + b"x" * 64
    path = _write_safetensors(tmp_path / "too_large.safetensors", payload)
    with pytest.raises(ValueError, match="corrupt or invalid"):
        load_torch_file(path)


def test_load_torch_file_incomplete_download(tmp_path):
    """A valid header whose tensor data is cut short (truncated download)
    raises the friendly 'corrupt/incomplete' message."""
    header = json.dumps({"a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    payload = struct.pack("<Q", len(header)) + header + b"\x00\x00"
    path = _write_safetensors(tmp_path / "truncated.safetensors", payload)
    with pytest.raises(ValueError, match="corrupt/incomplete"):
        load_torch_file(path)


def test_load_torch_file_valid_empty_file(tmp_path):
    """A valid (empty) safetensors file still loads without error."""
    payload = struct.pack("<Q", 2) + b"{}"
    path = _write_safetensors(tmp_path / "valid.safetensors", payload)
    assert load_torch_file(path) == {}

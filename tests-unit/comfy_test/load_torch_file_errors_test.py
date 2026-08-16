import struct

import pytest
import torch
from safetensors.torch import save_file

import comfy.memory_management
import comfy.utils


@pytest.fixture(autouse=True)
def use_safetensors_reader(monkeypatch):
    # Pin the plain safetensors reader so these tests exercise the safetensors
    # error messages rather than the aimdo mmap parser.
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", False)


def write_valid_file(path):
    save_file({"weight": torch.zeros(4)}, str(path))
    return path


def test_bogus_header_reports_wrong_filetype(tmp_path):
    # Declares a header far larger than the file: what you get when a ckpt/pt
    # file (or an HTML error page) is renamed to .safetensors.
    path = tmp_path / "bogus_header.safetensors"
    path.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")

    with pytest.raises(ValueError, match="not a ckpt or pt or other filetype"):
        comfy.utils.load_torch_file(str(path))


def test_truncated_file_reports_incomplete_download(tmp_path):
    good = write_valid_file(tmp_path / "good.safetensors")
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(good.read_bytes()[:-4])

    with pytest.raises(ValueError, match="corrupt/incomplete"):
        comfy.utils.load_torch_file(str(truncated))


def test_valid_file_still_loads(tmp_path):
    path = write_valid_file(tmp_path / "good.safetensors")

    sd = comfy.utils.load_torch_file(str(path))

    assert list(sd.keys()) == ["weight"]

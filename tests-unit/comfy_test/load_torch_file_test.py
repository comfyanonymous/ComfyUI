import pytest

import comfy.memory_management
import comfy.utils


def test_shape_mismatch_reports_corrupt_file(monkeypatch):
    monkeypatch.setattr(comfy.memory_management, "aimdo_enabled", True)

    def fake_load_safetensors(ckpt):
        raise RuntimeError("shape '[25600, 2560]' is invalid for input of size 3930620")

    monkeypatch.setattr(comfy.utils, "load_safetensors", fake_load_safetensors)

    with pytest.raises(ValueError, match="corrupt/incomplete"):
        comfy.utils.load_torch_file("fake_path.safetensors", safe_load=True)

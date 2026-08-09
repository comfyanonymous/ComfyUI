import os
import tempfile

import pytest
import safetensors.torch
import torch

import comfy.utils


@pytest.fixture
def safetensors_file():
    tensors = {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = os.path.join(tmpdirname, "model.safetensors")
        safetensors.torch.save_file(tensors, path, metadata={"format": "pt"})
        yield path, tensors


def test_disable_mmap_does_not_use_safe_open(safetensors_file, monkeypatch):
    path, tensors = safetensors_file
    monkeypatch.setattr(comfy.utils, "DISABLE_MMAP", True)

    def boom(*args, **kwargs):
        raise AssertionError("safetensors.safe_open should not be used when DISABLE_MMAP is set")

    monkeypatch.setattr(comfy.utils.safetensors, "safe_open", boom)

    sd, metadata = comfy.utils.load_torch_file(path, device=torch.device("cpu"), return_metadata=True)

    assert torch.equal(sd["weight"], tensors["weight"])
    assert metadata == {"format": "pt"}

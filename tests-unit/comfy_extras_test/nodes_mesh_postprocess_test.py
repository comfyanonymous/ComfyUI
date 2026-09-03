"""_mesh_postprocess_compute_device must route MPS to CPU (issue #16017).

PyTorch's MPS backend produces out-of-range scatter/index_put_ indices on
the large index tensors the QEM/dual-contouring mesh kernels build, so the
mesh postprocess nodes must not compute on MPS even when it is the active
device.
"""

import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
from comfy_extras.nodes_mesh_postprocess import _mesh_postprocess_compute_device


def test_mps_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    assert _mesh_postprocess_compute_device() == torch.device("cpu")


def test_non_mps_device_is_unchanged(monkeypatch):
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )

    assert _mesh_postprocess_compute_device() == torch.device("cpu")

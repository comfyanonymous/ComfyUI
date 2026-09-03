"""_mesh_postprocess_compute_device must route MPS to CPU (issue #16017).

PyTorch's MPS backend produces out-of-range scatter/index_put_ indices on
the large index tensors the mesh postprocess kernels build (QEM decimate,
dual-contouring remesh, hole-filling, pec UV chart segmentation), so these
nodes must not compute on MPS even when it is the active device.
"""

import types

import numpy as np
import torch

from comfy.cli_args import args as cli_args

if not torch.cuda.is_available():
    cli_args.cpu = True

import comfy.model_management
import comfy_extras.nodes_mesh_postprocess as nodes_mesh_postprocess
from comfy_api.latest._util.geometry_types import MESH
from comfy_extras.nodes_mesh_postprocess import (
    UnwrapMesh,
    _mesh_postprocess_compute_device,
    fill_holes_v2_fn,
)


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


def test_fill_holes_uses_guarded_device_on_mps(monkeypatch):
    """FillHoles' fill_holes_v2_fn also builds scatter_reduce_/scatter_add_ index
    tensors (component labeling, perimeter/centroid reduction); it must route
    through the same MPS->CPU guard as DecimateMesh/RemeshMesh."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    out_v, out_f, _out_c = fill_holes_v2_fn(verts, faces, max_perimeter=10.0, weld_epsilon_rel=0.0)

    assert out_v.device == torch.device("cpu")
    assert out_f.device == torch.device("cpu")


def test_unwrap_mesh_pec_uses_guarded_device_on_mps(monkeypatch):
    """UnwrapMesh's "pec" segmenter runs parallel-edge-collapse chart clustering
    (scatter_reduce_-heavy) on compute_device; it must also route through the
    MPS->CPU guard. "adaptive" is unaffected (already forced to CPU)."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    seen = {}

    def fake_uv_unwrap(positions, indices, segmenter, resolution, padding, weld_distance):
        seen["device"] = positions.device
        n = positions.shape[0]
        return np.arange(n), indices.cpu().numpy(), np.zeros((n, 2), dtype=np.float32)

    monkeypatch.setattr(nodes_mesh_postprocess, "_uv_unwrap", fake_uv_unwrap)
    monkeypatch.setattr(UnwrapMesh, "hidden", types.SimpleNamespace(unique_id=None), raising=False)

    verts = torch.zeros((4, 3))
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    mesh = MESH(vertices=verts, faces=faces)

    UnwrapMesh.execute(mesh, "pec", 1024, 1, 0.0)

    assert seen["device"] == torch.device("cpu")

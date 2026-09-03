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
    _uv_unwrap,
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


def test_fill_holes_selects_device_before_weld(monkeypatch):
    """The compute-device fallback must be selected (and inputs moved) before
    the adaptive weld runs; otherwise weld_vertices_fn's scatter_add_ calls
    still execute on MPS even though _fill_holes_v2_gpu itself is guarded."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    call_order = []
    real_weld = nodes_mesh_postprocess.weld_vertices_fn
    real_compute_device = nodes_mesh_postprocess._mesh_postprocess_compute_device

    def tracking_weld(*args, **kwargs):
        call_order.append("weld")
        return real_weld(*args, **kwargs)

    def tracking_compute_device():
        call_order.append("compute_device")
        return real_compute_device()

    monkeypatch.setattr(nodes_mesh_postprocess, "weld_vertices_fn", tracking_weld)
    monkeypatch.setattr(
        nodes_mesh_postprocess, "_mesh_postprocess_compute_device", tracking_compute_device
    )

    verts = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 2, 3], [1, 2, 3]], dtype=torch.long)

    fill_holes_v2_fn(verts, faces, max_perimeter=10.0, weld_epsilon_rel=1e-5)

    assert "compute_device" in call_order
    assert "weld" in call_order
    assert call_order.index("compute_device") < call_order.index("weld")


def test_fill_holes_empty_faces_routed_to_selected_device(monkeypatch):
    """The empty-face early return must also land on the selected device:
    a batch mixing an empty item with a non-empty item would otherwise hand
    torch.stack tensors on different devices (the non-empty item is moved by
    the guard below; the empty item wasn't)."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("cpu")
    )

    moved = []
    real_to = torch.Tensor.to

    def tracking_to(self, *args, **kwargs):
        moved.append(self)
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracking_to)

    verts = torch.zeros((0, 3))
    faces = torch.zeros((0, 3), dtype=torch.long)
    colors = torch.zeros((0, 4))

    fill_holes_v2_fn(verts, faces, max_perimeter=10.0, colors=colors)

    assert any(t is verts for t in moved)
    assert any(t is faces for t in moved)
    assert any(t is colors for t in moved)


def test_unwrap_mesh_pec_uses_guarded_device_on_mps(monkeypatch):
    """UnwrapMesh's "pec" segmenter runs parallel-edge-collapse chart clustering
    (scatter_reduce_-heavy) on compute_device; it must also route through the
    MPS->CPU guard. "adaptive" is unaffected (already forced to CPU)."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    seen = {}

    def fake_uv_unwrap(positions, indices, segmenter, resolution, padding, weld_distance, device=None):
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


def test_uv_unwrap_lscm_uses_passed_device_not_raw_get_torch_device(monkeypatch):
    """_uv_unwrap's LSCM batch solve must use the device passed in by the caller
    (UnwrapMesh.execute's guarded seg_device), not comfy.model_management.get_torch_device()
    directly -- otherwise the "pec" path can still build scatter tensors on MPS even
    though its inputs were already moved to CPU by the guard."""
    monkeypatch.setattr(
        comfy.model_management, "get_torch_device", lambda: torch.device("mps")
    )

    seen = {}
    real_lscm = nodes_mesh_postprocess._uv_param.lscm_charts_batch

    def tracking_lscm(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        return real_lscm(*args, **kwargs)

    monkeypatch.setattr(nodes_mesh_postprocess._uv_param, "lscm_charts_batch", tracking_lscm)

    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)

    _uv_unwrap(verts, faces, "pec", 1024, 1, 0.0, device=torch.device("cpu"))

    assert seen["device"] == torch.device("cpu")

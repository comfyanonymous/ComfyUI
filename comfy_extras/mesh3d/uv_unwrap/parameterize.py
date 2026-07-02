"""Chart parameterization: ortho PCA projection, falling back to ABF/LSCM."""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from . import mesh as _mesh


def solve_least_squares(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Solve ||Ax - b||^2 by factorizing AtA."""
    At = A.T.tocsr()
    AtA = (At @ A).tocsc()
    Atb = At @ b
    return spla.spsolve(AtA, Atb)


def _triangle_local_2d(verts_3d: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-triangle 2D coords [F, 3, 2] with v0 at origin, v1 along +x."""
    v0 = verts_3d[faces[:, 0]]
    v1 = verts_3d[faces[:, 1]]
    v2 = verts_3d[faces[:, 2]]
    e01 = v1 - v0
    e02 = v2 - v0
    L01 = np.linalg.norm(e01, axis=1).clip(min=1e-20)
    x_axis = e01 / L01[:, None]
    n = np.cross(e01, e02)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(min=1e-20)
    y_axis = np.cross(n, x_axis)

    out = np.zeros((faces.shape[0], 3, 2), dtype=np.float64)
    out[:, 1, 0] = L01
    out[:, 2, 0] = (e02 * x_axis).sum(axis=1)
    out[:, 2, 1] = (e02 * y_axis).sum(axis=1)
    return out


def _pick_pins(loops: List[List[int]], verts_3d: np.ndarray) -> Tuple[int, int]:
    """Pick the longest-diameter axis-extremal boundary vertex pair across all boundary verts."""
    if not loops:
        # Closed surface: two far verts via two-pass farthest.
        d2 = np.sum((verts_3d - verts_3d[0]) ** 2, axis=1)
        a = int(np.argmax(d2))
        d2 = np.sum((verts_3d - verts_3d[a]) ** 2, axis=1)
        b = int(np.argmax(d2))
        return a, b
    boundary_verts: List[int] = []
    for loop in loops:
        boundary_verts.extend(loop)
    seen = set()
    uniq = []
    for v in boundary_verts:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    bv = np.asarray(uniq, dtype=np.int64)
    pts = verts_3d[bv]
    pin_pairs = []
    for axis in range(3):
        i_min = int(bv[int(np.argmin(pts[:, axis]))])
        i_max = int(bv[int(np.argmax(pts[:, axis]))])
        d = float(np.linalg.norm(verts_3d[i_min] - verts_3d[i_max]))
        pin_pairs.append((d, i_min, i_max))
    d0, _, _ = pin_pairs[0]
    d1, _, _ = pin_pairs[1]
    d2, _, _ = pin_pairs[2]
    if d0 > d1 and d0 > d2:
        _, a, b = pin_pairs[0]
    elif d1 > d2:
        _, a, b = pin_pairs[1]
    else:
        _, a, b = pin_pairs[2]
    return a, b


def _ortho_project(verts_3d: np.ndarray) -> np.ndarray:
    """PCA-fit plane normal, axis-aligned tangent, project verts to 2D."""
    centroid = verts_3d.mean(axis=0)
    pts = verts_3d - centroid
    cov = pts.T @ pts
    _w, ev = np.linalg.eigh(cov)
    normal = ev[:, 0]
    a = np.abs(normal)
    if a[0] < a[1] and a[0] < a[2]:
        t = np.array([1.0, 0.0, 0.0])
    elif a[1] < a[2]:
        t = np.array([0.0, 1.0, 0.0])
    else:
        t = np.array([0.0, 0.0, 1.0])
    t = t - normal * float(np.dot(normal, t))
    t /= max(float(np.linalg.norm(t)), 1e-20)
    b = np.cross(normal, t)
    return np.stack([verts_3d @ t, verts_3d @ b], axis=1)


def _stretch_metrics(verts_3d: np.ndarray, uvs: np.ndarray, faces: np.ndarray) -> Tuple[float, float, int, int]:
    """Sander's stretch metric. Returns (rms, max, n_flipped, n_zero_area)."""
    p = verts_3d[faces]
    t = uvs[faces]
    parametric_area = 0.5 * (
        (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
        - (t[:, 2, 1] - t[:, 0, 1]) * (t[:, 1, 0] - t[:, 0, 0])
    )
    n_flipped = int((parametric_area < -1e-12).sum())
    n_zero = int((np.abs(parametric_area) < 1e-12).sum())
    pa = np.abs(parametric_area).clip(min=1e-20)
    geom_area = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )
    keep = (geom_area > 1e-12) & (np.abs(parametric_area) > 1e-12)
    if not keep.any():
        return float("inf"), float("inf"), n_flipped, n_zero
    t1 = t[:, 0, 0]
    s1 = t[:, 0, 1]
    t2 = t[:, 1, 0]
    s2 = t[:, 1, 1]
    t3 = t[:, 2, 0]
    s3 = t[:, 2, 1]
    inv_2pa = 1.0 / (2.0 * pa)
    Ss = (
        p[:, 0] * (t2 - t3)[:, None]
        + p[:, 1] * (t3 - t1)[:, None]
        + p[:, 2] * (t1 - t2)[:, None]
    ) * inv_2pa[:, None]
    St = (
        p[:, 0] * (s3 - s2)[:, None]
        + p[:, 1] * (s1 - s3)[:, None]
        + p[:, 2] * (s2 - s1)[:, None]
    ) * inv_2pa[:, None]
    a = (Ss * Ss).sum(axis=1)
    bb = (Ss * St).sum(axis=1)
    c = (St * St).sum(axis=1)
    sigma2_sq = 0.5 * (a + c + np.sqrt(np.maximum(0.0, (a - c) ** 2 + 4 * bb ** 2)))
    rms_sq = (a + c) * 0.5
    rms_stretch_sq_sum = float((rms_sq[keep] * geom_area[keep]).sum())
    total_geom = float(geom_area[keep].sum())
    total_param = float(pa[keep].sum())
    if total_geom <= 0.0:
        return float("inf"), float("inf"), n_flipped, n_zero
    norm_factor = np.sqrt(total_param / total_geom)
    rms_stretch = float(np.sqrt(rms_stretch_sq_sum / total_geom)) * norm_factor
    max_stretch = float(np.sqrt(sigma2_sq[keep].max())) * norm_factor
    return rms_stretch, max_stretch, n_flipped, n_zero


def _uv_boundary_self_intersects(
    uvs: np.ndarray, faces: np.ndarray, face_face: np.ndarray, eps: float = 1e-9
) -> bool:
    """True if any chart-boundary edge pair crosses in 2D (ortho folded the chart)."""
    fi, ei = np.nonzero(face_face < 0)
    n = fi.size
    if n < 2:
        return False
    a = uvs[faces[fi, ei]].astype(np.float64)
    b = uvs[faces[fi, (ei + 1) % 3]].astype(np.float64)
    d = b - a
    # Pairwise segment crossings, row-chunked to bound memory at chunk*n.
    chunk = max(1, min(n, 1_000_000 // max(n, 1)))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d1 = d[s:e, None, :]
        denom = d1[:, :, 0] * d[None, :, 1] - d1[:, :, 1] * d[None, :, 0]
        rx = a[None, :, 0] - a[s:e, None, 0]
        ry = a[None, :, 1] - a[s:e, None, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rx * d[None, :, 1] - ry * d[None, :, 0]) / denom
            u = (rx * d1[:, :, 1] - ry * d1[:, :, 0]) / denom
        cross = (
            (np.abs(denom) >= eps)
            & (t > eps) & (t < 1.0 - eps)
            & (u > eps) & (u < 1.0 - eps)
        )
        if bool(cross.any()):
            return True
    return False


def parametrize_chart(
    local_verts: Tensor, local_faces: Tensor, local_face_face: Tensor
) -> Tensor:
    """Parameterize one chart: ortho first, ABF/LSCM fallback; charts <=5 faces stay ortho."""
    verts_np = local_verts.detach().cpu().numpy().astype(np.float64)
    faces_np = local_faces.detach().cpu().numpy().astype(np.int64)
    if verts_np.shape[0] < 3 or faces_np.shape[0] == 0:
        return torch.zeros((verts_np.shape[0], 2), dtype=torch.float32, device=local_verts.device)

    ortho = _ortho_project(verts_np)
    n_faces = faces_np.shape[0]
    if n_faces <= 5:
        return torch.from_numpy(ortho.astype(np.float32)).to(local_verts.device)
    rms, mx, n_flip, n_zero = _stretch_metrics(verts_np, ortho, faces_np)
    flip_ok = n_flip == 0 or n_flip == n_faces
    if flip_ok and n_zero == 0 and rms <= 1.5 and mx <= 2.0:
        ff_np = local_face_face.detach().cpu().numpy().astype(np.int64)
        if not _uv_boundary_self_intersects(ortho, faces_np, ff_np):
            return torch.from_numpy(ortho.astype(np.float32)).to(local_verts.device)
    uvs_t = lscm_chart(local_verts, local_faces, local_face_face, pin_positions=ortho)
    # Collapsed UV island (aspect > 100:1) blows up packing scale; fall back to ortho.
    uvs_np = uvs_t.detach().cpu().numpy()
    bbox = uvs_np.max(axis=0) - uvs_np.min(axis=0)
    bbox_max = float(max(bbox[0], bbox[1], 1e-12))
    bbox_min = float(max(min(bbox[0], bbox[1]), 1e-12))
    if bbox_max / bbox_min > 100.0:
        return torch.from_numpy(ortho.astype(np.float32)).to(local_verts.device)
    return uvs_t


def _abf_face_coefficients(
    verts_3d: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-face ABF constraint (largest-sine vertex at local index 2); returns (faces_reordered, cosine, sine, valid_mask) with valid_mask False for degenerate tris."""
    p0 = verts_3d[faces[:, 0]]
    p1 = verts_3d[faces[:, 1]]
    p2 = verts_3d[faces[:, 2]]
    e01 = p1 - p0
    e12 = p2 - p1
    e20 = p0 - p2
    L01 = np.linalg.norm(e01, axis=1).clip(min=1e-20)
    L12 = np.linalg.norm(e12, axis=1).clip(min=1e-20)
    L20 = np.linalg.norm(e20, axis=1).clip(min=1e-20)
    cos_a0 = ((-e20) * e01).sum(axis=1) / (L20 * L01)
    cos_a1 = ((-e01) * e12).sum(axis=1) / (L01 * L12)
    cos_a2 = ((-e12) * e20).sum(axis=1) / (L12 * L20)
    cos_a0 = cos_a0.clip(-1.0, 1.0)
    cos_a1 = cos_a1.clip(-1.0, 1.0)
    cos_a2 = cos_a2.clip(-1.0, 1.0)
    a = np.arccos(cos_a0)
    b_ang = np.arccos(cos_a1)
    c_ang = np.arccos(cos_a2)
    angles = np.stack([a, b_ang, c_ang], axis=1)
    sines = np.stack([np.sin(a), np.sin(b_ang), np.sin(c_ang)], axis=1)
    valid = (angles > 1e-12).all(axis=1)
    ids = faces.astype(np.int64).copy()

    s0, s1, s2 = sines[:, 0], sines[:, 1], sines[:, 2]
    pattA = (s1 > s0) & (s1 > s2)
    pattB = (~pattA) & (s0 > s1) & (s0 > s2)

    if pattA.any():
        old_a = angles[pattA].copy()
        old_s = sines[pattA].copy()
        old_id = ids[pattA].copy()
        angles[pattA] = old_a[:, [2, 0, 1]]
        sines[pattA] = old_s[:, [2, 0, 1]]
        ids[pattA] = old_id[:, [2, 0, 1]]
    if pattB.any():
        old_a = angles[pattB].copy()
        old_s = sines[pattB].copy()
        old_id = ids[pattB].copy()
        angles[pattB] = old_a[:, [1, 2, 0]]
        sines[pattB] = old_s[:, [1, 2, 0]]
        ids[pattB] = old_id[:, [1, 2, 0]]

    a0 = angles[:, 0]
    s0 = sines[:, 0]
    s1 = sines[:, 1]
    s2 = sines[:, 2]
    c0 = np.cos(a0)
    ratio = np.where(s2 > 0.0, s1 / s2.clip(min=1e-20), 1.0)
    cosine = c0 * ratio
    sine = s0 * ratio
    return ids, cosine, sine, valid


def lscm_chart(
    local_verts: Tensor,
    local_faces: Tensor,
    local_face_face: Tensor,
    pin_positions: "np.ndarray | None" = None,
) -> Tensor:
    """ABF parameterization on one chart (degenerate faces use plain LSCM rows; two pins fix gauge at pin_positions)."""
    verts_np = local_verts.detach().cpu().numpy().astype(np.float64)
    faces_np = local_faces.detach().cpu().numpy().astype(np.int64)
    Vc = verts_np.shape[0]
    Fc = faces_np.shape[0]

    if Vc < 3 or Fc == 0:
        return torch.zeros((Vc, 2), dtype=torch.float32, device=local_verts.device)

    loops = _mesh.chart_boundary_loops(local_faces, local_face_face)
    pin_a, pin_b = _pick_pins(loops, verts_np)

    if pin_positions is not None and pin_positions.shape == (Vc, 2):
        pa = pin_positions[pin_a]
        pb = pin_positions[pin_b]
        u_a, v_a = float(pa[0]), float(pa[1])
        u_b, v_b = float(pb[0]), float(pb[1])
    else:
        u_a, v_a = 0.0, 0.0
        u_b, v_b = 1.0, 0.0

    abf_ids, abf_cos, abf_sin, abf_valid = _abf_face_coefficients(verts_np, faces_np)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    vals_list: List[np.ndarray] = []

    # ABF rows for valid faces.
    valid_idx = np.nonzero(abf_valid)[0]
    if valid_idx.size:
        Nv = valid_idx.size
        id0 = abf_ids[valid_idx, 0]
        id1 = abf_ids[valid_idx, 1]
        id2 = abf_ids[valid_idx, 2]
        cosf = abf_cos[valid_idx]
        sinf = abf_sin[valid_idx]
        r_real = valid_idx * 2
        r_imag = valid_idx * 2 + 1
        ones = np.ones(Nv, dtype=np.float64)
        rows_list.extend([r_real] * 5)
        cols_list.extend([id0, id0 + Vc, id1, id1 + Vc, id2])
        vals_list.extend([cosf - 1.0, -sinf, -cosf, sinf, ones])
        rows_list.extend([r_imag] * 5)
        cols_list.extend([id0, id0 + Vc, id1, id1 + Vc, id2 + Vc])
        vals_list.extend([sinf, cosf - 1.0, -sinf, -cosf, ones])

    # Plain-LSCM rows for invalid (degenerate) faces.
    invalid_idx = np.nonzero(~abf_valid)[0]
    if invalid_idx.size:
        tri2d_inv = _triangle_local_2d(verts_np, faces_np[invalid_idx])
        twice_area_inv = (
            tri2d_inv[:, 1, 0] * tri2d_inv[:, 2, 1]
            - tri2d_inv[:, 1, 1] * tri2d_inv[:, 2, 0]
        )
        weight_inv = 1.0 / np.sqrt(2.0 * np.abs(twice_area_inv).clip(min=1e-20))
        r_real_inv = invalid_idx * 2
        r_imag_inv = invalid_idx * 2 + 1
        for j in range(3):
            jp1 = (j + 1) % 3
            jp2 = (j + 2) % 3
            a_j = (tri2d_inv[:, jp1, 0] - tri2d_inv[:, jp2, 0]) * weight_inv
            b_j = (tri2d_inv[:, jp1, 1] - tri2d_inv[:, jp2, 1]) * weight_inv
            v_idx = faces_np[invalid_idx, j]
            rows_list.extend([r_real_inv, r_real_inv, r_imag_inv, r_imag_inv])
            cols_list.extend([v_idx, v_idx + Vc, v_idx, v_idx + Vc])
            vals_list.extend([a_j, -b_j, b_j, a_j])

    rows = np.concatenate(rows_list) if rows_list else np.empty(0, dtype=np.int64)
    cols = np.concatenate(cols_list) if cols_list else np.empty(0, dtype=np.int64)
    vals = np.concatenate(vals_list) if vals_list else np.empty(0, dtype=np.float64)

    A_full = sp.csr_matrix((vals, (rows, cols)), shape=(2 * Fc, 2 * Vc))

    pin_cols = np.array([pin_a, pin_b, pin_a + Vc, pin_b + Vc], dtype=np.int64)
    pin_vals = np.array([u_a, u_b, v_a, v_b], dtype=np.float64)

    free_mask = np.ones(2 * Vc, dtype=bool)
    free_mask[pin_cols] = False
    free_cols = np.nonzero(free_mask)[0]

    A_pinned = A_full[:, pin_cols]
    A_free = A_full[:, free_cols]
    b = -(A_pinned @ pin_vals)

    # Singular system (under-constrained chart) falls back to ortho.
    fallback_to_ortho = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=sp.linalg.MatrixRankWarning)
            x_free = solve_least_squares(A_free, b)
        if not np.all(np.isfinite(x_free)):
            fallback_to_ortho = True
    except (sp.linalg.MatrixRankWarning, RuntimeError):
        fallback_to_ortho = True  # singular / under-constrained system

    if fallback_to_ortho:
        if pin_positions is not None and pin_positions.shape == (Vc, 2):
            uvs = pin_positions.astype(np.float32)
        else:
            uvs = _ortho_project(verts_np).astype(np.float32)
        return torch.from_numpy(uvs).to(local_verts.device)

    full = np.zeros(2 * Vc, dtype=np.float64)
    full[free_cols] = x_free
    full[pin_cols] = pin_vals
    uvs = np.stack([full[:Vc], full[Vc:]], axis=1).astype(np.float32)
    if not np.all(np.isfinite(uvs)):
        if pin_positions is not None and pin_positions.shape == (Vc, 2):
            uvs = pin_positions.astype(np.float32)
        else:
            uvs = _ortho_project(verts_np).astype(np.float32)

    return torch.from_numpy(uvs).to(local_verts.device)

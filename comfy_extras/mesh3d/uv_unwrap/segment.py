"""Adaptive cost-grow chart segmentation (CPU); numba optional, numpy path is nd-only."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):  # noqa: ARG001
        def deco(fn):
            return fn
        return deco if not args else args[0]


from .mesh import MeshData, face_edge_lengths


DEFAULT_W_NORMAL_DEVIATION = 2.0
DEFAULT_W_ROUNDNESS = 0.01
DEFAULT_W_STRAIGHTNESS = 6.0
DEFAULT_MAX_COST = 2.0
NORMAL_DEVIATION_HARD_CUTOFF = 0.707  # ~75°


@njit(cache=True, fastmath=False)
def _cost_grow_iter_jit(
    face_chart: np.ndarray, face_face: np.ndarray, face_normal: np.ndarray,
    face_area: np.ndarray, face_edge_len: np.ndarray,
    chart_basis: np.ndarray, chart_normal_sum: np.ndarray,
    chart_area: np.ndarray, chart_perim: np.ndarray,
    nd_cutoff: float, max_cost: float,
    w_nd: float, w_round: float, w_straight: float,
):
    """One grow iter: each unassigned face joins its lowest-cost adjacent chart if cost<max_cost."""
    F = face_chart.shape[0]
    best_chart_per_face = np.full(F, -1, dtype=np.int64)
    best_cost_per_face = np.full(F, np.inf, dtype=np.float32)

    for f in range(F):
        if face_chart[f] != -1:
            continue
        nx = face_normal[f, 0]
        ny = face_normal[f, 1]
        nz = face_normal[f, 2]
        af = face_area[f]
        for e0 in range(3):
            nb0 = face_face[f, e0]
            if nb0 < 0:
                continue
            c = face_chart[nb0]
            if c < 0:
                continue
            d = (nx * chart_basis[c, 0] + ny * chart_basis[c, 1] + nz * chart_basis[c, 2])
            nd = np.float32(1.0) - d
            if nd > np.float32(1.0):
                nd = np.float32(1.0)
            if nd < np.float32(0.0):
                nd = np.float32(0.0)
            if nd >= nd_cutoff:
                continue
            l_in = np.float32(0.0)
            l_out = np.float32(0.0)
            for e1 in range(3):
                nb1 = face_face[f, e1]
                el = face_edge_len[f, e1]
                if nb1 < 0:
                    l_out += el
                elif face_chart[nb1] == c:
                    l_in += el
                else:
                    l_out += el
            ca = chart_area[c]
            cp = chart_perim[c]
            new_perim = cp - l_in + l_out
            new_area = ca + af
            if cp <= np.float32(1e-20) or ca <= np.float32(1e-20):
                round_cost = np.float32(0.0)
            else:
                old_r = (cp * cp) / ca
                new_r = (new_perim * new_perim) / new_area
                if new_r <= np.float32(1e-20):
                    round_cost = np.float32(0.0)
                else:
                    round_cost = np.float32(1.0) - old_r / new_r
            denom = l_out + l_in
            if denom <= np.float32(1e-20):
                straight_cost = np.float32(0.0)
            else:
                ratio = (l_out - l_in) / denom
                if ratio < np.float32(0.0):
                    straight_cost = ratio
                else:
                    straight_cost = np.float32(0.0)
            cost = (w_nd * nd + w_round * round_cost + w_straight * straight_cost)
            if cost < best_cost_per_face[f]:
                best_cost_per_face[f] = cost
                best_chart_per_face[f] = c

    n_assigned = 0
    for f in range(F):
        if face_chart[f] != -1:
            continue
        if best_chart_per_face[f] < 0:
            continue
        if best_cost_per_face[f] > max_cost:
            continue
        c = best_chart_per_face[f]
        l_in = np.float32(0.0)
        l_out = np.float32(0.0)
        for e1 in range(3):
            nb1 = face_face[f, e1]
            el = face_edge_len[f, e1]
            if nb1 < 0:
                l_out += el
            elif face_chart[nb1] == c:
                l_in += el
            else:
                l_out += el
        af = face_area[f]
        face_chart[f] = c
        chart_normal_sum[c, 0] += face_normal[f, 0] * af
        chart_normal_sum[c, 1] += face_normal[f, 1] * af
        chart_normal_sum[c, 2] += face_normal[f, 2] * af
        chart_area[c] += af
        chart_perim[c] = chart_perim[c] - l_in + l_out
        nx = chart_normal_sum[c, 0]
        ny = chart_normal_sum[c, 1]
        nz = chart_normal_sum[c, 2]
        nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
        if nlen > np.float32(1e-20):
            chart_basis[c, 0] = nx / nlen
            chart_basis[c, 1] = ny / nlen
            chart_basis[c, 2] = nz / nlen
        n_assigned += 1
    return n_assigned


def _renumber(face_chart: np.ndarray, device) -> Tensor:
    unique = np.unique(face_chart[face_chart >= 0])
    if unique.size == 0:
        return torch.from_numpy(face_chart).to(device)
    remap = np.full(int(unique.max()) + 1, -1, dtype=np.int64)
    remap[unique] = np.arange(unique.size)
    out = face_chart.copy()
    mask = out >= 0
    out[mask] = remap[out[mask]]
    return torch.from_numpy(out).to(device)


def segment_charts(
    mesh: MeshData,
    max_cost: float = DEFAULT_MAX_COST,
    w_normal_deviation: float = DEFAULT_W_NORMAL_DEVIATION,
    w_roundness: float = DEFAULT_W_ROUNDNESS,
    w_straightness: float = DEFAULT_W_STRAIGHTNESS,
) -> Tensor:
    """Segment mesh into charts (parallel batch cost-grow). Returns face -> chart_id."""
    F = mesh.faces.shape[0]
    device = mesh.faces.device
    if F == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    face_normal = mesh.face_normal.detach().cpu().numpy().astype(np.float32)
    face_area = mesh.face_area.detach().cpu().numpy().astype(np.float32)
    face_centroid = mesh.face_centroid.detach().cpu().numpy().astype(np.float32)
    face_face = mesh.face_face.detach().cpu().numpy()

    face_chart = np.full(F, -1, dtype=np.int64)
    nd_cutoff = np.float32(NORMAL_DEVIATION_HARD_CUTOFF)
    nd_threshold = np.float32(min(max_cost / max(w_normal_deviation, 1e-6),
                                  NORMAL_DEVIATION_HARD_CUTOFF * 0.99))

    component = (mesh.component.detach().cpu().numpy()
                 if hasattr(mesh.component, "detach") else np.asarray(mesh.component))
    if component.size:
        _, first_idx = np.unique(component, return_index=True)
        initial_seeds = first_idx.astype(np.int64)
    else:
        initial_seeds = np.empty(0, dtype=np.int64)

    seed_faces: List[int] = [int(s) for s in initial_seeds.tolist()]
    if not seed_faces:
        seed_faces = [0]

    K = len(seed_faces)
    chart_basis = np.zeros((K, 3), dtype=np.float32)
    chart_normal_sum = np.zeros((K, 3), dtype=np.float32)
    chart_area = np.zeros(K, dtype=np.float32)
    chart_perim = np.zeros(K, dtype=np.float32)
    face_edge_len = (
        face_edge_lengths(mesh.vertices, mesh.faces)
        .detach().cpu().numpy()
    )
    for cid, sf in enumerate(seed_faces):
        face_chart[sf] = cid
        n = face_normal[sf]
        a = face_area[sf]
        chart_basis[cid] = n.astype(np.float32)
        chart_normal_sum[cid] = (n * a).astype(np.float32)
        chart_area[cid] = float(a)
        chart_perim[cid] = float(face_edge_len[sf].sum())

    if K == 0:
        return _renumber(face_chart, device)

    min_dist_to_seed = np.full(F, np.inf, dtype=np.float32)
    for sf in seed_faces:
        d = ((face_centroid - face_centroid[sf]) ** 2).sum(axis=-1)
        min_dist_to_seed = np.minimum(min_dist_to_seed, d)

    if _HAVE_NUMBA:
        # Multi-pass threshold schedule (low-cost first); tau cap 0.5 keeps cones ~30deg.
        tau_final = min(max_cost * 0.25, 0.5)
        thresholds = [t for t in (0.05, 0.1, 0.25) if t < tau_final] + [tau_final]
        max_inner = max(64, int(np.sqrt(F)) * 2)
        max_total_charts = max(F, 8000)
        outer_iter = 0
        while True:
            outer_iter += 1
            if outer_iter > F + 16:
                break
            for tau in thresholds:
                for _ in range(max_inner):
                    n_added = _cost_grow_iter_jit(
                        face_chart, face_face, face_normal, face_area, face_edge_len,
                        chart_basis, chart_normal_sum, chart_area, chart_perim,
                        nd_cutoff, np.float32(tau),
                        np.float32(w_normal_deviation),
                        np.float32(w_roundness),
                        np.float32(w_straightness),
                    )
                    if n_added == 0:
                        break
            if (face_chart == -1).sum() == 0:
                break
            if chart_basis.shape[0] >= max_total_charts:
                break
            unassigned_mask = face_chart == -1
            cand = np.where(unassigned_mask, min_dist_to_seed, np.float32(-np.inf))
            new_seed = int(np.argmax(cand))
            n = face_normal[new_seed]
            a = face_area[new_seed]
            chart_basis = np.vstack([chart_basis, n[None, :].astype(np.float32)])
            chart_normal_sum = np.vstack(
                [chart_normal_sum, (n * a)[None, :].astype(np.float32)]
            )
            chart_area = np.concatenate([chart_area, np.array([a], dtype=np.float32)])
            chart_perim = np.concatenate(
                [chart_perim, np.array([face_edge_len[new_seed].sum()], dtype=np.float32)]
            )
            face_chart[new_seed] = chart_basis.shape[0] - 1
            new_d = ((face_centroid - face_centroid[new_seed]) ** 2).sum(axis=-1)
            min_dist_to_seed = np.minimum(min_dist_to_seed, new_d)
    else:
        # Numpy fallback: nd-only adaptive grow.
        for _ in range(max(64, int(np.sqrt(F)) + 32)):
            unassigned = face_chart == -1
            if not unassigned.any():
                break
            u_idx = np.nonzero(unassigned)[0]
            nbs = face_face[u_idx]
            nbs_safe = np.where(nbs >= 0, nbs, 0)
            nb_charts = np.where(nbs >= 0, face_chart[nbs_safe], -1)
            valid = (nb_charts >= 0)
            if not valid.any():
                break
            nb_charts_safe = np.where(valid, nb_charts, 0)
            nb_basis = chart_basis[nb_charts_safe]
            d = (face_normal[u_idx][:, None, :] * nb_basis).sum(axis=-1)
            nd = np.where(valid, np.float32(1.0) - d, np.inf).clip(max=1.0)
            nd = np.where(nd >= nd_cutoff, np.inf, nd)
            best_e = np.argmin(nd, axis=1)
            best_cost = nd[np.arange(u_idx.size), best_e]
            best_c = nb_charts_safe[np.arange(u_idx.size), best_e]
            accept = (best_cost <= nd_threshold) & np.isfinite(best_cost)
            if not accept.any():
                break
            pick_u = u_idx[accept]
            pick_c = best_c[accept]
            face_chart[pick_u] = pick_c
            for f, c in zip(pick_u, pick_c):
                chart_normal_sum[c] += face_normal[f] * face_area[f]
                chart_area[c] += face_area[f]

    # Orphan cleanup: leftover faces join their best-matching neighbor's chart.
    if (face_chart == -1).any() and chart_basis.shape[0] > 0:
        while True:
            orphans = np.nonzero(face_chart == -1)[0]
            if orphans.size == 0:
                break
            nbs = face_face[orphans]
            nbs_safe = np.where(nbs >= 0, nbs, 0)
            nb_charts = np.where(nbs >= 0, face_chart[nbs_safe], -1)
            valid = (nb_charts >= 0)
            if not valid.any():
                break
            nb_charts_safe = np.where(valid, nb_charts, 0)
            nb_basis = chart_basis[nb_charts_safe]
            d = (face_normal[orphans][:, None, :] * nb_basis).sum(axis=-1)
            nd = np.where(valid, np.float32(1.0) - d, np.inf)
            best_e = np.argmin(nd, axis=1)
            best_c = nb_charts_safe[np.arange(orphans.size), best_e]
            assignable = valid.any(axis=1)
            if not assignable.any():
                break
            assign_idx = orphans[assignable]
            assign_c = best_c[assignable]
            face_chart[assign_idx] = assign_c
    if (face_chart == -1).any():
        new_singletons = np.nonzero(face_chart == -1)[0]
        for f in new_singletons:
            face_chart[int(f)] = chart_basis.shape[0]
            chart_basis = np.concatenate(
                [chart_basis, face_normal[int(f)].astype(np.float32)[None, :]],
                axis=0,
            )

    return _renumber(face_chart, device)


# ---- Parallel edge-collapse (PEC) chart clustering (CUDA) ----
def _combine_normal_cones(
    axis_a: Tensor, half_a: Tensor,
    axis_b: Tensor, half_b: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Merge two normal cones along the great circle from axis_a; returns (combined_axis, combined_half_angle, axis_angle)."""
    cos_angle = (axis_a * axis_b).sum(dim=-1).clamp(-1.0, 1.0)
    axis_angle = torch.acos(cos_angle)
    new_low = torch.minimum(-half_a, axis_angle - half_b)
    new_high = torch.maximum(half_a, axis_angle + half_b)
    new_half = (new_high - new_low) * 0.5
    rot_angle = (new_high + new_low) * 0.5
    b_perp = axis_b - axis_a * cos_angle.unsqueeze(-1)
    b_perp_norm = b_perp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b_perp_unit = b_perp / b_perp_norm
    new_axis = (
        axis_a * torch.cos(rot_angle).unsqueeze(-1)
        + b_perp_unit * torch.sin(rot_angle).unsqueeze(-1)
    )
    new_axis_norm = new_axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    new_axis = new_axis / new_axis_norm
    return new_axis, new_half, axis_angle


def _build_chart_edges(
    face_face: Tensor,
    chart_id: Tensor,
    face_edge_len: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Build chart-edge list (chart_pairs[E,2] with a<b, edge_length[E]); same-chart edges dropped, duplicates summed."""
    F = face_face.shape[0]
    device = face_face.device
    f_idx = torch.arange(F, device=device).repeat_interleave(3)
    nb = face_face.flatten()
    valid = nb >= 0
    f_idx = f_idx[valid]
    nb = nb[valid]
    el = face_edge_len.flatten()[valid]

    ca = chart_id[f_idx]
    cb = chart_id[nb]
    diff = ca != cb
    ca = ca[diff]
    cb = cb[diff]
    el = el[diff]
    if ca.numel() == 0:
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.empty(0, device=device),
        )

    lo = torch.minimum(ca, cb)
    hi = torch.maximum(ca, cb)
    V = int(chart_id.max().item()) + 1
    key = lo * V + hi
    sort_idx = torch.argsort(key)
    sorted_key = key[sort_idx]
    sorted_lo = lo[sort_idx]
    sorted_hi = hi[sort_idx]
    sorted_el = el[sort_idx]
    unique_key, inverse, counts = torch.unique(
        sorted_key, return_inverse=True, return_counts=True
    )
    n_unique = unique_key.shape[0]
    reduced_el = torch.zeros(n_unique, device=device, dtype=el.dtype)
    reduced_el.scatter_add_(0, inverse, sorted_el)
    first_idx = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        counts.cumsum(0)[:-1],
    ])
    pair_lo = sorted_lo[first_idx]
    pair_hi = sorted_hi[first_idx]
    chart_pairs = torch.stack([pair_lo, pair_hi], dim=1)
    return chart_pairs, reduced_el


def cluster_charts_pec(
    mesh: MeshData,
    max_cost: float = 0.7,
    max_iters: int = 1024,
) -> Tensor:
    """Parallel edge-collapse clustering; returns face_chart [F]. max_cost is the per-merge cutoff (~0.7 rad ~ 40deg)."""
    device = mesh.faces.device
    F = mesh.faces.shape[0]
    faces = mesh.faces.to(torch.long)
    vertices = mesh.vertices.to(torch.float32)
    face_normal = mesh.face_normal.to(torch.float32)
    face_face = mesh.face_face.to(torch.long)

    face_edge_len = face_edge_lengths(vertices, faces)

    chart_id = torch.arange(F, dtype=torch.long, device=device)
    chart_axis = face_normal.clone()
    chart_half = torch.zeros(F, dtype=torch.float32, device=device)

    for it in range(max_iters):
        edges, _ = _build_chart_edges(face_face, chart_id, face_edge_len)
        if edges.shape[0] == 0:
            break

        a = edges[:, 0]
        b = edges[:, 1]
        axis_a = chart_axis[a]
        axis_b = chart_axis[b]
        half_a = chart_half[a]
        half_b = chart_half[b]
        _, new_half, _ = _combine_normal_cones(axis_a, half_a, axis_b, half_b)
        cost = new_half.clone()

        # Pack (cost, edge_id) so scatter_reduce amin picks the right edge.
        E = edges.shape[0]
        N = int(chart_id.max().item()) + 1
        edge_ids = torch.arange(E, dtype=torch.long, device=device)
        cost_i32 = torch.clamp(cost * 1e6, max=2e9).to(torch.int64)
        key = (cost_i32 << 32) | edge_ids
        chart_min = torch.full((N,), (2**62), dtype=torch.long, device=device)
        chart_min.scatter_reduce_(0, a, key, reduce="amin", include_self=True)
        chart_min.scatter_reduce_(0, b, key, reduce="amin", include_self=True)

        # Mutual-min collapse: each chart in at most one merge per iter (winners are disjoint pairs).
        is_a_min = chart_min[a] == key
        is_b_min = chart_min[b] == key
        mutual = is_a_min & is_b_min
        within = cost <= max_cost
        winners = mutual & within

        n_merge = int(winners.sum().item())
        if n_merge == 0:
            break

        win_a = a[winners]
        win_b = b[winners]

        axis_a_w = chart_axis[win_a]
        half_a_w = chart_half[win_a]
        axis_b_w = chart_axis[win_b]
        half_b_w = chart_half[win_b]
        new_axis, new_half_w, _ = _combine_normal_cones(
            axis_a_w, half_a_w, axis_b_w, half_b_w,
        )
        chart_axis[win_a] = new_axis
        chart_half[win_a] = new_half_w

        remap = torch.arange(N, dtype=torch.long, device=device)
        remap[win_b] = win_a
        chart_id = remap[chart_id]

    _, inverse = torch.unique(chart_id, sorted=True, return_inverse=True)
    return inverse

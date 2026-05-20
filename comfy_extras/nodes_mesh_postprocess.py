import torch
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
import copy
import comfy.utils
import logging
import scipy

def get_mesh_batch_item(mesh, index):
    if hasattr(mesh, "vertex_counts") and mesh.vertex_counts is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = None
        if hasattr(mesh, "colors") and mesh.colors is not None:
            if hasattr(mesh, "color_counts") and mesh.color_counts is not None:
                color_count = int(mesh.color_counts[index].item())
                colors = mesh.colors[index, :color_count]
            else:
                colors = mesh.colors[index, :vertex_count]
        return vertices, faces, colors

    colors = None
    if hasattr(mesh, "colors") and mesh.colors is not None:
        colors = mesh.colors[index]
    return mesh.vertices[index], mesh.faces[index], colors

def pack_variable_mesh_batch(vertices, faces, colors=None):
    batch_size = len(vertices)
    max_vertices = max(v.shape[0] for v in vertices)
    max_faces = max(f.shape[0] for f in faces)

    packed_vertices = vertices[0].new_zeros((batch_size, max_vertices, vertices[0].shape[1]))
    packed_faces = faces[0].new_zeros((batch_size, max_faces, faces[0].shape[1]))
    vertex_counts = torch.tensor([v.shape[0] for v in vertices], device=vertices[0].device, dtype=torch.int64)
    face_counts = torch.tensor([f.shape[0] for f in faces], device=faces[0].device, dtype=torch.int64)

    for i, (v, f) in enumerate(zip(vertices, faces)):
        packed_vertices[i, :v.shape[0]] = v
        packed_faces[i, :f.shape[0]] = f

    mesh = Types.MESH(packed_vertices, packed_faces)
    mesh.vertex_counts = vertex_counts
    mesh.face_counts = face_counts

    if colors is not None:
        max_colors = max(c.shape[0] for c in colors)
        packed_colors = colors[0].new_zeros((batch_size, max_colors, colors[0].shape[1]))
        color_counts = torch.tensor([c.shape[0] for c in colors], device=colors[0].device, dtype=torch.int64)
        for i, c in enumerate(colors):
            packed_colors[i, :c.shape[0]] = c
        mesh.vertex_colors = packed_colors
        mesh.color_counts = color_counts

    return mesh


def paint_mesh_with_voxels(mesh, voxel_coords, voxel_colors, resolution):
    """
    Generic function to paint a mesh using nearest-neighbor colors from a sparse voxel field.
    """
    device = comfy.model_management.vae_offload_device()

    origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
    voxel_size = 1.0 / resolution

    # map voxels
    voxel_pos = voxel_coords.to(device).float() * voxel_size + origin
    verts = mesh.vertices.to(device).squeeze(0)
    voxel_colors = voxel_colors.to(device)

    voxel_pos_np = voxel_pos.numpy()
    verts_np = verts.numpy()

    tree = scipy.spatial.cKDTree(voxel_pos_np)

    # nearest neighbour k=1
    _, nearest_idx_np = tree.query(verts_np, k=1, workers=-1)

    nearest_idx = torch.from_numpy(nearest_idx_np).long()
    v_colors = voxel_colors[nearest_idx]

    # to [0, 1]
    srgb_colors = v_colors.clamp(0, 1)#(v_colors * 0.5 + 0.5).clamp(0, 1)

    # to Linear RGB (required for GLTF)
    linear_colors = torch.pow(srgb_colors, 2.2)

    final_colors = linear_colors.unsqueeze(0)

    out_mesh = copy.deepcopy(mesh)
    out_mesh.vertex_colors = final_colors

    return out_mesh

class PaintMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PaintMesh",
            display_name="Paint Mesh",
            category="latent/3d",
            description=(
                "Paints the mesh using colors from the input voxel field by matching each vertex "
                "to the nearest voxel color."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Voxel.Input("voxel_colors")
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
            ]
        )

    @classmethod
    def execute(cls, mesh, voxel_colors):
        voxels = voxel_colors
        coords = voxels.data
        colors = voxels.voxel_colors
        resolution = voxels.resolution

        if coords.shape[0] == 0:
            return IO.NodeOutput(paint_mesh_default_colors(mesh))

        mesh_batch_size = mesh.vertices.shape[0]

        if coords.shape[-1] == 4 and mesh_batch_size > 1:
            batch_idx = coords[:, 0].long()
            voxel_coords = coords[:, 1:]
            mesh_batch_size = mesh.vertices.shape[0]

            out_verts, out_faces, out_colors = [], [], []
            for i in range(mesh_batch_size):
                sel = batch_idx == i
                item_coords = voxel_coords[sel]
                item_colors = colors[sel]
                item_vertices, item_faces, _ = get_mesh_batch_item(mesh, i)
                item_mesh = Types.MESH(vertices=item_vertices.unsqueeze(0), faces=item_faces.unsqueeze(0))

                if item_coords.shape[0] == 0:
                    painted = paint_mesh_default_colors(item_mesh)
                else:
                    painted = paint_mesh_with_voxels(item_mesh, item_coords, item_colors, resolution=resolution)

                out_verts.append(painted.vertices.squeeze(0))
                out_faces.append(painted.faces.squeeze(0))
                out_colors.append(painted.vertex_colors.squeeze(0))

            out_mesh = pack_variable_mesh_batch(out_verts, out_faces, out_colors)
            return IO.NodeOutput(out_mesh)

        if coords.shape[-1] == 4:
            coords = coords[:, 1:]

        out_mesh = paint_mesh_with_voxels(mesh, coords, colors, resolution=resolution)
        return IO.NodeOutput(out_mesh)

def paint_mesh_default_colors(mesh):
    out_mesh = copy.copy(mesh)
    vertex_count = mesh.vertices.shape[1]
    out_mesh.vertex_colors = mesh.vertices.new_zeros((1, vertex_count, 3))
    return out_mesh


def fill_holes_fn(vertices, faces, max_perimeter=0.03):
    is_batched = vertices.ndim == 3
    if is_batched:
        v_list, f_list = [], []
        for i in range(vertices.shape[0]):
            v_i, f_i = fill_holes_fn(vertices[i], faces[i], max_perimeter)
            v_list.append(v_i)
            f_list.append(f_i)
        max_v = max(v.shape[0] for v in v_list)
        for i in range(len(v_list)):
            if v_list[i].shape[0] < max_v:
                pad = torch.zeros(max_v - v_list[i].shape[0], 3, device=v_list[i].device, dtype=v_list[i].dtype)
                v_list[i] = torch.cat([v_list[i], pad], dim=0)
        return torch.stack(v_list), torch.stack(f_list)

    device = vertices.device
    v = vertices
    f = faces

    if f.numel() == 0:
        return v, f

    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    edges_sorted, _ = torch.sort(edges, dim=1)
    max_v = v.shape[0]
    packed_undirected = edges_sorted[:, 0].long() * max_v + edges_sorted[:, 1].long()
    unique_packed, counts = torch.unique(packed_undirected, return_counts=True)
    boundary_packed = unique_packed[counts == 1]

    if boundary_packed.numel() == 0:
        return v, f

    boundary_mask = torch.isin(packed_undirected, boundary_packed)
    b_edges = edges_sorted[boundary_mask]

    adj = {}
    for i in range(b_edges.shape[0]):
        a = b_edges[i, 0].item()
        b = b_edges[i, 1].item()
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Trace all boundary loops
    loops = []
    visited = set()
    for start_node in adj.keys():
        if start_node in visited:
            continue
        curr = start_node
        prev = -1
        loop = []
        while curr not in visited:
            visited.add(curr)
            loop.append(curr)
            neighbors = adj[curr]
            candidates = [n for n in neighbors if n != prev]
            if not candidates:
                loop = []
                break
            next_node = candidates[0]
            prev, curr = curr, next_node
            if curr == start_node:
                loops.append(loop)
                break

    if not loops:
        return v, f

    # Mesh normal for winding orientation only
    face_normals = torch.linalg.cross(
        v[f[:, 1]] - v[f[:, 0]],
        v[f[:, 2]] - v[f[:, 0]],
        dim=-1
    )
    mesh_normal = face_normals.mean(dim=0)
    mesh_normal = mesh_normal / (torch.norm(mesh_normal) + 1e-8)

    # === FIX: Fill ALL boundary loops below perimeter threshold ===
    new_verts = []
    new_faces = []
    v_idx = v.shape[0]

    for loop in loops:
        loop_t = torch.tensor(loop, device=device, dtype=torch.long)
        loop_v = v[loop_t]

        # Perimeter check
        next_v = torch.roll(loop_v, -1, dims=0)
        diffs = loop_v - next_v
        perimeter = torch.norm(diffs, dim=1).sum().item()

        if perimeter > max_perimeter:
            continue

        # Ensure CCW winding consistent with mesh
        cross = torch.linalg.cross(loop_v, next_v, dim=-1)
        loop_normal = cross.sum(dim=0)
        loop_normal = loop_normal / (torch.norm(loop_normal) + 1e-8)
        if torch.dot(loop_normal, mesh_normal) < 0:
            loop = loop[::-1]
            loop_t = torch.tensor(loop, device=device, dtype=torch.long)
            loop_v = v[loop_t]

        if len(loop) == 3:
            new_faces.append([loop[0], loop[1], loop[2]])
        else:
            centroid = loop_v.mean(dim=0)
            new_verts.append(centroid)
            for i in range(len(loop)):
                new_faces.append([loop[i], loop[(i + 1) % len(loop)], v_idx])
            v_idx += 1

    if new_verts:
        v = torch.cat([v, torch.stack(new_verts)], dim=0)
    if new_faces:
        f = torch.cat([f, torch.tensor(new_faces, device=device, dtype=torch.long)], dim=0)

    return v, f

def _cleanup_mesh(verts, faces, min_angle_deg=0.5, max_aspect=100.0):
    if faces.numel() == 0:
        return verts, faces

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2
    l0 = torch.norm(e0, dim=-1)
    l1 = torch.norm(e1, dim=-1)
    l2 = torch.norm(e2, dim=-1)
    n = torch.cross(e0, e2, dim=-1)
    area = torch.norm(n, dim=-1)

    max_edge = torch.max(torch.max(l0, l1), l2)
    aspect = max_edge * max_edge / (2.0 * area + 1e-12)

    cos_a = (l1 * l1 + l2 * l2 - l0 * l0) / (2 * l1 * l2 + 1e-12)
    cos_b = (l0 * l0 + l2 * l2 - l1 * l1) / (2 * l0 * l2 + 1e-12)
    cos_c = (l0 * l0 + l1 * l1 - l2 * l2) / (2 * l0 * l1 + 1e-12)
    cos_all = torch.stack([cos_a, cos_b, cos_c], dim=-1)
    angles = torch.acos(torch.clamp(cos_all, -1, 1)) * 180 / np.pi

    good = (aspect < max_aspect) & (angles.min(dim=1)[0] > min_angle_deg) & (area > 1e-12)
    faces = faces[good]

    if faces.numel() == 0:
        return verts, faces

    used = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
    used[faces[:, 0]] = True
    used[faces[:, 1]] = True
    used[faces[:, 2]] = True

    remap = torch.full((verts.shape[0],), -1, dtype=torch.int64, device=verts.device)
    remap[used] = torch.arange(used.sum().item(), device=verts.device)
    verts = verts[used]
    faces = remap[faces]
    return verts, faces

def _pytorch_edge_errors_fast(verts, Q, edges, stabilizer, max_edge_length_sq, mesh_scale_sq):
    n_edges = edges.shape[0]
    dtype = verts.dtype
    if n_edges == 0:
        return (torch.empty((0, 3), dtype=dtype, device=verts.device),
                torch.empty((0,), dtype=dtype, device=verts.device),
                torch.zeros((0,), dtype=torch.bool, device=verts.device))

    device = verts.device
    mesh_scale = (mesh_scale_sq) ** 0.5

    va = edges[:, 0]
    vb = edges[:, 1]
    Q0 = Q[va]
    Q1 = Q[vb]
    Qe = Q0 + Q1

    A = Qe[:, :3, :3] + torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * stabilizer
    b = -Qe[:, :3, 3].unsqueeze(-1)

    dets = torch.det(A)
    good = dets.abs() > 1e-12
    opt = torch.zeros((n_edges, 3), dtype=dtype, device=device)

    if good.any():
        try:
            sol = torch.linalg.solve(A[good], b[good])
            opt[good] = sol.squeeze(-1)
        except Exception:
            good = torch.zeros_like(good)

    if (~good).any():
        bad_idx = torch.nonzero(~good, as_tuple=True)[0]
        opt[bad_idx] = (verts[va[bad_idx]] + verts[vb[bad_idx]]) * 0.5

    pa = verts[va]
    pb = verts[vb]
    el = torch.norm(pb - pa, dim=-1)
    dist_a = torch.norm(opt - pa, dim=-1)
    dist_b = torch.norm(opt - pb, dim=-1)
    wander_bad = (dist_a > 4.0 * el) | (dist_b > 4.0 * el)

    if wander_bad.any():
        bad_idx = torch.nonzero(wander_bad, as_tuple=True)[0]
        opt[bad_idx] = (verts[va[bad_idx]] + verts[vb[bad_idx]]) * 0.5

    v4 = torch.cat([opt, torch.ones((n_edges, 1), device=device, dtype=dtype)], dim=1)
    err = torch.abs(torch.einsum("ei,eij,ej->e", v4, Qe, v4))

    length_ok = el > mesh_scale * 1e-5
    error_ok = err < max_edge_length_sq
    nan_ok = ~torch.isnan(opt).any(dim=-1) & ~torch.isnan(err)
    valid = length_ok & error_ok & nan_ok

    return opt, err, valid


def _build_quadrics_fast(verts, faces):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    n = torch.cross(e1, e2, dim=-1)
    area = torch.norm(n, dim=-1)
    mask = area > 1e-12
    n_norm = torch.zeros_like(n)
    n_norm[mask] = n[mask] / area[mask].unsqueeze(-1)
    d = -(n_norm * v0).sum(dim=-1, keepdim=True)
    p = torch.cat([n_norm, d], dim=-1)
    K = torch.einsum("fi,fj->fij", p, p)
    K = K * area[:, None, None]
    V = verts.shape[0]
    Q = torch.zeros((V, 4, 4), dtype=verts.dtype, device=verts.device)
    K_flat = K.reshape(-1, 16)
    Q_flat = Q.reshape(V, 16)
    for corner in range(3):
        idx = faces[:, corner].unsqueeze(1).expand(-1, 16)
        Q_flat.scatter_add_(0, idx, K_flat)
    return Q_flat.reshape(V, 4, 4)


def _gpu_greedy_matching_fast(edges, err, v_alive, max_select):
    """Vectorized greedy matching.

    Selects an independent set of edges (no two share a vertex) preferring
    lowest error. Replaces _gpu_greedy_sampled's Python per-edge loop with
    two scatter_reduce calls.
    """
    device = edges.device
    n_edges = edges.shape[0]
    if n_edges == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    va = edges[:, 0]
    vb = edges[:, 1]
    num_verts = v_alive.shape[0]

    # Pack (error_bits, edge_idx) into one int64 so amin gives a unique winner.
    # err is non-negative finite float32 -> IEEE bits are monotonic.
    err32 = err.to(torch.float32).clamp(min=0).contiguous()
    err_bits = err32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    edge_idx = torch.arange(n_edges, device=device, dtype=torch.int64)
    key = (err_bits << 32) | edge_idx

    INT64_MAX = torch.iinfo(torch.int64).max
    best_key = torch.full((num_verts,), INT64_MAX, dtype=torch.int64, device=device)
    best_key.scatter_reduce_(0, va, key, reduce='amin', include_self=True)
    best_key.scatter_reduce_(0, vb, key, reduce='amin', include_self=True)

    # An edge wins iff it is the min-key edge incident to BOTH its endpoints
    # AND both endpoints are still alive.
    is_winner = (key == best_key[va]) & (key == best_key[vb]) & v_alive[va] & v_alive[vb]

    sel = torch.nonzero(is_winner, as_tuple=True)[0]

    if sel.numel() > max_select:
        sel_err = err[sel]
        top = torch.topk(sel_err, max_select, largest=False).indices
        sel = sel[top]

    return sel


def _qem_simplify_fast(vertices, faces_in, colors_in, normals_in, target_faces, device, max_edge_length=None):
    # Use float32 instead of float64. RTX-class consumer GPUs run FP32 ~32-64x
    # faster than FP64, and QEM only needs the stabilizer for conditioning.
    # Always copy=True so we can safely mutate verts/colors/normals in-place.
    verts = vertices.detach().to(device=device, dtype=torch.float32, copy=True)
    faces = faces_in.detach().to(device=device, dtype=torch.int64)
    colors = (
        colors_in.detach().to(device=device, dtype=torch.float32, copy=True)
        if colors_in is not None
        else None
    )
    # ADDED: Initialize normals
    normals = (
        normals_in.detach().to(device=device, dtype=torch.float32, copy=True)
        if normals_in is not None
        else None
    )

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    logging.debug(f"[QEM-fast] Input: {num_verts} verts, {num_faces} faces, target={target_faces}")

    v_alive = torch.ones(num_verts, dtype=torch.bool, device=device)
    f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)

    Q = _build_quadrics_fast(verts, faces)

    bbox = verts.max(dim=0)[0] - verts.min(dim=0)[0]
    mesh_scale = torch.norm(bbox).item()

    if max_edge_length is None or max_edge_length <= 0:
        max_edge_length = mesh_scale * 2.0

    if max_edge_length < 1e-6:
        max_edge_length = 1.0

    stabilizer = mesh_scale * mesh_scale * 0.001
    max_edge_length_sq = max_edge_length * max_edge_length
    mesh_scale_sq = mesh_scale * mesh_scale

    iteration = 0
    total_collapses = 0
    last_faces = num_faces

    while True:
        n_faces = int(f_alive.sum().item())

        if n_faces <= target_faces:
            break

        alive_v = torch.nonzero(v_alive, as_tuple=True)[0]
        alive_f = torch.nonzero(f_alive, as_tuple=True)[0]

        if alive_v.numel() <= 4 or alive_f.numel() == 0:
            break

        # Compact active mesh
        vmap = torch.full((num_verts,), -1, dtype=torch.int64, device=device)
        vmap[alive_v] = torch.arange(alive_v.numel(), device=device)

        active_faces = faces[alive_f]
        remapped = vmap[active_faces]

        # Extract edges
        e0 = remapped[:, [0, 1]]
        e1 = remapped[:, [1, 2]]
        e2 = remapped[:, [2, 0]]
        edges = torch.cat([e0, e1, e2], dim=0)
        edges = torch.sort(edges, dim=1)[0]
        edges = edges[(edges >= 0).all(dim=1)]
        edges = edges[edges[:, 0] != edges[:, 1]]

        if edges.shape[0] == 0:
            break

        # Deduplicate edges
        num_compact = alive_v.numel()
        packed = edges[:, 0].long() * num_compact + edges[:, 1].long()
        packed = torch.unique(packed)
        edges = torch.stack([packed // num_compact, packed % num_compact], dim=1)

        edges_orig = alive_v[edges]

        # Filter by edge length
        pa = verts[edges_orig[:, 0]]
        pb = verts[edges_orig[:, 1]]
        el = torch.norm(pb - pa, dim=-1)
        short_enough = el < max_edge_length

        if not short_enough.any():
            max_edge_length = el.max().item() * 2.0
            max_edge_length_sq = max_edge_length * max_edge_length
            short_enough = el < max_edge_length
            if not short_enough.any():
                break

        edges_orig = edges_orig[short_enough]
        if edges_orig.shape[0] == 0:
            break

        # Sample edges for processing
        n_edges_total = edges_orig.shape[0]
        max_edges_to_process = 10_000_000

        if n_edges_total > max_edges_to_process:
            perm = torch.randint(0, n_edges_total, (max_edges_to_process,), device=device)
            edges_orig = edges_orig[perm]
            n_edges = max_edges_to_process
        else:
            n_edges = n_edges_total

        optimal, err, valid = _pytorch_edge_errors_fast(
            verts, Q, edges_orig, stabilizer, max_edge_length_sq, mesh_scale_sq
        )

        if not valid.any():
            valid = torch.ones(n_edges, dtype=torch.bool, device=device)

        valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        edges_orig = edges_orig[valid_idx]
        optimal = optimal[valid_idx]
        err = err[valid_idx]

        faces_to_remove = n_faces - target_faces
        max_collapses = min(1_000_000, max(10_000, faces_to_remove // 4))

        sel = _gpu_greedy_matching_fast(edges_orig, err, v_alive, max_collapses)

        if sel.numel() == 0:
            break

        v_a = edges_orig[sel, 0]
        v_b = edges_orig[sel, 1]

        # Apply collapses
        verts[v_a] = optimal[sel]
        v_alive[v_b] = False
        Q[v_a] += Q[v_b]

        if colors is not None:
            colors[v_a] = (colors[v_a] + colors[v_b]) * 0.5

        if normals is not None:
            normals[v_a] = (normals[v_a] + normals[v_b]) * 0.5

        merge_map = torch.arange(num_verts, device=device)
        merge_map[v_b] = v_a
        faces = merge_map[faces]

        bad = (
            (faces[:, 0] == faces[:, 1])
            | (faces[:, 1] == faces[:, 2])
            | (faces[:, 2] == faces[:, 0])
        )
        f_alive &= ~bad

        total_collapses += v_a.numel()
        iteration += 1

        if iteration % 50 == 0 or n_faces < last_faces * 0.9:
            logging.debug(f"[QEM-fast] Iter {iteration}: {total_collapses} collapses, {int(f_alive.sum().item())} faces, applied {v_a.numel()}")
            last_faces = n_faces

        if iteration % 5 == 0 and int(f_alive.sum().item()) < num_faces * 0.5:
            faces = faces[f_alive]
            f_alive = torch.ones(faces.shape[0], dtype=torch.bool, device=device)
            num_faces = faces.shape[0]

        if iteration > 5000:
            break

    # Finalize
    final_v = verts[v_alive]
    final_c = colors[v_alive] if colors is not None else None
    final_n = normals[v_alive] if normals is not None else None

    remap = torch.full((num_verts,), -1, dtype=torch.int64, device=device)
    remap[v_alive] = torch.arange(int(v_alive.sum().item()), device=device)

    final_f_raw = faces[f_alive]
    alive_mask = v_alive[final_f_raw].all(dim=1)
    final_f_raw = final_f_raw[alive_mask]
    final_f = remap[final_f_raw]
    valid_faces = (final_f >= 0).all(dim=1)
    final_f = final_f[valid_faces]

    if final_f.numel() > 0:
        final_f = torch.unique(torch.sort(final_f, dim=1)[0], dim=0)

    if final_n is not None and final_f.numel() > 0:
        v0, v1, v2 = final_v[final_f[:, 0]], final_v[final_f[:, 1]], final_v[final_f[:, 2]]

        # calculate the actual normal of the simplified faces
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Get the average reference normal for each face
        n0, n1, n2 = final_n[final_f[:, 0]], final_n[final_f[:, 1]], final_n[final_f[:, 2]]
        ref_face_normals = (n0 + n1 + n2) / 3.0

        # Dot product to check if they point in the same direction
        dot_products = (face_normals * ref_face_normals).sum(dim=-1)

        # Flip the indices of ONLY the incorrect faces (swap vertex 1 and 2)
        wrong_way_mask = dot_products < 0
        final_f[wrong_way_mask] = final_f[wrong_way_mask][:, [0, 2, 1]]

    final_v, final_f = _cleanup_mesh(final_v, final_f, min_angle_deg=0.5, max_aspect=100.0)

    return final_v, final_f, final_c, final_n


def simplify_fn_fast(vertices, faces, colors=None, normals=None, target=100000, max_edge_length=None):
    if vertices.ndim == 3:
        v_list, f_list, c_list, n_list = [], [], [], []
        for i in range(vertices.shape[0]):
            c_in = colors[i] if colors is not None else None
            n_in = normals[i] if normals is not None else None
            v_i, f_i, c_i, n_i = simplify_fn_fast(vertices[i], faces[i], c_in, n_in, target, max_edge_length)
            v_list.append(v_i)
            f_list.append(f_i)
            if c_i is not None:
                c_list.append(c_i)
            if n_i is not None:
                n_list.append(n_i)

        c_out = torch.stack(c_list) if len(c_list) > 0 else None
        n_out = torch.stack(n_list) if len(n_list) > 0 else None
        return torch.stack(v_list), torch.stack(f_list), c_out, n_out

    if faces.shape[0] <= target:
        return vertices, faces, colors, normals

    device = vertices.device
    dtype = vertices.dtype
    face_dtype = faces.dtype
    color_dtype = colors.dtype if colors is not None else None
    # ADDED: Normal dtype
    normal_dtype = normals.dtype if normals is not None else None

    # Pass tensors directly; _qem_simplify_fast handles dtype/device + copy.
    out_v, out_f, out_c, out_n = _qem_simplify_fast(
        vertices, faces, colors, normals, target, device, max_edge_length
    )

    final_v = out_v.to(device=device, dtype=dtype)
    final_f = out_f.to(device=device, dtype=face_dtype)
    final_c = (
        out_c.to(device=device, dtype=color_dtype)
        if out_c is not None
        else None
    )
    final_n = (
        out_n.to(device=device, dtype=normal_dtype)
        if out_n is not None
        else None
    )
    return final_v, final_f, final_c, final_n

def compute_vertex_normals(verts, faces):
    """Computes area-weighted vertex normals."""
    # QUICK FIX: Ensure indices are int64 for scatter_add_
    faces_long = faces.to(torch.int64)

    i0, i1, i2 = faces_long[:, 0], faces_long[:, 1], faces_long[:, 2]
    v0, v1, v2 = verts[i0], verts[i1], verts[i2]

    # calculate unnormalized face normals (magnitude is proportional to area)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

    # accumulate face normals to vertices
    vertex_normals = torch.zeros_like(verts)
    vertex_normals.scatter_add_(0, i0.unsqueeze(-1).expand_as(face_normals), face_normals)
    vertex_normals.scatter_add_(0, i1.unsqueeze(-1).expand_as(face_normals), face_normals)
    vertex_normals.scatter_add_(0, i2.unsqueeze(-1).expand_as(face_normals), face_normals)

    return torch.nn.functional.normalize(vertex_normals, p=2, dim=-1, eps=1e-6)

def _process_mesh_batch(mesh, per_item_fn):
    """Handles list/batched/single mesh dispatching, color extraction, and stacking."""
    mesh = copy.deepcopy(mesh)

    def process_single(v, f, c, bar):
        v, f, c = per_item_fn(v, f, c)
        bar.update(1)
        return v, f, c

    is_list = isinstance(mesh.vertices, list)
    is_batched_tensor = not is_list and mesh.vertices.ndim == 3

    if is_list or is_batched_tensor:
        out_v, out_f, out_c = [], [], []
        bsz = len(mesh.vertices) if is_list else mesh.vertices.shape[0]
        bar = comfy.utils.ProgressBar(bsz)

        for i in range(bsz):
            v_i = mesh.vertices[i]
            f_i = mesh.faces[i]
            c_i = None
            if hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None:
                c_i = mesh.vertex_colors[i] if (isinstance(mesh.vertex_colors, list) or mesh.vertex_colors.ndim == 3) else mesh.vertex_colors

            v_i, f_i, c_i = process_single(v_i, f_i, c_i, bar)

            out_v.append(v_i)
            out_f.append(f_i)
            if c_i is not None:
                out_c.append(c_i)

        if all(v.shape == out_v[0].shape for v in out_v) and all(f.shape == out_f[0].shape for f in out_f):
            mesh.vertices = torch.stack(out_v)
            mesh.faces = torch.stack(out_f)
            if out_c:
                mesh.vertex_colors = torch.stack(out_c)
        else:
            mesh.vertices = out_v
            mesh.faces = out_f
            if out_c:
                mesh.vertex_colors = out_c
    else:
        c = mesh.vertex_colors if hasattr(mesh, 'vertex_colors') and mesh.vertex_colors is not None else None
        bar = comfy.utils.ProgressBar(1)
        v, f, c = process_single(mesh.vertices, mesh.faces, c, bar)
        mesh.vertices = v
        mesh.faces = f
        if c is not None:
            mesh.vertex_colors = c

    return IO.NodeOutput(mesh)


class DecimateMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="DecimateMesh",
            display_name="Decimate Mesh",
            category="latent/3d",
            description="Simplifies a mesh to a target face count using QEM.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("target_face_count", default=200_000, min=0, max=50_000_000,
                             tooltip="Target maximum number of faces. Set to 0 to disable."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, target_face_count):
        def _fn(v, f, c):
            if target_face_count > 0 and f.shape[0] > target_face_count:
                n = compute_vertex_normals(v, f)
                v, f, c, _ = simplify_fn_fast(v, f, colors=c, normals=n, target=target_face_count)
            return v, f, c
        return _process_mesh_batch(mesh, _fn)


class FillHoles(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FillHoles",
            display_name="Fill Holes",
            category="latent/3d",
            description="Fills holes in a mesh up to a maximum perimeter threshold.",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("max_perimeter", default=0.03, min=0.0, step=0.0001,
                               tooltip="Maximum hole perimeter to fill. Set to 0 to disable."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, max_perimeter):
        def _fn(v, f, c):
            if max_perimeter > 0:
                v, f = fill_holes_fn(v, f, max_perimeter=max_perimeter)
            return v, f, c
        return _process_mesh_batch(mesh, _fn)

class PostProcessMeshExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FillHoles,
            DecimateMesh,
            PaintMesh
        ]


async def comfy_entrypoint() -> PostProcessMeshExtension:
    return PostProcessMeshExtension()

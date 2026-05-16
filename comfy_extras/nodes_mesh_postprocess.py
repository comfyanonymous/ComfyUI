import torch
import numpy as np
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO
import copy
import comfy.utils
import logging
import scipy

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
                IO.Voxel.Input("color_voxel")
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
            ]
        )

    @classmethod
    def execute(cls, mesh, color_voxel):
        """
        Generic function to paint a mesh using nearest-neighbor colors from a sparse voxel field.
        """
        resolution = color_voxel.resolution
        voxel_colors = color_voxel.voxel_colors
        voxel_coords = color_voxel.data


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

        return IO.NodeOutput(out_mesh)


def paint_mesh_default_colors(mesh):
    out_mesh = copy.copy(mesh)
    vertex_count = mesh.vertices.shape[1]
    out_mesh.vertex_colors = mesh.vertices.new_zeros((1, vertex_count, 3))
    return out_mesh

def _pytorch_edge_errors(verts, Q, edges, stabilizer, max_edge_length_sq, mesh_scale_sq):
    n_edges = edges.shape[0]
    if n_edges == 0:
        return (torch.empty((0, 3), dtype=torch.float64, device=verts.device),
                torch.empty((0,), dtype=torch.float64, device=verts.device),
                torch.zeros((0,), dtype=torch.bool, device=verts.device))

    device = verts.device
    mesh_scale = (mesh_scale_sq) ** 0.5

    va = edges[:, 0]
    vb = edges[:, 1]
    Q0 = Q[va]
    Q1 = Q[vb]
    Qe = Q0 + Q1

    A = Qe[:, :3, :3] + torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0) * stabilizer
    b = -Qe[:, :3, 3].unsqueeze(-1)

    dets = torch.det(A)
    good = dets.abs() > 1e-12
    opt = torch.zeros((n_edges, 3), dtype=torch.float64, device=device)

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

    v4 = torch.cat([opt, torch.ones((n_edges, 1), device=device, dtype=torch.float64)], dim=1)
    err = torch.abs(torch.einsum("ei,eij,ej->e", v4, Qe, v4))

    length_ok = el > mesh_scale * 1e-5
    error_ok = err < max_edge_length_sq
    nan_ok = ~torch.isnan(opt).any(dim=-1) & ~torch.isnan(err)
    valid = length_ok & error_ok & nan_ok

    return opt, err, valid

def _build_quadrics(verts, faces):
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
    Q = torch.zeros((V, 4, 4), dtype=torch.float64, device=verts.device)
    K_flat = K.reshape(-1, 16)
    Q_flat = Q.reshape(V, 16)
    for corner in range(3):
        idx = faces[:, corner].unsqueeze(1).expand(-1, 16)
        Q_flat.scatter_add_(0, idx, K_flat)
    return Q_flat.reshape(V, 4, 4)


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


def _build_vertex_face_csr(faces, num_verts):
    vf_verts = faces.view(-1)
    vf_faces = torch.arange(faces.shape[0], device=faces.device).repeat_interleave(3)

    sort_idx = torch.argsort(vf_verts, stable=True)
    sorted_verts = vf_verts[sort_idx]
    sorted_faces = vf_faces[sort_idx]

    unique_verts, counts = torch.unique_consecutive(sorted_verts, return_counts=True)

    ptrs = torch.zeros(num_verts + 1, dtype=torch.int64, device=faces.device)
    ptrs[unique_verts + 1] = counts
    ptrs = torch.cumsum(ptrs, dim=0)

    return sorted_faces, ptrs


def _get_vertex_faces(v, face_indices, vert_ptrs):
    start = vert_ptrs[v]
    end = vert_ptrs[v + 1]
    return face_indices[start:end]

def _gpu_greedy_sampled(edges, errors, v_alive, max_select):
    device = edges.device
    n_edges = edges.shape[0]

    if n_edges == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    # Sort by error
    sorted_idx = torch.argsort(errors)
    sorted_edges = edges[sorted_idx]

    # Sample K edges from the sorted list
    # This gives us diverse edges spread across the mesh
    K = min(max_select * 20, n_edges)
    if K < n_edges:
        sample_positions = torch.linspace(0, n_edges - 1, K, device=device).long()
        sampled_edges = sorted_edges[sample_positions]
        sampled_idx = sorted_idx[sample_positions]
    else:
        sampled_edges = sorted_edges
        sampled_idx = sorted_idx

    # Greedy selection on GPU
    used = torch.zeros(v_alive.shape[0], dtype=torch.bool, device=device)
    used[~v_alive] = True

    selected = []
    batch_size = 8192

    for start in range(0, sampled_edges.shape[0], batch_size):
        end = min(start + batch_size, sampled_edges.shape[0])
        batch = sampled_edges[start:end]
        batch_idx = sampled_idx[start:end]

        va = batch[:, 0]
        vb = batch[:, 1]

        # Vectorized free check
        free = ~used[va] & ~used[vb]

        if not free.any():
            continue

        # Get free edges
        free_local = torch.nonzero(free, as_tuple=True)[0]
        free_edges = batch[free_local]
        free_idx = batch_idx[free_local]

        # Process free edges greedily but in larger chunks
        # Transfer to CPU but only the small free subset
        free_va = free_edges[:, 0].cpu().numpy()
        free_vb = free_edges[:, 1].cpu().numpy()
        free_edges_idx = free_idx.cpu().numpy()

        for i in range(len(free_va)):
            a = int(free_va[i])
            b = int(free_vb[i])
            if not used[a].item() and not used[b].item():
                selected.append(int(free_edges_idx[i]))
                used[a] = True
                used[b] = True
                if len(selected) >= max_select:
                    return torch.tensor(selected, dtype=torch.int64, device=device)

    if len(selected) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.tensor(selected, dtype=torch.int64, device=device)

def _qem_simplify(verts_np, faces_np, colors_np, target_faces, device, max_edge_length=None):
    verts = torch.from_numpy(verts_np).to(device=device, dtype=torch.float64)
    faces = torch.from_numpy(faces_np).to(device=device, dtype=torch.int64)
    colors = (
        torch.from_numpy(colors_np).to(device=device, dtype=torch.float64)
        if colors_np is not None
        else None
    )

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    logging.debug(f"[QEM] Input: {num_verts} verts, {num_faces} faces, target={target_faces}")

    v_alive = torch.ones(num_verts, dtype=torch.bool, device=device)
    f_alive = torch.ones(num_faces, dtype=torch.bool, device=device)

    Q = _build_quadrics(verts, faces)

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
        max_edges_to_process = 10_000_000  # 10M edges per iteration

        if n_edges_total > max_edges_to_process:
            perm = torch.randperm(n_edges_total, device=device)[:max_edges_to_process]
            edges_orig = edges_orig[perm]
            n_edges = max_edges_to_process
        else:
            n_edges = n_edges_total

        optimal, err, valid = _pytorch_edge_errors(
            verts, Q, edges_orig, stabilizer, max_edge_length_sq, mesh_scale_sq
        )

        if not valid.any():
            valid = torch.ones(n_edges, dtype=torch.bool, device=device)

        valid_idx = torch.nonzero(valid, as_tuple=True)[0]
        edges_orig = edges_orig[valid_idx]
        optimal = optimal[valid_idx]
        err = err[valid_idx]

        # KEY: Much larger batch size
        faces_to_remove = n_faces - target_faces
        max_collapses = min(50000, max(1000, faces_to_remove // 20))

        sel = _gpu_greedy_sampled(edges_orig, err, v_alive, max_collapses)

        if sel.numel() == 0:
            break

        v_a = edges_orig[sel, 0]
        v_b = edges_orig[sel, 1]

        # Build adjacency
        face_indices, vert_ptrs = _build_vertex_face_csr(active_faces, num_verts)

        # Build (edge, face) pairs
        pair_edge_idx = []
        pair_face_idx = []

        va_cpu = v_a.cpu()
        vb_cpu = v_b.cpu()

        for ei, (vai, vbi) in enumerate(zip(va_cpu, vb_cpu)):
            f_va = _get_vertex_faces(vai.item(), face_indices, vert_ptrs)
            f_vb = _get_vertex_faces(vbi.item(), face_indices, vert_ptrs)

            faces_vb = active_faces[f_vb]
            mask_b = (faces_vb[:, 0] != vai) & (faces_vb[:, 1] != vai) & (faces_vb[:, 2] != vai)
            f_vb_valid = f_vb[mask_b]

            faces_va = active_faces[f_va]
            mask_a = (faces_va[:, 0] != vbi) & (faces_va[:, 1] != vbi) & (faces_va[:, 2] != vbi)
            f_va_valid = f_va[mask_a]

            all_faces = torch.cat([f_vb_valid, f_va_valid])
            if all_faces.numel() > 0:
                pair_edge_idx.extend([ei] * all_faces.numel())
                pair_face_idx.extend(all_faces.cpu().tolist())

        keep_mask = torch.ones(v_a.numel(), dtype=torch.bool, device=device)

        if not keep_mask.any():
            break

        keep_idx = torch.nonzero(keep_mask, as_tuple=True)[0]
        v_a = v_a[keep_idx]
        v_b = v_b[keep_idx]
        sel = sel[keep_idx]

        # Apply collapses
        verts[v_a] = optimal[sel]
        v_alive[v_b] = False
        Q[v_a] += Q[v_b]

        if colors is not None:
            colors[v_a] = (colors[v_a] + colors[v_b]) * 0.5

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

        # Log only every 50 iterations to reduce sync overhead
        if iteration % 50 == 0 or n_faces < last_faces * 0.9:
            logging.debug(f"[QEM] Iter {iteration}: {total_collapses} collapses, {int(f_alive.sum().item())} faces, applied {v_a.numel()}")
            last_faces = n_faces

        # Periodic compaction
        if iteration % 5 == 0 and int(f_alive.sum().item()) < num_faces * 0.5:
            faces = faces[f_alive]
            f_alive = torch.ones(faces.shape[0], dtype=torch.bool, device=device)
            num_faces = faces.shape[0]

        if iteration > 5000:
            break

    # Finalize
    final_v = verts[v_alive]
    final_c = colors[v_alive] if colors is not None else None

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

    final_v, final_f = _cleanup_mesh(final_v, final_f, min_angle_deg=0.5, max_aspect=100.0)

    return final_v, final_f, final_c

def simplify_fn(vertices, faces, colors=None, target=100000, max_edge_length=None):
    if vertices.ndim == 3:
        v_list, f_list, c_list = [], [], []
        for i in range(vertices.shape[0]):
            c_in = colors[i] if colors is not None else None
            v_i, f_i, c_i = simplify_fn(vertices[i], faces[i], c_in, target, max_edge_length)
            v_list.append(v_i)
            f_list.append(f_i)
            if c_i is not None:
                c_list.append(c_i)
        c_out = torch.stack(c_list) if len(c_list) > 0 else None
        return torch.stack(v_list), torch.stack(f_list), c_out

    if faces.shape[0] <= target:
        return vertices, faces, colors

    device = vertices.device
    dtype = vertices.dtype

    verts_np = vertices.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.int64)
    colors_np = (
        colors.detach().cpu().numpy().astype(np.float64)
        if colors is not None
        else None
    )

    out_v, out_f, out_c = _qem_simplify(
        verts_np, faces_np, colors_np, target, device, max_edge_length
    )

    final_v = out_v.to(device=device, dtype=dtype)
    final_f = out_f.to(device=device, dtype=faces.dtype)
    final_c = (
        out_c.to(device=device, dtype=colors.dtype)
        if out_c is not None
        else None
    )
    return final_v, final_f, final_c

def fill_holes_fn(vertices, faces, max_perimeter=0.03):
    is_batched = vertices.ndim == 3
    if is_batched:
        v_list, f_list = [],[]
        for i in range(vertices.shape[0]):
            v_i, f_i = fill_holes_fn(vertices[i], faces[i], max_perimeter)
            v_list.append(v_i)
            f_list.append(f_i)
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

    packed_directed_sorted = edges[:, 0].min(edges[:, 1]).long() * max_v + edges[:, 0].max(edges[:, 1]).long()
    is_boundary = torch.isin(packed_directed_sorted, boundary_packed)
    b_edges = edges[is_boundary]

    adj = {u.item(): v_idx.item() for u, v_idx in b_edges}

    loops =[]
    visited = set()

    for start_node in adj.keys():
        if start_node in visited:
            continue

        curr = start_node
        loop = []

        while curr not in visited:
            visited.add(curr)
            loop.append(curr)
            curr = adj.get(curr, -1)

            if curr == -1:
                loop = []
                break
            if curr == start_node:
                loops.append(loop)
                break

    new_verts =[]
    new_faces = []
    v_idx = v.shape[0]

    for loop in loops:
        loop_t = torch.tensor(loop, device=device, dtype=torch.long)
        loop_v = v[loop_t]

        diffs = loop_v - torch.roll(loop_v, shifts=-1, dims=0)
        perimeter = torch.norm(diffs, dim=1).sum().item()

        if perimeter <= max_perimeter:
            new_verts.append(loop_v.mean(dim=0))

            for i in range(len(loop)):
                new_faces.append([loop[(i + 1) % len(loop)], loop[i], v_idx])
            v_idx += 1

    if new_verts:
        v = torch.cat([v, torch.stack(new_verts)], dim=0)
        f = torch.cat([f, torch.tensor(new_faces, device=device, dtype=torch.long)], dim=0)

    return v, f


def make_double_sided(vertices, faces):
    is_batched = vertices.ndim == 3
    if is_batched:
        f_list = []
        for i in range(faces.shape[0]):
            f_inv = faces[i][:, [0, 2, 1]]
            f_list.append(torch.cat([faces[i], f_inv], dim=0))
        return vertices, torch.stack(f_list)

    faces_inv = faces[:, [0, 2, 1]]
    return vertices, torch.cat([faces, faces_inv], dim=0)

class PostProcessMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PostProcessMesh",
            category="latent/3d",
            description=(
            "Applies a sequence of mesh post-processing operations including optional hole filling"
            " and mesh simplification to a target face count."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("target_face_count", default=1_000_000, min=0, max=50_000_000,
                             tooltip="Target maximum number of faces after mesh simplification. Set to 0 to disable simplification."),
                IO.Float.Input("fill_holes_perimeter", default=0.03, min=0.0, step=0.0001,
                               tooltip=(
                                "Maximum hole perimeter threshold for filling holes in the mesh. "
                                "Smaller values only fill tiny holes, larger values fill larger gaps. "
                                "Set to 0 to disable hole filling."))
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
            ]
        )

    @classmethod
    def execute(cls, mesh, target_face_count, fill_holes_perimeter):
        mesh = copy.deepcopy(mesh)

        def process_single(v, f, c, bar):
            if fill_holes_perimeter > 0:
                v, f = fill_holes_fn(v, f, max_perimeter=fill_holes_perimeter)
            bar.update(1)

            if target_face_count > 0 and f.shape[0] > target_face_count:
                v, f, c = simplify_fn(v, f, colors=c, target=target_face_count)
            bar.update(1)

            v, f = make_double_sided(v, f)
            bar.update(1)
            return v, f, c

        is_list = isinstance(mesh.vertices, list)
        is_batched_tensor = not is_list and mesh.vertices.ndim == 3

        if is_list or is_batched_tensor:
            out_v, out_f, out_c = [], [],[]
            bsz = len(mesh.vertices) if is_list else mesh.vertices.shape[0]
            bar = comfy.utils.ProgressBar(3 * bsz)

            for i in range(bsz):
                v_i = mesh.vertices[i]
                f_i = mesh.faces[i]

                # Safely grab colors if they exist
                c_i = None
                if hasattr(mesh, 'colors') and mesh.colors is not None:
                    c_i = mesh.colors[i] if (isinstance(mesh.colors, list) or mesh.colors.ndim == 3) else mesh.colors

                v_i, f_i, c_i = process_single(v_i, f_i, c_i, bar)

                out_v.append(v_i)
                out_f.append(f_i)
                if c_i is not None:
                    out_c.append(c_i)

            # If the output meshes happen to have the exact same shape, stack them nicely.
            # Otherwise, just leave them as a List! (ComfyUI native standard)
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
            # Single Unbatched Mesh[V, 3]
            c = mesh.colors if hasattr(mesh, 'colors') and mesh.colors is not None else None
            v, f, c = process_single(mesh.vertices, mesh.faces, c)
            mesh.vertices = v
            mesh.faces = f
            if c is not None:
                mesh.vertex_colors = c

        return IO.NodeOutput(mesh)



class PostProcessMeshExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            PostProcessMesh,
            PaintMesh
        ]


async def comfy_entrypoint() -> PostProcessMeshExtension:
    return PostProcessMeshExtension()

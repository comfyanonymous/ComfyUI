import torch
import numpy as np
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
import copy
import comfy.utils
import comfy.model_management
from server import PromptServer
from comfy_extras.mesh3d.postprocess.qem_decimate import QEMConfig, qem_decimate_simplify, qem_cluster_decimate, _compute_vertex_normals
from comfy_extras.mesh3d.postprocess.remesh import remesh_narrow_band_dc, _point_tri_closest
from comfy_extras.mesh3d.uv_unwrap import mesh as _uv_mesh
from comfy_extras.mesh3d.uv_unwrap import segment as _uv_seg
from comfy_extras.mesh3d.uv_unwrap import parameterize as _uv_param
from comfy_extras.mesh3d.uv_unwrap import pack as _uv_pack
import logging
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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
    """Paint a mesh using nearest-neighbor colors from a sparse voxel field."""
    device = comfy.model_management.vae_offload_device()

    origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
    voxel_size = 1.0 / resolution

    voxel_pos = voxel_coords.to(device).float() * voxel_size + origin
    verts = mesh.vertices.to(device).squeeze(0)
    voxel_colors = voxel_colors.to(device)

    voxel_pos_np = voxel_pos.numpy()
    verts_np = verts.numpy()

    tree = cKDTree(voxel_pos_np)
    _, nearest_idx_np = tree.query(verts_np, k=1, workers=-1)

    nearest_idx = torch.from_numpy(nearest_idx_np).long()
    v_colors = voxel_colors[nearest_idx]
    # Voxel field may carry full PBR; vertex colors use only base_color RGB.
    if v_colors.shape[-1] > 3:
        v_colors = v_colors[:, :3]

    srgb_colors = v_colors.clamp(0, 1)

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
            description="Paints each mesh vertex with its nearest voxel color from the input voxel field.",
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


def _rasterize_uv_barycentric(faces_np, uvs_np, texture_size):
    """Rasterize the mesh in UV space (tiled point-in-triangle, pure torch). Returns per-texel
    face index [H,W], barycentric coords [H,W,3] and coverage mask [H,W], on the torch device.
    Interpolate any per-vertex attribute from these with _interp_vertex_attr."""
    dev = comfy.model_management.get_torch_device()
    H = W = int(texture_size)
    face_idx = torch.zeros((H, W), dtype=torch.long, device=dev)
    bary = torch.zeros((H, W, 3), device=dev)
    cov = torch.zeros((H, W), dtype=torch.bool, device=dev)
    if faces_np.shape[0] == 0:
        return face_idx, bary, cov

    uvs = torch.from_numpy(np.ascontiguousarray(uvs_np, dtype=np.float32)).to(dev)
    faces = torch.from_numpy(np.ascontiguousarray(faces_np).astype(np.int64)).to(dev)

    # GL convention: window coord = uv * resolution, coverage tested at texel centre.
    tri_uv = (uvs * float(W))[faces]              # [F,3,2]
    x0, y0 = tri_uv[:, 0, 0], tri_uv[:, 0, 1]
    x1, y1 = tri_uv[:, 1, 0], tri_uv[:, 1, 1]
    x2, y2 = tri_uv[:, 2, 0], tri_uv[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegen = denom.abs() > 1e-20

    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, W - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, W - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, H - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, H - 1).long()

    # Tile so point-in-triangle only runs over the triangles whose bbox hits the tile.
    TILE = 64
    eps = 1e-6
    for ty in range(0, H, TILE):
        ty_end = min(ty + TILE, H)
        for tx in range(0, W, TILE):
            tx_end = min(tx + TILE, W)
            tri_mask = (nondegen & (xmin < tx_end) & (xmax >= tx)
                        & (ymin < ty_end) & (ymax >= ty))
            if not tri_mask.any():
                continue
            idx = torch.nonzero(tri_mask, as_tuple=True)[0]
            ys = torch.arange(ty, ty_end, dtype=torch.float32, device=dev) + 0.5
            xs = torch.arange(tx, tx_end, dtype=torch.float32, device=dev) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")            # [th,tw]
            sx0, sy0 = x0[idx][:, None, None], y0[idx][:, None, None]
            sx1, sy1 = x1[idx][:, None, None], y1[idx][:, None, None]
            sx2, sy2 = x2[idx][:, None, None], y2[idx][:, None, None]
            sden = denom[idx][:, None, None]
            b0 = ((sy1 - sy2) * (xx - sx2) + (sx2 - sx1) * (yy - sy2)) / sden
            b1 = ((sy2 - sy0) * (xx - sx2) + (sx0 - sx2) * (yy - sy2)) / sden
            b2 = 1.0 - b0 - b1
            inside = (b0 >= -eps) & (b1 >= -eps) & (b2 >= -eps)       # [K,th,tw]
            if not inside.any():
                continue
            hit = inside.any(dim=0)                                    # [th,tw]
            sel = inside.int().argmax(dim=0)                           # [th,tw] first covering local tri
            bsel = torch.stack([b0.gather(0, sel[None]).squeeze(0),
                                b1.gather(0, sel[None]).squeeze(0),
                                b2.gather(0, sel[None]).squeeze(0)], dim=-1)   # [th,tw,3]
            face_idx[ty:ty_end, tx:tx_end][hit] = idx[sel][hit]      # slice is a view → writes through
            bary[ty:ty_end, tx:tx_end][hit] = bsel[hit]
            cov[ty:ty_end, tx:tx_end] |= hit

    return face_idx, bary, cov


def _interp_vertex_attr(attr_v, faces, face_idx, bary, mask):
    """Interpolate a per-vertex attribute [N,C] into a [H,W,C] map via a rasterized
    (face_idx, bary, mask). Uncovered texels stay zero."""
    H, W = mask.shape
    out = torch.zeros((H, W, attr_v.shape[1]), device=attr_v.device, dtype=attr_v.dtype)
    if mask.any():
        vtri = attr_v[faces[face_idx[mask]]]                          # [K,3,C]
        out[mask] = (bary[mask][:, :, None] * vtri).sum(1)
    return out


def _bake_position_map(verts_np, faces_np, uvs_np, texture_size):
    """Barycentric-interpolate a per-vertex vec3 (world position, or any vec3 e.g. normals)
    at each covered texel. Returns (attr_map [H,W,3] float32, mask [H,W] bool)."""
    dev = comfy.model_management.get_torch_device()
    H = W = int(texture_size)
    if faces_np.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.float32), np.zeros((H, W), dtype=bool)

    face_idx, bary, mask = _rasterize_uv_barycentric(faces_np, uvs_np, texture_size)
    verts = torch.from_numpy(np.ascontiguousarray(verts_np, dtype=np.float32)).to(dev)
    faces = torch.from_numpy(np.ascontiguousarray(faces_np).astype(np.int64)).to(dev)
    attr = _interp_vertex_attr(verts, faces, face_idx, bary, mask)
    return attr.cpu().numpy(), mask.cpu().numpy()


def _trilinear_sample_sparse(positions, voxel_coords_np, color_np, resolution):
    """Normalized trilinear over a SPARSE voxel field (only occupied corners of the 8,
    renormalized; matches official o_voxel.to_glb but without dense-volume zero-bleed).
    Returns (vals [K,C] float64, ok [K] bool); ok=False where no corner is occupied."""
    R = int(resolution)
    origin = -0.5
    voxel_size = 1.0 / R
    # Cell-CENTER convention: coord c sits at origin+(c+0.5)*voxel_size (matches
    # official grid_sample_3d); the -0.5 puts integer gc on centres so the 8 corners
    # bracket the query (omitting it bleeds colour at boundaries/thin features).
    gc = (positions.astype(np.float64) - origin) / voxel_size - 0.5
    base = np.floor(gc).astype(np.int64)
    frac = gc - base

    vc = voxel_coords_np.astype(np.int64)
    occ_keys = (vc[:, 0] * R + vc[:, 1]) * R + vc[:, 2]
    order = np.argsort(occ_keys)
    occ_sorted = occ_keys[order]

    K = positions.shape[0]
    C = color_np.shape[1]
    acc = np.zeros((K, C), dtype=np.float64)
    wsum = np.zeros((K, 1), dtype=np.float64)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cx = base[:, 0] + dx
                cy = base[:, 1] + dy
                cz = base[:, 2] + dz
                inb = (cx >= 0) & (cx < R) & (cy >= 0) & (cy < R) & (cz >= 0) & (cz < R)
                key = (cx * R + cy) * R + cz
                ins = np.clip(np.searchsorted(occ_sorted, key), 0, len(occ_sorted) - 1)
                matched = inb & (occ_sorted[ins] == key)
                idx = order[ins]                              # garbage where !matched
                w = np.where(matched, wx * wy * wz, 0.0)[:, None]
                acc += w * color_np[idx]                      # w=0 cancels garbage rows
                wsum += w
    ok = wsum[:, 0] > 1e-8
    vals = np.zeros((K, C), dtype=np.float64)
    vals[ok] = acc[ok] / wsum[ok]
    return vals, ok


def _trilinear_sample_sparse_gpu(positions, voxel_coords_np, color_np, resolution):
    """GPU port of `_trilinear_sample_sparse`. Returns (vals [K,C] float32, ok [K] bool)."""
    dev = comfy.model_management.get_torch_device()
    R = int(resolution)
    origin = -0.5
    voxel_size = 1.0 / R
    P = torch.from_numpy(np.ascontiguousarray(positions)).to(dev).float()
    VC = torch.from_numpy(np.ascontiguousarray(voxel_coords_np)).to(dev).long()
    col = torch.from_numpy(np.ascontiguousarray(color_np)).to(dev).float()
    K, C = P.shape[0], col.shape[1]
    M = VC.shape[0]
    # Cell-CENTER convention (see NumPy path): -0.5 to bracket the query.
    gc = (P - origin) / voxel_size - 0.5
    base = torch.floor(gc).long()
    frac = gc - base.float()
    key = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
    skey, order = key.sort()
    acc = torch.zeros((K, C), device=dev)
    wsum = torch.zeros((K, 1), device=dev)
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                cx = base[:, 0] + dx
                cy = base[:, 1] + dy
                cz = base[:, 2] + dz
                inb = (cx >= 0) & (cx < R) & (cy >= 0) & (cy < R) & (cz >= 0) & (cz < R)
                qk = (cx * R + cy) * R + cz
                ins = torch.searchsorted(skey, qk).clamp(max=M - 1)
                matched = inb & (skey[ins] == qk)
                idx = order[ins]                              # garbage where !matched
                w = torch.where(matched, wx * wy * wz, torch.zeros_like(wx))[:, None]
                acc += w * col[idx]                           # w=0 cancels garbage rows
                wsum += w
    ok = wsum[:, 0] > 1e-8
    vals = torch.zeros((K, C), device=dev)
    vals[ok] = acc[ok] / wsum[ok].clamp_min(1e-8)
    return vals.cpu().numpy(), ok.cpu().numpy()


# Above this many grid-scan stragglers, the O(N·M) GPU brute force (and its chunk loop)
# is slower than a one-off cKDTree build, so the nearest fallback defers them to cKDTree.
_BRUTE_NEAREST_MAX = 8192


def _nearest_voxel_sample_gpu(positions, voxel_coords_np, color_np, resolution):
    """GPU nearest-occupied-voxel lookup via sorted-key grid scan. Returns (vals [K,C]
    float32, found [K] bool); `found` is False for stragglers left to the caller's cKDTree."""
    dev = comfy.model_management.get_torch_device()
    R = int(resolution)
    P = torch.from_numpy(np.ascontiguousarray(positions)).to(dev).float()
    VC = torch.from_numpy(np.ascontiguousarray(voxel_coords_np)).to(dev).long()
    col = torch.from_numpy(np.ascontiguousarray(color_np)).to(dev).float()
    M, K = VC.shape[0], P.shape[0]
    key = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
    skey, order = key.sort()

    def _search(idx, radius):
        """Nearest occupied voxel within ±radius cells, for query subset P[idx]."""
        Ps = P[idx]
        # Cell-CENTER convention: nearest coord = round((p+0.5)*R-0.5) (matches official).
        rc = ((Ps + 0.5) * R - 0.5).round().long()
        n = idx.shape[0]
        bd = torch.full((n,), 1e30, device=dev)
        bi = torch.zeros(n, dtype=torch.long, device=dev)
        fnd = torch.zeros(n, dtype=torch.bool, device=dev)
        rng = range(-radius, radius + 1)
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    cc = rc + torch.tensor([dx, dy, dz], device=dev)
                    inb = ((cc >= 0) & (cc < R)).all(1)
                    qk = (cc[:, 0] * R + cc[:, 1]) * R + cc[:, 2]
                    ins = torch.searchsorted(skey, qk).clamp(max=M - 1)
                    match = inb & (skey[ins] == qk)
                    dd = (((cc.float() + 0.5) / R - 0.5 - Ps) ** 2).sum(1)
                    upd = match & (dd < bd)
                    bd = torch.where(upd, dd, bd)
                    bi = torch.where(upd, order[ins], bi)
                    fnd |= match
        return bi, fnd

    def _brute_nearest(idx):
        """Exact nearest occupied voxel for the few grid-scan stragglers, chunked GPU
        brute force (avoids a seconds-long cKDTree build over all M voxels)."""
        Ps = P[idx]                                        # [N,3] world
        N = Ps.shape[0]
        vox_pos = (VC.float() + 0.5) / R - 0.5             # [M,3] voxel centres
        best_d = torch.full((N,), 1e30, device=dev)
        best_j = torch.zeros(N, dtype=torch.long, device=dev)
        # Bound the N×chunk matrix to ~64M elements.
        chunk = max(1, (1 << 26) // max(1, N))
        for s in range(0, M, chunk):
            vc = vox_pos[s:s + chunk]                      # [B,3]
            dd = (Ps[:, None, :] - vc[None, :, :]).pow(2).sum(-1)  # [N,B]
            md, mj = dd.min(1)
            upd = md < best_d
            best_d = torch.where(upd, md, best_d)
            best_j = torch.where(upd, mj + s, best_j)
        return best_j

    all_idx = torch.arange(K, device=dev)
    best_i = torch.zeros(K, dtype=torch.long, device=dev)
    found = torch.zeros(K, dtype=torch.bool, device=dev)
    # Pass 1: radius 1 over everything; Pass 2: radius 4 on misses; Pass 3: brute force.
    bi1, fnd1 = _search(all_idx, 1)
    best_i[all_idx] = bi1
    found[all_idx] = fnd1
    miss = torch.nonzero(~found, as_tuple=True)[0]
    if miss.numel() > 0:
        bi2, fnd2 = _search(miss, 4)
        best_i[miss] = bi2
        found[miss] = fnd2
    # Pass 3: stragglers >4 cells from any voxel. A handful → GPU brute force; many
    # (coarse mesh, texels far from the voxel shell) → leave unfound for the caller's
    # cKDTree, since brute force is O(N·M) and its chunk loop blows up at large N.
    miss2 = torch.nonzero(~found, as_tuple=True)[0]
    if 0 < miss2.numel() <= _BRUTE_NEAREST_MAX:
        best_i[miss2] = _brute_nearest(miss2)
        found[miss2] = True
    vals = col[best_i]
    return vals.cpu().numpy(), found.cpu().numpy()


def _sample_voxel_attrs_per_texel(position_map, mask, voxel_coords, voxel_colors, resolution):
    """Sample all voxel attribute channels at every masked texel. Returns (H,W,C)
    float32 in [0,1] (C = feature width: 3 color, 6 PBR). Normalized trilinear over
    occupied voxels (matches official), nearest fallback where all 8 corners empty."""
    H, W, _ = position_map.shape
    color_np = voxel_colors.detach().cpu().numpy().astype(np.float32)
    C = color_np.shape[-1]
    out = np.zeros((H, W, C), dtype=np.float32)
    if not mask.any():
        return out

    origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    voxel_size = 1.0 / float(resolution)
    coords_np = voxel_coords.detach().cpu().numpy()
    # Cell-CENTER convention (+0.5 voxel) — same world mapping as the GPU paths; this
    # cKDTree only serves the rare non-CUDA nearest fallback.
    voxel_pos = (coords_np.astype(np.float32) + 0.5) * voxel_size + origin
    valid_positions = position_map[mask]

    def _nearest(query):
        # GPU grid scan + small-N brute tail. Large straggler counts (coarse mesh) and
        # non-CUDA come back unfound → resolve with one cKDTree (build amortizes over N).
        vals, found = _nearest_voxel_sample_gpu(query, coords_np, color_np, resolution)
        if not found.all():
            tree = cKDTree(voxel_pos)
            _, nearest_idx = tree.query(query[~found], k=1, workers=-1)
            vals[~found] = color_np[nearest_idx]
        return vals

    try:
        vals, ok = _trilinear_sample_sparse_gpu(valid_positions, coords_np, color_np, resolution)
    except Exception as e:
        logging.warning(f"[BakeTextureFromVoxel] GPU trilinear failed ({e}); falling back to CPU")
        vals, ok = _trilinear_sample_sparse(valid_positions, coords_np, color_np, resolution)
    if not ok.all():
        vals[~ok] = _nearest(valid_positions[~ok])  # no occupied neighbour
    out[mask] = np.clip(vals, 0.0, 1.0).astype(np.float32)
    return out


def _msb_int64(x):
    """floor(log2(x)) elementwise for int64 x >= 1 (bit-search, no float)."""
    r = torch.zeros_like(x)
    xx = x.clone()
    for s in (32, 16, 8, 4, 2, 1):
        sh = xx >> s
        m = sh > 0
        r = torch.where(m, r + s, r)
        xx = torch.where(m, sh, xx)
    return r


def _morton_expand21(v):
    """Spread the low 21 bits of v across every 3rd bit (for a 63-bit Morton code)."""
    v = v & 0x1fffff
    v = (v | (v << 32)) & 0x1f00000000ffff
    v = (v | (v << 16)) & 0x1f0000ff0000ff
    v = (v | (v << 8))  & 0x100f00f00f00f00f
    v = (v | (v << 4))  & 0x10c30c30c30c30c3
    v = (v | (v << 2))  & 0x1249249249249249
    return v


def _build_triangle_bvh(tri):
    """Linear BVH (Karras 2012) over triangle AABBs, pure torch, no external deps
    (the cuMesh approach, in torch). Internal nodes 0..T-2; leaves encoded LEAF+i,
    leaf i holds triangle order[i]. Returns dict(LEAF, left, right, nmin, nmax over
    2T entries, order, T)."""
    dev = tri.device
    T = tri.shape[0]
    amin = tri.amin(1)
    amax = tri.amax(1)
    cent = (amin + amax) * 0.5
    lo = cent.amin(0)
    hi = cent.amax(0)
    span = (hi - lo).clamp_min(1e-12)
    q = (((cent - lo) / span).clamp(0, 1) * float((1 << 21) - 1)).long()
    morton = (_morton_expand21(q[:, 0]) << 2 | _morton_expand21(q[:, 1]) << 1 | _morton_expand21(q[:, 2])).long()
    order = torch.argsort(morton)
    msort = morton[order]

    # delta(i,j): common-prefix length of (morton, index) keys of leaves i,j (index
    # breaks ties so duplicate codes still split); -1 if OOB.
    def delta(i, j):
        ok = (j >= 0) & (j < T)
        jj = j.clamp(0, T - 1)
        x = msort[i] ^ msort[jj]
        same = x == 0
        cp = torch.where(same, torch.full_like(x, 63), 62 - _msb_int64(x.clamp_min(1)))
        xi = i ^ jj
        cpi = torch.where(xi == 0, torch.full_like(x, 32), 31 - _msb_int64(xi.clamp_min(1)))
        return torch.where(ok, cp + torch.where(same, cpi, torch.zeros_like(cp)), torch.full_like(x, -1))

    I = torch.arange(T - 1, device=dev)
    dplus = delta(I, I + 1)
    dminus = delta(I, I - 1)
    direction = torch.where(dplus >= dminus, torch.ones_like(I), -torch.ones_like(I))
    dmin = torch.minimum(dplus, dminus)
    # range length: exponential probe then binary search
    lmax = torch.full_like(I, 2)
    while True:
        cond = delta(I, I + lmax * direction) > dmin
        if not bool(cond.any()):
            break
        lmax = torch.where(cond, lmax * 2, lmax)
        if int(lmax.max()) > 2 * T:
            break
    l = torch.zeros_like(I)
    t = lmax.clone()
    while True:
        t = t // 2
        if int(t.max()) == 0:
            break
        cond = delta(I, I + (l + t) * direction) > dmin
        l = torch.where(cond, l + t, l)
    j = I + l * direction
    first = torch.minimum(I, j)
    last = torch.maximum(I, j)
    # split position: binary search on delta within [first, last]
    dnode = delta(first, last)
    s = torch.zeros_like(I)
    div = torch.full_like(I, 2)
    rng = last - first
    while True:
        step = (rng + div - 1) // div
        cond = delta(first, (first + s + step).clamp(max=T - 1)) > dnode
        s = torch.where(cond, s + step, s)
        if int(step.max()) <= 1:
            cond1 = delta(first, (first + s + 1).clamp(max=T - 1)) > dnode
            s = torch.where(cond1, s + 1, s)
            break
        div = div * 2
    gamma = first + s
    LEAF = T
    left = torch.where(gamma == first, LEAF + gamma, gamma)
    right = torch.where(gamma + 1 == last, LEAF + gamma + 1, gamma + 1)

    # node AABBs: leaves seeded, internal unioned bottom-up (~log2(T) passes; cap is a backstop).
    nmin = torch.empty((2 * T, 3), device=dev)
    nmax = torch.empty((2 * T, 3), device=dev)
    nmin[LEAF:] = amin[order]
    nmax[LEAF:] = amax[order]
    setm = torch.zeros(2 * T, dtype=torch.bool, device=dev)
    setm[LEAF:] = True
    for _ in range(128):
        need = ~setm[:T - 1]
        if not bool(need.any()):
            break
        idx = torch.nonzero(need, as_tuple=True)[0]
        ii = idx[setm[left[idx]] & setm[right[idx]]]
        if ii.numel() == 0:
            break
        nmin[ii] = torch.minimum(nmin[left[ii]], nmin[right[ii]])
        nmax[ii] = torch.maximum(nmax[left[ii]], nmax[right[ii]])
        setm[ii] = True
    return dict(LEAF=LEAF, left=left, right=right, nmin=nmin, nmax=nmax, order=order, T=T)


def _closest_points_on_mesh_bvh(Q, tri, bvh, max_stack=64, return_face=False):
    """Exact closest surface point per query via per-query BVH stack traversal
    (nearest-child-first), pure torch. Returns [N,3], or (points [N,3], face_idx [N])
    when return_face=True (face_idx indexes `tri`). `max_stack` bounds the stack
    (= tree height); overflow is counted+warned, not silently wrong."""
    dev = Q.device
    N = Q.shape[0]
    LEAF = bvh['LEAF']
    nmin = bvh['nmin']
    nmax = bvh['nmax']
    left = bvh['left']
    right = bvh['right']
    order = bvh['order']
    stack = torch.full((N, max_stack), -1, dtype=torch.long, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    best = torch.full((N,), 1e30, device=dev)
    bestp = Q.clone()
    bestf = torch.full((N,), -1, dtype=torch.long, device=dev)
    active = torch.arange(N, device=dev)
    overflow = 0

    def aabb_d2(node, q):
        d = (nmin[node] - q).clamp_min(0) + (q - nmax[node]).clamp_min(0)
        return (d * d).sum(-1)

    while active.numel() > 0:
        a = active
        qa = Q[a]
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        within = aabb_d2(node, qa) < best[a]
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            fidx = order[node[lv] - LEAF]                # triangle index of each leaf
            tt = tri[fidx]
            cp, d2 = _point_tri_closest(qa[lv], tt)
            upd = d2 < best[ga]
            gu = ga[upd]
            best[gu] = d2[upd]
            bestp[gu] = cp[upd]
            bestf[gu] = fidx[upd]
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            qi = qa[iv]
            lc = left[node[iv]]
            rc = right[node[iv]]
            dl = aabb_d2(lc, qi)
            dr = aabb_d2(rc, qi)
            near = torch.where(dl <= dr, lc, rc)
            far = torch.where(dl <= dr, rc, lc)
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = far
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            overflow += int((s1 >= max_stack).sum())
            stack[gi, s1.clamp(max=max_stack - 1)] = near
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    if overflow:
        logging.warning(f"[back-project] BVH stack overflow on {overflow} pushes "
                        f"(max_stack={max_stack}); a few texels may be slightly off — "
                        f"raise max_stack if this is large.")
    if return_face:
        return bestp, bestf
    return bestp


def _back_project_positions(position_map, mask, ref_v, ref_f):
    """Snap covered texels onto the reference mesh's true surface (pure-torch BVH, no
    cumesh/scipy/trimesh) so the voxel field is sampled at full detail, not along flat
    triangle chords. Returns a new position_map."""
    valid = np.ascontiguousarray(position_map[mask].astype(np.float32))
    if valid.shape[0] == 0:
        return position_map

    dev = comfy.model_management.get_torch_device()
    rv = ref_v.detach().to(dev).float()
    rf = ref_f.detach().to(dev).long()
    tri = rv[rf]
    Q = torch.from_numpy(valid).to(dev)

    bvh = _build_triangle_bvh(tri)
    bp = _closest_points_on_mesh_bvh(Q, tri, bvh)

    out = position_map.copy()
    out[mask] = bp.detach().cpu().numpy().astype(position_map.dtype)
    return out


def _ray_tri_hit(o, d, tri, tmin, tmax):
    """Möller-Trumbore any-hit per (ray, triangle) pair, double-sided. Returns bool [N]."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b - a, c - a
    p = torch.cross(d, e2, dim=-1)
    det = (e1 * p).sum(-1)
    inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    tvec = o - a
    u = (tvec * p).sum(-1) * inv
    q = torch.cross(tvec, e1, dim=-1)
    v = (d * q).sum(-1) * inv
    t = (e2 * q).sum(-1) * inv
    return (det.abs() > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin) & (t < tmax)


def _any_hit_rays_bvh(orig, dirs, tri, bvh, tmin=0.0, tmax=1e30, max_stack=64):
    """Any-hit ray test over the BVH (slab cull + Möller-Trumbore), pure torch. Returns bool
    [N]: True if the ray hits any triangle in (tmin, tmax). Rays early-out once they hit."""
    dev = orig.device
    N = orig.shape[0]
    LEAF = bvh['LEAF']
    nmin, nmax = bvh['nmin'], bvh['nmax']
    left, right, order = bvh['left'], bvh['right'], bvh['order']
    inv = 1.0 / torch.where(dirs.abs() < 1e-20, torch.full_like(dirs, 1e-20), dirs)
    hit = torch.zeros(N, dtype=torch.bool, device=dev)
    # int32 stack: node indices fit in 31 bits and this [N, max_stack] array dominates memory.
    stack = torch.full((N, max_stack), -1, dtype=torch.int32, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    active = torch.arange(N, device=dev)

    def slab(node, o, i):
        t1 = (nmin[node] - o) * i
        t2 = (nmax[node] - o) * i
        tnear = torch.minimum(t1, t2).amax(-1)
        tfar = torch.maximum(t1, t2).amin(-1)
        return (tfar >= tnear.clamp_min(tmin)) & (tnear <= tmax) & (tfar >= tmin)

    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        within = slab(node, orig[a], inv[a])
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            tt = tri[order[node[lv] - LEAF]]
            h = _ray_tri_hit(orig[ga], dirs[ga], tt, tmin, tmax)
            hit[ga[h]] = True
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = left[node[iv]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=max_stack - 1)] = right[node[iv]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[(sp[a] > 0) & ~hit[a]]                  # drop finished + already-hit rays
    return hit


def _ray_tri_intersect(o, d, tri, tmin, tmax, cull_backface=False):
    """Möller-Trumbore per (ray, triangle) pair. Returns (hit [N], t [N]) where t is the ray
    parameter and hit means the meeting is in (tmin, tmax). With cull_backface, drops faces whose
    outward (winding) normal points along the ray — i.e. only keep surfaces facing the origin."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    e1, e2 = b - a, c - a
    p = torch.cross(d, e2, dim=-1)
    det = (e1 * p).sum(-1)
    inv = 1.0 / torch.where(det.abs() < 1e-20, torch.full_like(det, 1e-20), det)
    tvec = o - a
    u = (tvec * p).sum(-1) * inv
    q = torch.cross(tvec, e1, dim=-1)
    v = (d * q).sum(-1) * inv
    t = (e2 * q).sum(-1) * inv
    hit = (det.abs() > 1e-20) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > tmin) & (t < tmax)
    if cull_backface:
        hit = hit & ((torch.cross(e1, e2, dim=-1) * d).sum(-1) < 0)   # keep only front-facing
    return hit, t


def _closest_hit_rays_bvh(orig, dirs, tri, bvh, tmin=0.0, tmax=1e30, max_stack=64, cull_backface=False):
    """Nearest-hit ray cast over the BVH, pure torch. Returns (t [N], face [N] long, -1 on
    miss; hit [N] bool) — the closest intersection in (tmin, tmax), pruning nodes past best_t."""
    dev = orig.device
    N = orig.shape[0]
    LEAF = bvh['LEAF']
    nmin, nmax = bvh['nmin'], bvh['nmax']
    left, right, order = bvh['left'], bvh['right'], bvh['order']
    inv = 1.0 / torch.where(dirs.abs() < 1e-20, torch.full_like(dirs, 1e-20), dirs)
    best_t = torch.full((N,), float(tmax), device=dev)
    best_f = torch.full((N,), -1, dtype=torch.long, device=dev)
    stack = torch.full((N, max_stack), -1, dtype=torch.int32, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev)
    stack[:, 0] = 0
    active = torch.arange(N, device=dev)

    while active.numel() > 0:
        a = active
        node = stack[a, sp[a] - 1]
        sp[a] = sp[a] - 1
        t1 = (nmin[node] - orig[a]) * inv[a]
        t2 = (nmax[node] - orig[a]) * inv[a]
        tnear = torch.minimum(t1, t2).amax(-1)
        tfar = torch.maximum(t1, t2).amin(-1)
        within = (tfar >= tnear.clamp_min(tmin)) & (tfar >= tmin) & (tnear < best_t[a])   # prune past best
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]
            fidx = order[node[lv] - LEAF]
            h, t = _ray_tri_intersect(orig[ga], dirs[ga], tri[fidx], tmin, tmax, cull_backface)
            upd = h & (t < best_t[ga])
            gu = ga[upd]
            best_t[gu] = t[upd]
            best_f[gu] = fidx[upd]
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = left[node[iv]].to(torch.int32)
            sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]
            stack[gi, s1.clamp(max=max_stack - 1)] = right[node[iv]].to(torch.int32)
            sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    return best_t, best_f, best_f >= 0


def _onb(n):
    """Branchless orthonormal basis (t, b) around unit normals n [N,3]."""
    up = torch.where(n[..., 2:3].abs() < 0.999,
                     torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n),
                     torch.tensor([1.0, 0.0, 0.0], device=n.device).expand_as(n))
    t = torch.nn.functional.normalize(torch.cross(up, n, dim=-1), dim=-1, eps=1e-6)
    return t, torch.cross(n, t, dim=-1)


def _bake_ambient_occlusion(high_v, high_f, low_v_np, low_f_np, low_uv_np, low_n, resolution,
                            num_samples=64, max_distance=0.5, strength=1.0, bias=0.01,
                            ray_chunk=None, pbar=None):
    """Bake high-poly ambient occlusion into the low-poly's UV layout: per texel, cosine-weight
    a hemisphere of rays around the normal and cast them at the high-poly. AO = 1 - hit-fraction
    (cosine weighting makes the hit-fraction the estimator). Returns ao_img [H,W,3] in [0,1].

    ray_chunk caps rays cast at once (the per-chunk BVH stack is its dominant transient VRAM);
    None auto-sizes it to a slice of free VRAM — big chunks (fast) on large GPUs, small (safe)
    on small ones."""
    dev = comfy.model_management.get_torch_device()
    H = W = int(resolution)
    S = int(num_samples)
    if ray_chunk is None:
        # ~376 B/ray (int32 stack max_stack*4 + a few [N,3] ray buffers); spend a quarter of free
        # VRAM. Speed saturates around 4M rays/chunk, so cap there (≈2 GB peak) rather than grow
        # memory for no gain; floor keeps tiny GPUs from thrashing into too many chunks.
        try:
            free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else (2 << 30)
        except Exception:
            free = 2 << 30
        ray_chunk = int(min(1 << 22, max(1 << 20, (free * 0.25) / (num_samples * 4 + 200))))
    face_idx, bary_uv, mask = _rasterize_uv_barycentric(low_f_np, low_uv_np, resolution)
    if not mask.any():
        return np.ones((H, W, 3), dtype=np.float32)
    lf = torch.from_numpy(np.ascontiguousarray(low_f_np).astype(np.int64)).to(dev)
    lv = torch.from_numpy(np.ascontiguousarray(low_v_np, dtype=np.float32)).to(dev)
    low_n = low_n.to(dev).float()
    m = mask
    vtri = lf[face_idx[m]]                                                 # [K,3] vertex ids
    bsel = bary_uv[m]                                                      # [K,3]
    P = (bsel[:, :, None] * lv[vtri]).sum(1)                               # [K,3]
    Nl = torch.nn.functional.normalize((bsel[:, :, None] * low_n[vtri]).sum(1), dim=-1, eps=1e-6)

    hv = high_v.to(dev).float()
    hf = high_f.to(dev).long()
    tri = hv[hf]
    bvh = _build_triangle_bvh(tri)
    diag = float((hv.amax(0) - hv.amin(0)).norm().clamp_min(1e-6))
    biasw = max(1e-5, float(bias) * diag)
    tmax = float(max_distance) * diag

    # Back-project onto the high surface, then lift along the normal: the low-poly chord can sit
    # below the high surface, and casting from below floods false self-occlusion (dark blotches).
    bp = _closest_points_on_mesh_bvh(P, tri, bvh)
    origins = bp + Nl * biasw

    K = P.shape[0]
    T, B = _onb(Nl)
    occ = torch.zeros(K, device=dev)
    tex_per_chunk = max(1, int(ray_chunk) // max(1, S))
    for s in range(0, K, tex_per_chunk):
        e = min(s + tex_per_chunk, K)
        kk = e - s
        o, n, t, b = origins[s:e], Nl[s:e], T[s:e], B[s:e]
        r1 = torch.rand(kk, S, device=dev)
        r2 = torch.rand(kk, S, device=dev)
        sr = r1.sqrt()
        lz = r1.mul_(-1.0).add_(1.0).clamp_min_(0.0).sqrt_()             # sqrt(1-r1) (r1 dead after sr)
        ang = r2.mul_(2.0 * math.pi)                                      # in place (r2 dead)
        lx = ang.cos().mul_(sr)
        ly = ang.sin().mul_(sr)
        d = t[:, None, :] * lx[..., None]                                 # cosine-weighted hemisphere,
        d.addcmul_(b[:, None, :], ly[..., None])                          # fused d += b*ly
        d.addcmul_(n[:, None, :], lz[..., None])                          # fused d += n*lz  (no extra temps)
        d = torch.nn.functional.normalize(d.reshape(-1, 3), dim=-1, eps=1e-6)
        oo = o[:, None, :].expand(-1, S, -1).reshape(-1, 3)
        hit = _any_hit_rays_bvh(oo, d, tri, bvh, tmin=biasw, tmax=tmax)
        occ[s:e] = hit.reshape(kk, S).sum(1, dtype=torch.float32).div_(float(S))   # mean without a float copy
        if pbar is not None:
            pbar.update(1)

    ao = occ.mul_(-float(strength)).add_(1.0).clamp_(0.0, 1.0)   # 1 - occ*strength, in place (occ is dead)
    out = torch.ones((H, W), device=dev)
    out[m] = ao
    out3 = np.repeat(out.cpu().numpy()[..., None], 3, axis=2)
    return _jfa_fill_gpu(np.ascontiguousarray(out3, dtype=np.float32), mask.cpu().numpy())


def _compute_vertex_tangents(verts, faces, uvs, normals):
    """Per-vertex tangents (Lengyel) orthonormalized against `normals`. Returns [N,4]:
    unit tangent xyz + handedness w (the bitangent is w * cross(N, T)). Pure torch."""
    N = verts.shape[0]
    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()
    e1, e2 = verts[i1] - verts[i0], verts[i2] - verts[i0]
    d1, d2 = uvs[i1] - uvs[i0], uvs[i2] - uvs[i0]
    denom = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = 1.0 / torch.where(denom.abs() < 1e-20, torch.full_like(denom, 1e-20), denom)
    tan = (d2[:, 1:2] * e1 - d1[:, 1:2] * e2) * r[:, None]          # [F,3]
    bit = (d1[:, 0:1] * e2 - d2[:, 0:1] * e1) * r[:, None]
    tacc = torch.zeros((N, 3), device=verts.device, dtype=verts.dtype)
    bacc = torch.zeros((N, 3), device=verts.device, dtype=verts.dtype)
    for idx in (i0, i1, i2):
        tacc.scatter_add_(0, idx[:, None].expand(-1, 3), tan)
        bacc.scatter_add_(0, idx[:, None].expand(-1, 3), bit)
    n = torch.nn.functional.normalize(normals, dim=-1, eps=1e-6)
    # Gram-Schmidt: drop the normal component, then renormalize.
    t = torch.nn.functional.normalize(tacc - n * (n * tacc).sum(-1, keepdim=True), dim=-1, eps=1e-6)
    w = torch.sign((torch.cross(n, t, dim=-1) * bacc).sum(-1))
    w = torch.where(w == 0, torch.ones_like(w), w)                 # degenerate → right-handed
    return torch.cat([t, w[:, None]], dim=-1)


def _vertex_tangents_for_item(lv, lf, uv, low_n_attr_i, dev):
    """Per-item shading normals + tangents. Shared by the bake (BakeNormalMapFromMesh) and the
    export attach (ApplyTextureToMesh) so their basis can't diverge. `low_n_attr_i` is the
    mesh's per-item normals or None (then computed). Returns (low_n [N,3], tangents [N,4])."""
    low_n = low_n_attr_i.to(dev).float() if low_n_attr_i is not None else _compute_vertex_normals(lv, lf)
    tangents = _compute_vertex_tangents(lv, lf, uv.to(dev).float(), low_n)
    return low_n, tangents


def _barycentric(p, tri):
    """Barycentric coords [N,3] of points p [N,3] wrt triangles tri [N,3,3] (per-pair)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1)
    d01 = (v0 * v1).sum(-1)
    d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1)
    d21 = (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    denom = torch.where(denom.abs() < 1e-20, torch.full_like(denom, 1e-20), denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return torch.stack([1.0 - v - w, v, w], dim=-1)


def _bake_normal_map(high_v, high_f, high_n, low_v_np, low_f_np, low_uv_np, low_n, tangents,
                     resolution, cage_distance=0.05, ignore_backfaces=True):
    """Tangent-space normal map (glTF/OpenGL +Y) of the high-poly baked into the low-poly's UV
    layout. Per texel a cage ray (along the normal, over cage_distance * bbox-diagonal) finds the
    matching high-poly surface, whose normal is projected into the texel's TBN frame.
    ignore_backfaces skips surfaces facing away (crevices/enclosures); misses fall back to
    closest-point. Returns [H,W,3] in [0,1]."""
    dev = comfy.model_management.get_torch_device()
    H = W = int(resolution)
    flat = np.array([0.5, 0.5, 1.0], dtype=np.float32)

    # One rasterization, then interpolate position/normal/tangent/handedness by indexing it.
    face_idx, bary_uv, mask = _rasterize_uv_barycentric(low_f_np, low_uv_np, resolution)
    if not mask.any():
        return np.tile(flat, (H, W, 1))
    lf = torch.from_numpy(np.ascontiguousarray(low_f_np).astype(np.int64)).to(dev)
    lv = torch.from_numpy(np.ascontiguousarray(low_v_np, dtype=np.float32)).to(dev)
    low_n = low_n.to(dev).float()
    tangents = tangents.to(dev).float()
    m = mask
    fsel = face_idx[m]                                                  # [K] source face per texel
    bsel = bary_uv[m]                                                   # [K,3]
    vtri = lf[fsel]                                                     # [K,3] vertex ids

    def _interp(attr):                                                  # attr [N,C] -> [K,C]
        return (bsel[:, :, None] * attr[vtri]).sum(1)

    P = _interp(lv)                                                     # [K,3] world pos
    Nl = torch.nn.functional.normalize(_interp(low_n), dim=-1, eps=1e-6)
    Tl = _interp(tangents[:, :3])
    Wl = _interp(tangents[:, 3:4])[:, 0]

    hv = high_v.to(dev).float()
    hf = high_f.to(dev).long()
    tri = hv[hf]
    bvh = _build_triangle_bvh(tri)

    # Cage ray-cast: from cage outward, march back along -normal and take the nearest (outermost)
    # hit. Closest-point is the fallback where the ray misses.
    diag = float((hv.amax(0) - hv.amin(0)).norm().clamp_min(1e-6))
    cage = max(1e-6, float(cage_distance) * diag)
    origin = P + Nl * cage
    t_hit, f_hit, ray_hit = _closest_hit_rays_bvh(origin, -Nl, tri, bvh, tmin=0.0, tmax=2.0 * cage,
                                                  cull_backface=bool(ignore_backfaces))
    bface = f_hit.clamp_min(0)
    hitpoint = origin - t_hit[:, None] * Nl
    # Closest-point fallback only for texels the ray missed (usually few) — running it over every
    # texel wastes a full BVH traversal on the ones already resolved by the ray.
    miss = ~ray_hit
    if bool(miss.any()):
        bp_m, bf_m = _closest_points_on_mesh_bvh(P[miss], tri, bvh, return_face=True)
        bface = bface.clone()
        hitpoint = hitpoint.clone()
        bface[miss] = bf_m.clamp_min(0)
        hitpoint[miss] = bp_m

    htri = tri[bface]                                                  # [K,3,3]
    bary = _barycentric(hitpoint, htri)
    hn_tri = high_n.to(dev).float()[hf[bface]]                         # [K,3,3] vertex normals
    Nh = torch.nn.functional.normalize((bary[:, :, None] * hn_tri).sum(1), dim=-1, eps=1e-6)

    # Per-texel TBN (Gram-Schmidt tangent against the interpolated normal).
    T = torch.nn.functional.normalize(Tl - Nl * (Nl * Tl).sum(-1, keepdim=True), dim=-1, eps=1e-6)
    Bn = Wl[:, None] * torch.cross(Nl, T, dim=-1)
    nz = (Nh * Nl).sum(-1)                                             # reused as z-channel and the back-face test
    ts = torch.stack([(Nh * T).sum(-1), (Nh * Bn).sum(-1), nz], dim=-1)
    ts = torch.nn.functional.normalize(ts, dim=-1, eps=1e-6)
    # Safety net: if the matched high normal faces away from the texel (a back surface the fallback
    # grabbed in a deep crevice), use the flat base normal rather than a wrong one.
    ts[nz < 0.0] = torch.tensor([0.0, 0.0, 1.0], device=dev)
    enc = ts.mul_(0.5).add_(0.5).clamp_(0.0, 1.0)                      # encode in place (ts is dead)

    out = torch.from_numpy(np.tile(flat, (H, W, 1))).to(dev)
    out[m] = enc
    # Dilate into the UV gutter so bilinear/mip sampling at chart edges doesn't bleed flat blue.
    return _jfa_fill_gpu(out.cpu().numpy(), mask.cpu().numpy())


def _jfa_fill_gpu(img01, mask):
    """Fill uncovered texels with nearest covered value via GPU Jump Flooding
    (O(log n) passes; replaces cv2.inpaint). img01 [H,W,C] float, mask [H,W] bool."""
    if not mask.any():
        return img01
    dev = comfy.model_management.get_torch_device()
    it = torch.from_numpy(np.ascontiguousarray(img01)).to(dev).float()
    mm = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
    H, W = mm.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    by = torch.where(mm, yy, torch.full_like(yy, -1))
    bx = torch.where(mm, xx, torch.full_like(xx, -1))
    INF = torch.full_like(yy, 1 << 30)
    step = 1 << ((max(H, W) - 1).bit_length() - 1)
    while step >= 1:
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                if dy == 0 and dx == 0:
                    continue
                ny = (yy + dy).clamp(0, H - 1)
                nx = (xx + dx).clamp(0, W - 1)
                cby = by[ny, nx]
                cbx = bx[ny, nx]
                valid = cby >= 0
                dc = torch.where(valid, (yy - cby) ** 2 + (xx - cbx) ** 2, INF)
                db = torch.where(by >= 0, (yy - by) ** 2 + (xx - bx) ** 2, INF)
                take = valid & (dc < db)
                by = torch.where(take, cby, by)
                bx = torch.where(take, cbx, bx)
        step //= 2
    filled = it[by.clamp(0).long(), bx.clamp(0).long()]
    return filled.cpu().numpy()


def _seam_fill(img01, mask, inpaint_radius):
    """Fill UV-gutter texels (so seams don't pull in black) via JFA. `inpaint_radius<=0`
    disables; the radius value itself is ignored (JFA fills all uncovered by nearest)."""
    if inpaint_radius <= 0:
        return img01
    return _jfa_fill_gpu(img01, mask)


def _normalize_uvs_to_unit(uv_np, normalize=True, log_prefix=None):
    """Uniformly fit a UV bbox into [0,1] when it spills outside (preserves aspect;
    no-op if already in [0,1]; not a UDIM de-tiler). Shared deterministic helper —
    bake and ApplyTextureToMesh both call it so UVs agree (keep both paths in sync).
    Returns float32 [N,2]."""
    uv_np = uv_np.astype(np.float32)
    uv_min = uv_np.min(axis=0)
    uv_max = uv_np.max(axis=0)
    out_of_unit = (uv_min.min() < -1e-4) or (uv_max.max() > 1.0001)
    if not (normalize and out_of_unit):
        return uv_np
    extent = float((uv_max - uv_min).max())
    span = max(float(uv_max[0] - uv_min[0]), float(uv_max[1] - uv_min[1]))
    if span > 1.5 and log_prefix:
        logging.warning(
            f"{log_prefix} UV span {span:.2f} looks like a tiled/UDIM layout; "
            f"uniform-fitting it into [0,1] will overlap tiles. Re-unwrap upstream instead.")
    if extent > 0:
        uv_np = ((uv_np - uv_min) / extent).astype(np.float32)
        if log_prefix:
            logging.info(f"{log_prefix} normalized UVs into [0,1] (uniform scale 1/{extent:.4f})")
    return uv_np


def bake_texture_from_voxel_fn(vertices, faces, voxel_coords, voxel_colors,
                               resolution, texture_size, uvs, inpaint_radius=3,
                               normalize_uvs=True, reference=None, pbar=None):
    """Bake a baseColor (+ optional metallicRoughness) texture: rasterize in UV space,
    sample each texel from the sparse voxel volume. `uvs` (N,2) is the existing layout,
    1:1 with `vertices` (never unwraps). Returns (v, f, uvs, texture, mr). Ticks `pbar`
    once per stage; size it 5 per bake."""
    # _tick fires once per stage boundary, including no-op stages, so the 5-tick pbar stays aligned.
    _tq = tqdm(total=5, desc="Bake texture", leave=False)

    def _tick(name):
        _tq.set_postfix_str(name)
        _tq.update(1)
        if pbar is not None:
            pbar.update(1)

    v_np = vertices.detach().cpu().numpy().astype(np.float32)
    f_np = faces.detach().cpu().numpy().astype(np.uint32)
    fcount = int(f_np.shape[0])

    uv_np = uvs.detach().cpu().numpy().astype(np.float32)
    if uv_np.shape[0] != v_np.shape[0]:
        raise ValueError(
            f"BakeTextureFromVoxel: UVs ({uv_np.shape[0]}) must be 1:1 "
            f"with vertices ({v_np.shape[0]})."
        )
    uv_min = uv_np.min(axis=0)
    uv_max = uv_np.max(axis=0)
    oob = int(((uv_np < 0.0) | (uv_np > 1.0)).any(axis=1).sum())
    logging.info(f"[BakeTextureFromVoxel] using existing UVs: {v_np.shape[0]} verts, "
                 f"{fcount} faces")
    logging.info(f"[BakeTextureFromVoxel]   UV range: u[{uv_min[0]:.3f},{uv_max[0]:.3f}] "
                 f"v[{uv_min[1]:.3f},{uv_max[1]:.3f}]  out-of-[0,1] verts: {oob}/{uv_np.shape[0]}")
    uv_np = _normalize_uvs_to_unit(uv_np, normalize_uvs, log_prefix="[BakeTextureFromVoxel]  ")
    new_verts, new_faces, new_uvs = v_np, f_np, uv_np

    _tick("uvs")

    position_map, mask = _bake_position_map(new_verts, new_faces, new_uvs, texture_size)
    _tick("rasterize")

    if reference is not None:
        # Back-project onto the dense surface before sampling (smooth bake on coarse
        # meshes, not along flat triangle chords).
        position_map = _back_project_positions(position_map, mask, reference[0], reference[1])
    _tick("back-project")

    attrs = _sample_voxel_attrs_per_texel(
        position_map, mask, voxel_coords, voxel_colors, resolution,
    )
    _tick("sample")

    # PBR layout (upstream pbr_attr_layout): 0:3 base_color, 3 metallic, 4 roughness, 5 alpha.
    C = attrs.shape[-1]
    base_color = attrs[..., 0:3]
    has_pbr = C >= 5
    metallic = attrs[..., 3:4] if C >= 4 else None
    roughness = attrs[..., 4:5] if C >= 5 else None
    # alpha (idx 5) ignored — meshes kept opaque (upstream OPAQUE alpha_mode).

    base_color = _seam_fill(np.ascontiguousarray(base_color), mask, inpaint_radius)
    mr_image = None
    if has_pbr:
        # glTF metallicRoughness: R unused, G=roughness, B=metallic.
        mr = np.concatenate([np.zeros_like(roughness), roughness, metallic], axis=-1)
        mr_image = _seam_fill(np.ascontiguousarray(mr), mask, inpaint_radius)

    device = vertices.device
    out_v = torch.from_numpy(new_verts).to(device=device, dtype=torch.float32)
    out_f = torch.from_numpy(new_faces.astype(np.int32)).to(device=device, dtype=torch.int32)
    out_uvs = torch.from_numpy(new_uvs).to(device=device, dtype=torch.float32)
    out_tex = torch.from_numpy(np.ascontiguousarray(base_color)).to(device=device, dtype=torch.float32)
    out_mr = (torch.from_numpy(np.ascontiguousarray(mr_image)).to(device=device, dtype=torch.float32)
              if mr_image is not None else None)
    _tick("finalize")
    _tq.close()
    return out_v, out_f, out_uvs, out_tex, out_mr


def _mr_channel(packed_mr, ch, ref):
    """Pull one channel (G=roughness idx 1, B=metallic idx 2) from a packed glTF MR map
    as 3-channel grayscale [H,W,3] in [0,1]. Black sized like `ref` if no MR map."""
    if packed_mr is None:
        return torch.zeros_like(ref.float().cpu())
    m = packed_mr.float().clamp(0.0, 1.0).cpu()
    return m[..., ch:ch + 1].expand(-1, -1, 3).contiguous()


class BakeTextureFromVoxel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BakeTextureFromVoxel",
            display_name="Bake Texture From Voxel",
            category="latent/3d",
            description=(
                "Bakes PBR textures onto the mesh's existing UV layout (trilinear-sample the "
                "sparse voxel volume). Does NOT unwrap — connect a UV unwrap node upstream. Outputs "
                "base_color + metallic/roughness grayscale IMAGEs (black if no PBR); feed them to "
                "ApplyTextureToMesh (SAME mesh) for SaveGLB."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Voxel.Input("voxel_colors"),
                IO.Int.Input("texture_size", default=1024, min=64, max=8192,
                             tooltip="Square texture resolution."),
                IO.Mesh.Input("reference_mesh", optional=True,
                              tooltip=(
                                  "Optional dense pre-decimation mesh; back-projects each texel onto its "
                                  "true surface before sampling, removing faceted baking on coarse meshes.")),
            ],
            outputs=[
                IO.Image.Output(display_name="base_color"),
                IO.Image.Output(display_name="metallic"),
                IO.Image.Output(display_name="roughness"),
            ],
        )

    @classmethod
    def execute(cls, mesh, voxel_colors, texture_size, reference_mesh=None):
        # Matches official to_glb; effectively on/off since the gutter fill ignores the value.
        inpaint_radius = 3
        voxels = voxel_colors
        coords = voxels.data
        colors = voxels.voxel_colors
        resolution = voxels.resolution
        mesh_uvs = getattr(mesh, "uvs", None)
        if mesh_uvs is None:
            raise ValueError(
                "BakeTextureFromVoxel: input mesh has no UVs. This node bakes onto the "
                "mesh's existing UV layout and never unwraps — connect a UV unwrap node "
                "(e.g. Trellis2OfficialUnwrap or TorchXatlasUVWrap) before it.")

        if coords.shape[-1] == 4:
            # Sparse coords have a batch column; bake per-item.
            batch_idx = coords[:, 0].long()
            voxel_xyz = coords[:, 1:]
            mesh_batch_size = int(mesh.vertices.shape[0])
            out_tex, out_mr = [], []
            # 5 ticks per item; skipped items tick all 5 to stay aligned.
            pbar = comfy.utils.ProgressBar(mesh_batch_size * 5)
            for i in range(mesh_batch_size):
                sel = batch_idx == i
                item_coords = voxel_xyz[sel]
                item_colors = colors[sel]
                v_i, f_i, _ = get_mesh_batch_item(mesh, i)
                if item_coords.shape[0] == 0 or f_i.numel() == 0:
                    logging.warning(f"BakeTextureFromVoxel: skipping batch {i} (empty voxel/mesh)")
                    pbar.update(5)
                    continue
                ev_i = mesh_uvs[i, :v_i.shape[0]]
                ref_i = None
                if reference_mesh is not None:
                    rv_i, rf_i, _ = get_mesh_batch_item(reference_mesh, i)
                    ref_i = (rv_i, rf_i)
                _bv, _bf, _bu, bt, bmr = bake_texture_from_voxel_fn(
                    v_i, f_i, item_coords, item_colors,
                    resolution=resolution, texture_size=texture_size,
                    uvs=ev_i, inpaint_radius=inpaint_radius,
                    reference=ref_i, pbar=pbar,
                )
                out_tex.append(bt)
                out_mr.append(bmr)
            if not out_tex:
                # All items skipped — emit one black map so IMAGE outputs stay valid.
                black = torch.zeros((1, texture_size, texture_size, 3))
                return IO.NodeOutput(black, black, black)
            # Stack [B,H,W,3]; split packed MR (G=roughness, B=metallic) into grayscale maps.
            base_img = torch.stack([t.float().clamp(0.0, 1.0).cpu() for t in out_tex], dim=0)
            metallic_img = torch.stack([_mr_channel(m, 2, out_tex[0]) for m in out_mr], dim=0)
            roughness_img = torch.stack([_mr_channel(m, 1, out_tex[0]) for m in out_mr], dim=0)
            return IO.NodeOutput(base_img, metallic_img, roughness_img)

        # Single-item path.
        v0 = mesh.vertices.squeeze(0)
        f0 = mesh.faces.squeeze(0)
        ev0 = mesh_uvs.squeeze(0)
        ref0 = None
        if reference_mesh is not None:
            ref0 = (reference_mesh.vertices.squeeze(0), reference_mesh.faces.squeeze(0))
        pbar = comfy.utils.ProgressBar(5)  # 5 stage ticks
        _bv, _bf, _bu, bt, bmr = bake_texture_from_voxel_fn(
            v0, f0, coords, colors,
            resolution=resolution, texture_size=texture_size,
            uvs=ev0, inpaint_radius=inpaint_radius,
            reference=ref0, pbar=pbar,
        )
        base_img = bt.float().clamp(0.0, 1.0).cpu().unsqueeze(0)
        metallic_img = _mr_channel(bmr, 2, bt).unsqueeze(0)
        roughness_img = _mr_channel(bmr, 1, bt).unsqueeze(0)
        return IO.NodeOutput(base_img, metallic_img, roughness_img)


class MeshTextureToImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshTextureToImage",
            display_name="Mesh Texture to Image",
            category="latent/3d",
            description=(
                "Extracts a mesh's baked textures as individual IMAGEs: base_color, metallic, "
                "roughness, occlusion and normal_map. Channels with nothing baked come back "
                "neutral (occlusion white, normal flat)."
            ),
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[
                IO.Image.Output(display_name="base_color"),
                IO.Image.Output(display_name="metallic"),
                IO.Image.Output(display_name="roughness"),
                IO.Image.Output(display_name="occlusion"),
                IO.Image.Output(display_name="normal_map"),
            ],
        )

    @classmethod
    def execute(cls, mesh):
        def _as_image(tex):
            # Mesh textures are (B,H,W,3) float [0,1] — already IMAGE layout.
            if tex is None:
                return None
            t = tex.float().clamp(0.0, 1.0).cpu()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t

        base = _as_image(getattr(mesh, "texture", None))
        mr = _as_image(getattr(mesh, "metallic_roughness", None))
        normal_map = _as_image(getattr(mesh, "normal_map", None))

        if base is None:
            raise ValueError(
                "MeshTextureToImage: mesh has no baseColor texture. Run "
                "BakeTextureFromVoxel first (PaintMesh only sets vertex colors, not a texture)."
            )
        if mr is None:
            mr = torch.zeros_like(base)
        if normal_map is None:
            normal_map = torch.ones_like(base) * torch.tensor([0.5, 0.5, 1.0])   # neutral flat normal
        # Unpack the ORM map (R=occlusion, G=roughness, B=metallic) to 3-channel grayscale.
        metallic = mr[..., 2:3].expand(-1, -1, -1, 3).contiguous()
        roughness = mr[..., 1:2].expand(-1, -1, -1, 3).contiguous()
        # R is real occlusion only if AO was baked; else it's the unused zero channel, which as
        # "occlusion" would read fully-dark — so report white unless occlusion_in_mr is set.
        if getattr(mesh, "occlusion_in_mr", False):
            occlusion = mr[..., 0:1].expand(-1, -1, -1, 3).contiguous()
        else:
            occlusion = torch.ones_like(base)
        return IO.NodeOutput(base, metallic, roughness, occlusion, normal_map)


class ApplyTextureToMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ApplyTextureToMesh",
            display_name="Apply Texture to Mesh",
            category="latent/3d",
            description=(
                "Attaches baked texture IMAGEs to a mesh's UV layout for SaveGLB. Feed the SAME mesh you baked"
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Image.Input("base_color"),
                IO.Image.Input("metallic", optional=True),
                IO.Image.Input("roughness", optional=True),
                IO.Image.Input("occlusion", optional=True),
                IO.Image.Input("normal_map", optional=True),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, base_color, metallic=None, roughness=None, occlusion=None, normal_map=None):
        mesh_uvs = getattr(mesh, "uvs", None)
        if mesh_uvs is None:
            raise ValueError(
                "ApplyTextureToMesh: mesh has no UVs. Connect the same UV-unwrapped mesh "
                "you fed to BakeTextureFromVoxel.")

        # Re-derive the exact UVs the bake used (shared _normalize_uvs_to_unit), per item.
        if mesh_uvs.ndim == 3:
            new_uvs = mesh_uvs.clone()
            for i in range(mesh_uvs.shape[0]):
                v_i, _f_i, _ = get_mesh_batch_item(mesh, i)
                n = v_i.shape[0]
                norm = _normalize_uvs_to_unit(mesh_uvs[i, :n].detach().cpu().numpy())
                new_uvs[i, :n] = torch.from_numpy(norm).to(new_uvs)
        else:
            norm = _normalize_uvs_to_unit(mesh_uvs.detach().cpu().numpy())
            new_uvs = torch.from_numpy(norm).to(mesh_uvs)

        out_mesh = copy.copy(mesh)
        out_mesh.uvs = new_uvs
        out_mesh.texture = base_color.float().clamp(0.0, 1.0).cpu()
        if normal_map is not None:
            # Recompute tangents (shared helper, same normalized UVs → same basis as the bake)
            # and export the smooth normals the TBN was built on — without a NORMAL attribute the
            # viewer shades flat and the tangent-space detail fights the faceting.
            dev = comfy.model_management.get_torch_device()
            low_n_attr = getattr(mesh, "normals", None)
            B = int(mesh.vertices.shape[0])
            Nmax = int(mesh.vertices.shape[1]) if mesh.vertices.ndim == 3 else int(mesh.vertices.shape[0])
            tangents_padded = torch.zeros((B, Nmax, 4), dtype=torch.float32)
            normals_padded = torch.zeros((B, Nmax, 3), dtype=torch.float32)
            for i in range(B):
                v_i, f_i, _ = get_mesh_batch_item(mesh, i)
                n = int(v_i.shape[0])
                if f_i.numel() == 0:
                    continue
                lv, lf = v_i.to(dev).float(), f_i.to(dev).long()
                uv_i = new_uvs[i, :n] if new_uvs.ndim == 3 else new_uvs[:n]
                n_attr_i = low_n_attr[i, :n] if low_n_attr is not None else None
                low_n, tangents = _vertex_tangents_for_item(lv, lf, uv_i, n_attr_i, dev)
                tangents_padded[i, :n] = tangents.cpu()
                normals_padded[i, :n] = low_n.cpu()
            out_mesh.normal_map = normal_map.float().clamp(0.0, 1.0).cpu()
            out_mesh.tangents = tangents_padded
            out_mesh.normals = normals_padded
        if metallic is not None or roughness is not None or occlusion is not None:
            # Pack glTF ORM (R=occlusion, G=roughness, B=metallic); missing → 1/1/0. Maps may
            # arrive at different resolutions, so resize each channel to a common H×W first.
            provided = [x for x in (metallic, roughness, occlusion) if x is not None]
            B = int(provided[0].shape[0])
            H = max(int(x.shape[1]) for x in provided)
            W = max(int(x.shape[2]) for x in provided)

            def _chan(img, default):
                if img is None:
                    return torch.full((B, H, W, 1), float(default))
                t = img.float().clamp(0.0, 1.0).cpu()[..., 0:1]
                if int(t.shape[1]) != H or int(t.shape[2]) != W:
                    t = torch.nn.functional.interpolate(t.permute(0, 3, 1, 2), size=(H, W),
                                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
                return t

            out_mesh.metallic_roughness = torch.cat(
                [_chan(occlusion, 1.0), _chan(roughness, 1.0), _chan(metallic, 0.0)], dim=-1)
            if occlusion is not None:
                # Tells SaveGLB to also point occlusionTexture at the MR image (R = AO).
                out_mesh.occlusion_in_mr = True
        return IO.NodeOutput(out_mesh)


class BakeNormalMapFromMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BakeNormalMapFromMesh",
            display_name="Bake Normal Map from Mesh",
            category="latent/3d",
            description=(
                "Bakes a tangent-space normal map (glTF/OpenGL +Y) from a high-poly mesh into a "
                "low-poly's UV layout, capturing detail lost to decimation. Feed the UV-unwrapped "
                "low_poly and the same-frame high_poly it was decimated from. Outputs an IMAGE for "
                "ApplyTextureToMesh's normal_map input."
            ),
            inputs=[
                IO.Mesh.Input("low_poly"),
                IO.Mesh.Input("high_poly"),
                IO.Int.Input("resolution", default=1024, min=64, max=8192, step=64,
                             display_name="resolution"),
                IO.Float.Input("cage_distance", default=0.05, min=0.001, max=0.5, step=0.001,
                               tooltip="Surface search band, as a fraction of the bbox diagonal. "
                                       "Raise for wrong/missing patches under heavy decimation; "
                                       "lower if it grabs across gaps."),
                IO.Boolean.Input("ignore_backfaces", default=True,
                                 tooltip="Skip high-poly surfaces facing away from the texel, so "
                                         "crevices/enclosed spaces don't grab the opposite wall. "
                                         "Disable only if the high-poly winding is inconsistent."),
            ],
            outputs=[IO.Image.Output(display_name="normal_map")],
        )

    @classmethod
    def execute(cls, low_poly, high_poly, resolution, cage_distance=0.05, ignore_backfaces=True):
        low_uvs = getattr(low_poly, "uvs", None)
        if low_uvs is None:
            raise ValueError(
                "BakeNormalMapFromMesh: low_poly has no UVs. Connect the UV-unwrapped "
                "low-poly (the same one you fed to BakeTextureFromVoxel); this node bakes "
                "onto existing UVs and never unwraps.")
        dev = comfy.model_management.get_torch_device()

        low_n_attr = getattr(low_poly, "normals", None)
        high_n_attr = getattr(high_poly, "normals", None)
        B = int(low_poly.vertices.shape[0])
        h_batch = int(high_poly.vertices.shape[0])

        imgs = []
        for i in range(B):
            v_i, f_i, _ = get_mesh_batch_item(low_poly, i)
            n = int(v_i.shape[0])
            if f_i.numel() == 0:
                logging.warning(f"BakeNormalMapFromMesh: skipping batch {i} (empty mesh)")
                imgs.append(torch.full((int(resolution), int(resolution), 3), 0.5))
                continue

            uv_i = low_uvs[i, :n] if low_uvs.ndim == 3 else low_uvs[:n]
            uv_np = _normalize_uvs_to_unit(uv_i.detach().cpu().numpy(), log_prefix="[BakeNormalMapFromMesh]  ")

            lv = v_i.to(dev).float()
            lf = f_i.to(dev).long()
            # Tangents build the per-texel TBN; ApplyTextureToMesh recomputes the same basis on export.
            n_attr_i = low_n_attr[i, :n] if low_n_attr is not None else None
            low_n, tangents = _vertex_tangents_for_item(lv, lf, torch.from_numpy(uv_np).to(dev), n_attr_i, dev)

            hv_i, hf_i, _ = get_mesh_batch_item(high_poly, i if h_batch > 1 else 0)
            hv = hv_i.to(dev).float()
            hf = hf_i.to(dev).long()
            high_n = (high_n_attr[i, :hv.shape[0]].to(dev).float() if high_n_attr is not None
                      else _compute_vertex_normals(hv, hf))

            img = _bake_normal_map(
                hv, hf, high_n,
                lv.detach().cpu().numpy(), lf.detach().cpu().numpy().astype(np.uint32), uv_np,
                low_n, tangents, resolution, cage_distance=float(cage_distance),
                ignore_backfaces=bool(ignore_backfaces),
            )
            imgs.append(torch.from_numpy(np.ascontiguousarray(img)).float())

        normal_img = torch.stack([t.clamp(0.0, 1.0) for t in imgs], dim=0)
        return IO.NodeOutput(normal_img)


class BakeAmbientOcclusion(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BakeAmbientOcclusion",
            display_name="Bake Ambient Occlusion",
            category="latent/3d",
            description=(
                "Bakes an ambient-occlusion map from a high-poly mesh into a low-poly's UV "
                "layout (white = open, dark = crevices). Feed the UV-unwrapped low_poly and the "
                "high_poly it was decimated from. Outputs a grayscale IMAGE for "
                "ApplyTextureToMesh's occlusion input (packed into the ORM map / occlusionTexture)."
            ),
            inputs=[
                IO.Mesh.Input("low_poly"),
                IO.Mesh.Input("high_poly"),
                IO.Int.Input("resolution", default=1024, min=64, max=8192, step=64),
                IO.Int.Input("samples", default=64, min=4, max=1024, step=4,
                             tooltip="Rays per texel. More = smoother, slower. Raise if grainy."),
                IO.Float.Input("max_distance", default=0.5, min=0.01, max=2.0, step=0.01,
                               tooltip="Ray length, as a fraction of the bbox diagonal. "
                                       "Smaller = tighter, more local occlusion."),
                IO.Float.Input("strength", default=1.0, min=0.0, max=2.0, step=0.05,
                               tooltip="Scales the occlusion. >1 darkens, <1 lightens."),
                IO.Float.Input("bias", default=0.01, min=0.0001, max=0.2, step=0.0005,
                               tooltip="Ray origin lift off the surface, as a fraction of the bbox "
                                       "diagonal. Raise if even surfaces show dark blotches/holes."),
            ],
            outputs=[IO.Image.Output(display_name="occlusion")],
        )

    @classmethod
    def execute(cls, low_poly, high_poly, resolution, samples, max_distance, strength, bias):
        low_uvs = getattr(low_poly, "uvs", None)
        if low_uvs is None:
            raise ValueError(
                "BakeAmbientOcclusion: low_poly has no UVs. Connect the UV-unwrapped low-poly "
                "(the same one used for the other bakes); this node never unwraps.")
        dev = comfy.model_management.get_torch_device()
        low_n_attr = getattr(low_poly, "normals", None)
        B = int(low_poly.vertices.shape[0])
        h_batch = int(high_poly.vertices.shape[0])

        pbar = comfy.utils.ProgressBar(max(1, B))   # one tick per batch item
        imgs = []
        for i in range(B):
            v_i, f_i, _ = get_mesh_batch_item(low_poly, i)
            n = int(v_i.shape[0])
            if f_i.numel() == 0:
                logging.warning(f"BakeAmbientOcclusion: skipping batch {i} (empty mesh)")
                imgs.append(torch.ones((int(resolution), int(resolution), 3)))
                pbar.update(1)
                continue

            uv_i = low_uvs[i, :n] if low_uvs.ndim == 3 else low_uvs[:n]
            uv_np = _normalize_uvs_to_unit(uv_i.detach().cpu().numpy(), log_prefix="[BakeAmbientOcclusion]  ")
            lv = v_i.to(dev).float()
            lf = f_i.to(dev).long()
            low_n = (low_n_attr[i, :n].to(dev).float() if low_n_attr is not None
                     else _compute_vertex_normals(lv, lf))

            hv_i, hf_i, _ = get_mesh_batch_item(high_poly, i if h_batch > 1 else 0)
            img = _bake_ambient_occlusion(
                hv_i.to(dev).float(), hf_i.to(dev).long(),
                lv.detach().cpu().numpy(), lf.detach().cpu().numpy().astype(np.uint32), uv_np,
                low_n, resolution, num_samples=int(samples),
                max_distance=float(max_distance), strength=float(strength), bias=float(bias),
            )
            imgs.append(torch.from_numpy(np.ascontiguousarray(img)).float())
            pbar.update(1)

        ao_img = torch.stack([t.clamp(0.0, 1.0) for t in imgs], dim=0)
        return IO.NodeOutput(ao_img)


class SetMeshMaterial(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SetMeshMaterial",
            display_name="Set Mesh Material",
            category="latent/3d",
            description=(
                "Sets glTF material properties SaveGLB can't derive from textures: emissive "
                "(color + strength + optional texture), baseColor tint, metallic/roughness "
                "factors, normal scale, occlusion strength, double-sided. Place before SaveGLB."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("emissive_r", default=0.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("emissive_g", default=0.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("emissive_b", default=0.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("emissive_strength", default=1.0, min=0.0, max=100.0, step=0.1,
                               tooltip=">1 for HDR glow (KHR_materials_emissive_strength)."),
                IO.Image.Input("emissive_texture", optional=True),
                IO.Float.Input("base_color_r", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("base_color_g", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("base_color_b", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Float.Input("metallic_factor", default=-1.0, min=-1.0, max=1.0, step=0.01,
                               tooltip="-1 = leave auto; 0..1 overrides."),
                IO.Float.Input("roughness_factor", default=-1.0, min=-1.0, max=1.0, step=0.01,
                               tooltip="-1 = leave auto; 0..1 overrides."),
                IO.Float.Input("normal_scale", default=1.0, min=0.0, max=10.0, step=0.05),
                IO.Float.Input("occlusion_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                IO.Boolean.Input("double_sided", default=True),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, emissive_r, emissive_g, emissive_b, emissive_strength,
                base_color_r, base_color_g, base_color_b, metallic_factor, roughness_factor,
                normal_scale, occlusion_strength, double_sided, emissive_texture=None):
        out_mesh = copy.copy(mesh)
        material = dict(getattr(mesh, "material", {}) or {})   # merge over any prior material
        material.update({
            "emissive_factor": [float(emissive_r), float(emissive_g), float(emissive_b)],
            "emissive_strength": float(emissive_strength),
            "base_color_factor": [float(base_color_r), float(base_color_g), float(base_color_b), 1.0],
            "metallic_factor": float(metallic_factor),         # <0 => leave auto
            "roughness_factor": float(roughness_factor),
            "normal_scale": float(normal_scale),
            "occlusion_strength": float(occlusion_strength),
            "double_sided": bool(double_sided),
        })
        out_mesh.material = material
        if emissive_texture is not None:
            out_mesh.emissive = emissive_texture.float().clamp(0.0, 1.0).cpu()
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

    # Fill all boundary loops below the perimeter threshold.
    new_verts = []
    new_faces = []
    v_idx = v.shape[0]

    for loop in loops:
        loop_t = torch.tensor(loop, device=device, dtype=torch.long)
        loop_v = v[loop_t]

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


def _fill_holes_v2_gpu(verts, faces, max_perimeter, colors=None, fill_chains=False, max_verts=16):
    # Bidirectional (not pointer-doubling) CC labeling so low-id chains propagate
    # backward. Cycles-only by default; fill_chains=True opts into noisy chain fills.
    device = verts.device
    V = verts.shape[0]
    dtype = verts.dtype

    e_all = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e_sorted, _ = e_all.sort(dim=1)
    packed = e_sorted[:, 0].long() * V + e_sorted[:, 1].long()
    unique_packed, counts = torch.unique(packed, return_counts=True)
    boundary_packed = unique_packed[counts == 1]
    if boundary_packed.numel() == 0:
        return verts, faces, colors, 0
    is_b = torch.isin(packed, boundary_packed)

    b_directed = e_all[is_b]
    src = b_directed[:, 0].long()
    tgt = b_directed[:, 1].long()

    # Undirected bidirectional min-prop with path compression.
    labels = torch.arange(V, dtype=torch.long, device=device)
    for _ in range(64):
        edge_min = torch.minimum(labels[src], labels[tgt])
        new_labels = labels.clone()
        new_labels.scatter_reduce_(0, src, edge_min, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(0, tgt, edge_min, reduce="amin", include_self=True)
        new_labels = new_labels[new_labels]  # path compression
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    # After bidir-prop, labels[src] == labels[tgt], so labels[src] is the edge's component.
    edge_component = labels[src]
    unique_components, component_idx = torch.unique(edge_component, return_inverse=True)
    L = unique_components.shape[0]
    edge_count = torch.bincount(component_idx, minlength=L)
    edge_len = (verts[src] - verts[tgt]).norm(dim=-1)
    perim = torch.zeros(L, dtype=dtype, device=device)
    perim.scatter_add_(0, component_idx, edge_len)

    # Unique boundary verts per component, via packed (comp,vert) keys.
    pair_keys = torch.cat([
        component_idx.long() * V + src,
        component_idx.long() * V + tgt,
    ])
    pair_keys = torch.unique(pair_keys)
    pair_v = pair_keys % V
    pair_c = pair_keys // V
    vert_count = torch.bincount(pair_c, minlength=L)

    centroids = torch.zeros((L, 3), dtype=dtype, device=device)
    centroids.scatter_add_(0, pair_c[:, None].expand(-1, 3), verts[pair_v])
    centroids = centroids / vert_count.clamp_min(1).to(dtype).unsqueeze(-1)

    # Closed cycle ⇔ every boundary vert has degree 2 ⇔ vert_count == edge_count.
    is_cycle_component = (vert_count == edge_count) & (vert_count > 0)

    # Keep cycles (and chains if fill_chains) under perim/vert limits; centroid-fan
    # only works for small near-planar holes (else centroid lands off-surface → overlap).
    size_ok = (vert_count >= 3) & (vert_count <= max_verts) & (perim < max_perimeter)
    if fill_chains:
        keep_component = size_ok
    else:
        keep_component = is_cycle_component & size_ok
    if not keep_component.any():
        return verts, faces, colors, 0
    # Only centroid-fan components allocate a new vertex (threshold mirrored below).
    use_centroid_per_comp_pre = keep_component & (vert_count > 8)
    centroid_long = use_centroid_per_comp_pre.long()
    centroid_idx_per_comp = V + centroid_long.cumsum(0) - 1

    # vertex-fan (small cycles): boundary vert as apex, on-surface. centroid-fan (large):
    # insert centroid (near-planar only, but avoids skinny tris on big holes).
    CENTROID_FAN_THRESHOLD = 8

    edge_kept = keep_component[component_idx]
    edge_comp = component_idx[edge_kept]
    kept_src = src[edge_kept]
    kept_tgt = tgt[edge_kept]

    use_centroid_per_comp = keep_component & (vert_count > CENTROID_FAN_THRESHOLD)
    use_centroid_per_edge = use_centroid_per_comp[edge_comp]

    fan_pieces = []
    # Centroid-fan branch
    if use_centroid_per_edge.any():
        kept_centroid = centroid_idx_per_comp[edge_comp[use_centroid_per_edge]]
        fan_pieces.append(torch.stack([
            kept_tgt[use_centroid_per_edge],
            kept_src[use_centroid_per_edge],
            kept_centroid,
        ], dim=1).to(faces.dtype))

    # Vertex-fan branch (small cycles)
    use_vertex_fan_per_comp = keep_component & (vert_count <= CENTROID_FAN_THRESHOLD)
    if use_vertex_fan_per_comp.any():
        # Apex = smallest-id boundary vert; fan (apex, src, tgt) skipping apex-incident edges.
        apex_per_comp = labels[unique_components]
        vf_mask = use_vertex_fan_per_comp[edge_comp]
        if vf_mask.any():
            vf_src = kept_src[vf_mask]
            vf_tgt = kept_tgt[vf_mask]
            vf_comp = edge_comp[vf_mask]
            vf_apex = apex_per_comp[vf_comp]
            non_apex = (vf_src != vf_apex) & (vf_tgt != vf_apex)
            fan_pieces.append(torch.stack([
                vf_tgt[non_apex], vf_src[non_apex], vf_apex[non_apex],
            ], dim=1).to(faces.dtype))

    fan_faces = torch.cat(fan_pieces, dim=0) if fan_pieces else torch.empty((0, 3), dtype=faces.dtype, device=device)

    # Close open chains (centroid-fan only; no-op when fill_chains=False).
    if fill_chains:
        vert_degree = torch.zeros(V, dtype=torch.long, device=device)
        vert_degree.scatter_add_(0, src, torch.ones_like(src))
        vert_degree.scatter_add_(0, tgt, torch.ones_like(tgt))
        is_endpoint = (vert_degree[pair_v] == 1) & use_centroid_per_comp_pre[pair_c]
        if is_endpoint.any():
            ep_v = pair_v[is_endpoint]
            ep_c = pair_c[is_endpoint]
            order = torch.argsort(ep_c, stable=True)
            ep_v_sorted = ep_v[order]
            ep_c_sorted = ep_c[order]
            ep_count_per_c = torch.bincount(ep_c_sorted, minlength=L)
            is_chain_comp = ep_count_per_c == 2
            ep_is_chain = is_chain_comp[ep_c_sorted]
            if ep_is_chain.any():
                chain_ep_v = ep_v_sorted[ep_is_chain]
                chain_ep_c = ep_c_sorted[ep_is_chain]
                assert chain_ep_v.numel() % 2 == 0
                chain_ep_v = chain_ep_v.view(-1, 2)
                chain_ep_c = chain_ep_c.view(-1, 2)[:, 0]
                close_centroid = centroid_idx_per_comp[chain_ep_c]
                close_faces = torch.stack(
                    [chain_ep_v[:, 0], chain_ep_v[:, 1], close_centroid], dim=1
                ).to(faces.dtype)
                fan_faces = torch.cat([fan_faces, close_faces], dim=0)

    # Only centroid-fan components contribute a new vertex; vertex-fan reuses existing.
    new_centroids_v = centroids[use_centroid_per_comp_pre]
    out_v = torch.cat([verts, new_centroids_v], dim=0)
    out_f = torch.cat([faces, fan_faces], dim=0)

    out_c = colors
    if colors is not None:
        c_sum = torch.zeros((L, colors.shape[1]), dtype=colors.dtype, device=device)
        c_sum.scatter_add_(
            0, pair_c[:, None].expand(-1, colors.shape[1]), colors[pair_v])
        c_avg = c_sum / vert_count.clamp_min(1).to(colors.dtype).unsqueeze(-1)
        out_c = torch.cat([colors, c_avg[use_centroid_per_comp_pre]], dim=0)

    return out_v, out_f, out_c, int(keep_component.sum().item())


def weld_vertices_fn(vertices, faces, epsilon=None, epsilon_rel=1e-5, colors=None):
    """Merge coincident vertices via L_inf grid quantization. `epsilon` absolute (None →
    epsilon_rel*bbox_diag); colors averaged per cluster. Returns (v, f, colors, n_welded)."""
    if vertices.ndim == 3:
        v_out, f_out, c_out = [], [], [] if colors is not None else None
        total = 0
        for i in range(vertices.shape[0]):
            ci = colors[i] if colors is not None else None
            v_i, f_i, c_i, n = weld_vertices_fn(vertices[i], faces[i], epsilon, epsilon_rel, ci)
            v_out.append(v_i)
            f_out.append(f_i)
            total += n
            if c_out is not None:
                c_out.append(c_i)
        max_v = max(v.shape[0] for v in v_out)
        for i in range(len(v_out)):
            pad_n = max_v - v_out[i].shape[0]
            if pad_n > 0:
                v_out[i] = torch.cat([v_out[i],
                    torch.zeros(pad_n, 3, device=v_out[i].device, dtype=v_out[i].dtype)], dim=0)
                if c_out is not None:
                    c_out[i] = torch.cat([c_out[i],
                        torch.zeros(pad_n, c_out[i].shape[1], device=c_out[i].device, dtype=c_out[i].dtype)], dim=0)
        c_stack = torch.stack(c_out) if c_out is not None else None
        return torch.stack(v_out), torch.stack(f_out), c_stack, total

    if vertices.shape[0] == 0:
        return vertices, faces, colors, 0
    device = vertices.device
    if epsilon is None:
        bbox = vertices.max(dim=0)[0] - vertices.min(dim=0)[0]
        eps = torch.norm(bbox) * float(epsilon_rel)
        eps = max(float(eps.item()), 1e-12)
    else:
        eps = float(epsilon)
        if eps <= 0:
            return vertices, faces, colors, 0

    scale = 1.0 / eps
    bbox_min = vertices.min(dim=0)[0]
    q = ((vertices - bbox_min) * scale).round().to(torch.int64)
    extent = ((vertices.max(dim=0)[0] - bbox_min) * scale).round().to(torch.int64) + 2
    key = (q[:, 0] * extent[1] + q[:, 1]) * extent[2] + q[:, 2]
    unique_key, inv = torch.unique(key, return_inverse=True)
    n_unique = unique_key.shape[0]
    if n_unique == vertices.shape[0]:
        return vertices, faces, colors, 0

    counts = torch.zeros(n_unique, dtype=vertices.dtype, device=device)
    counts.scatter_add_(0, inv, torch.ones(vertices.shape[0], dtype=vertices.dtype, device=device))
    counts_div = counts.unsqueeze(-1).clamp_min(1.0)

    new_verts = torch.zeros((n_unique, 3), dtype=vertices.dtype, device=device)
    new_verts.scatter_add_(0, inv.unsqueeze(-1).expand_as(vertices), vertices)
    new_verts = new_verts / counts_div

    new_colors = None
    if colors is not None:
        new_colors = torch.zeros((n_unique, colors.shape[1]), dtype=colors.dtype, device=device)
        new_colors.scatter_add_(0, inv.unsqueeze(-1).expand_as(colors), colors)
        new_colors = new_colors / counts_div.to(colors.dtype)

    new_faces = inv[faces.long()].to(faces.dtype) if faces.numel() > 0 else faces
    return new_verts, new_faces, new_colors, int(vertices.shape[0] - n_unique)


def fill_holes_v2_fn(vertices, faces, max_perimeter=0.03, colors=None, weld_epsilon_rel=1e-5, fill_chains=False, max_verts=16):
    """Batched v2 GPU hole-filler (v1 CPU walker fallback on non-CUDA). Pre-welds verts
    first — boundary detection needs shared edges; `weld_epsilon_rel=0` skips it."""
    if vertices.ndim == 3:
        v_list, f_list, c_list = [], [], [] if colors is not None else None
        pbar = comfy.utils.ProgressBar(vertices.shape[0])
        for i in range(vertices.shape[0]):
            ci = colors[i] if colors is not None else None
            v_i, f_i, c_i = fill_holes_v2_fn(vertices[i], faces[i], max_perimeter, ci, weld_epsilon_rel, fill_chains, max_verts)
            v_list.append(v_i)
            f_list.append(f_i)
            if c_list is not None:
                c_list.append(c_i)
            pbar.update(1)
        max_v = max(v.shape[0] for v in v_list)
        for i in range(len(v_list)):
            pad_n = max_v - v_list[i].shape[0]
            if pad_n > 0:
                v_list[i] = torch.cat([v_list[i],
                    torch.zeros(pad_n, 3, device=v_list[i].device, dtype=v_list[i].dtype)], dim=0)
                if c_list is not None:
                    c_list[i] = torch.cat([c_list[i],
                        torch.zeros(pad_n, c_list[i].shape[1], device=c_list[i].device, dtype=c_list[i].dtype)], dim=0)
        c_out = torch.stack(c_list) if c_list is not None else None
        return torch.stack(v_list), torch.stack(f_list), c_out

    if faces.numel() == 0:
        return vertices, faces, colors
    # Adaptive weld: welded surfaces have V/F ≈ 0.5-1.0; V/F > 1 means unwelded (hole-fill
    # would emit a bogus tri per face). Double epsilon until V/F < WELDED_THRESHOLD or WELD_CAP.
    if weld_epsilon_rel > 0:
        eps = float(weld_epsilon_rel)
        WELD_CAP = 1e-2          # ≈ 10 voxels at 1024-res
        WELDED_THRESHOLD = 1.0   # V/F below this is welded enough
        total_welded = 0
        n_escalations = 0
        while True:
            vertices, faces, colors, n = weld_vertices_fn(
                vertices, faces, epsilon=None, epsilon_rel=eps, colors=colors,
            )
            total_welded += n
            ratio = vertices.shape[0] / max(faces.shape[0], 1)
            if ratio < WELDED_THRESHOLD or eps >= WELD_CAP:
                break
            eps = min(eps * 2.0, WELD_CAP)
            n_escalations += 1
        if total_welded > 0 or n_escalations > 0:
            tag = f" (escalated weld epsilon_rel→{eps:.1e} after {n_escalations} step{'s' if n_escalations != 1 else ''})" if n_escalations > 0 else ""
            logging.info(f"[FillHoles] pre-welded {total_welded} verts, V/F={ratio:.2f}{tag}")
        if ratio >= WELDED_THRESHOLD:
            logging.warning(
                f"[FillHoles] even at weld epsilon_rel={WELD_CAP} the mesh stays "
                f"unwelded (V/F={ratio:.2f}, want < {WELDED_THRESHOLD}). Source mesh has "
                f"duplicate verts at distances >{WELD_CAP}× bbox; fix upstream "
                f"(decimate node settings) or run WeldVertices manually with a larger epsilon."
            )
    if vertices.device.type == "cuda":
        out_v, out_f, out_c, _ = _fill_holes_v2_gpu(vertices, faces, max_perimeter, colors, fill_chains, max_verts)
        return out_v, out_f, out_c
    # CPU fallback: v1 walker (no attribute prop, but topologically equivalent for manifold boundary).
    out_v, out_f = fill_holes_fn(vertices, faces, max_perimeter=max_perimeter)
    return out_v, out_f, colors


def _process_mesh_batch(mesh, per_item_fn):
    """Dispatch list/batched/single mesh, extract colors, stack results."""
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


def _fmt_count(n) -> str:
    """Compact integer for status lines, e.g. 853, 12.3K, 1.23M."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return str(n)


def _fmt_face_change(n_in, n_out) -> str:
    """'faces: 1.23M → 200K  (-84%)' — the count delta for decimate/remesh status."""
    n_in, n_out = int(n_in), int(n_out)
    pct = f"  ({(n_out - n_in) / n_in * 100:+.0f}%)" if n_in else ""
    return f"faces: {_fmt_count(n_in)} → {_fmt_count(n_out)}{pct}"


class DecimateMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        # qem sub-widgets show only when 'qem' is selected (DynamicCombo).
        placement_options = [
            IO.DynamicCombo.Option(key="midpoint", inputs=[]),
            IO.DynamicCombo.Option(key="qem", inputs=[
                IO.Float.Input("line_quadric_weight", default=0.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Per-edge line-quadric weight; preserves sharp ridges/valleys. 0 = off."),
                IO.Float.Input("feature_edge_quadric_weight", default=0.0, min=0.0, max=1000.0, step=1.0,
                               tooltip="Extra quadric weight on dihedral feature edges (creases). 0 = off."),
                IO.Float.Input("feature_edge_min_dihedral_deg", default=30.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Min dihedral angle (deg) to count an edge as a feature edge."),
                IO.Boolean.Input("clamp_v_to_edge", default=True,
                                 tooltip="Project the QEM-optimal position onto the collapsed edge segment."),
            ]),
        ]
        return IO.Schema(
            node_id="DecimateMesh",
            display_name="Decimate Mesh",
            category="latent/3d",
            description=(
                "Simplifies a mesh to a target face count using QEM, on the active compute "
                "device. 'midpoint' is the cumesh-faithful preset (best quality, preserves thin "
                "features / hair); 'qem' places verts at the QEM optimum with line/feature-edge "
                "controls. Output stays welded."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("target_face_count", default=200_000, min=0, max=50_000_000,
                             tooltip="Target max faces. 0 disables."),
                IO.DynamicCombo.Input("placement_mode", options=placement_options,
                                      display_name="placement_mode",
                                      tooltip="midpoint: cumesh-faithful (recommended). qem: QEM-optimal placement."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mesh, target_face_count, placement_mode):
        mode = placement_mode.get("placement_mode", "midpoint")
        if mode == "qem":
            # QEM-optimum placement; rest inherit defaults.
            cfg = QEMConfig(
                placement_mode="qem",
                line_quadric_weight=float(placement_mode.get("line_quadric_weight", 0.0)),
                feature_edge_quadric_weight=float(placement_mode.get("feature_edge_quadric_weight", 0.0)),
                feature_edge_min_dihedral_deg=float(placement_mode.get("feature_edge_min_dihedral_deg", 30.0)),
                clamp_v_to_edge=bool(placement_mode.get("clamp_v_to_edge", True)),
            )
        else:
            cfg = QEMConfig()  # midpoint defaults

        # ComfyUI passes meshes on CPU (QEM much slower there); compute on device, return on original.
        compute_device = comfy.model_management.get_torch_device()

        counts = {"in": 0, "out": 0}

        def _fn(v, f, c):
            counts["in"] += int(f.shape[0])
            if target_face_count > 0 and f.shape[0] > target_face_count:
                try:
                    src_device = v.device
                    rv, rf, rc, _rn, _rs = qem_decimate_simplify(
                        v.to(compute_device), f.to(compute_device), int(target_face_count),
                        colors=(c.to(compute_device) if c is not None else None),
                        config=cfg)
                    v = rv.to(src_device)
                    f = rf.to(src_device)
                    if rc is not None:
                        c = rc.to(src_device)
                except Exception as e:
                    logging.warning(f"DecimateMesh: QEM simplify failed, passing mesh through unchanged: {e!r}")
            counts["out"] += int(f.shape[0])
            return v, f, c

        result = _process_mesh_batch(mesh, _fn)

        # Display the face reduction on the node
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                _fmt_face_change(counts["in"], counts["out"]), cls.hidden.unique_id)

        return result


class RemeshMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        # sub-widgets show per sign_mode (DynamicCombo).
        sign_mode_options = [
            IO.DynamicCombo.Option(key="udf", inputs=[
                IO.Boolean.Input("qef", default=False, advanced=True,
                                 tooltip="QEF dual-vertex placement for sharper edges."),
                IO.Boolean.Input("drop_inverted_components", default=False, advanced=True,
                                 tooltip="Drop inward-normal (negative-volume) closed components — the UDF inner shell."),
                IO.Boolean.Input("drop_enclosed_components", default=False, advanced=True,
                                 tooltip="Drop components inside the largest's bbox that fail a point-in-mesh raycast. Disable for legitimate nested parts."),
            ]),
            IO.DynamicCombo.Option(key="sdf", inputs=[
                IO.Boolean.Input("qef", default=True,
                                 tooltip="QEF dual-vertex placement (recovers sharp features) vs edge-crossing centroid."),
                IO.Boolean.Input("manifold", default=False,
                                 tooltip="Manifold Dual Contouring: 1-4 dual verts/voxel for multi-sheet cases. Slower."),
            ]),
        ]
        return IO.Schema(
            node_id="RemeshMesh",
            display_name="Remesh Mesh (Narrow-Band DC)",
            category="latent/3d",
            description=(
                "Re-extracts a uniformly tessellated mesh via a narrow-band distance field + Dual "
                "Contouring, on the active compute device. Normalizes messy / non-manifold / "
                "self-intersecting topology; run before DecimateMesh to hit an exact face count. "
                "Output stays welded."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("resolution", default=512, min=32, max=1024,
                             tooltip="Voxel grid resolution (output density). 256 ~ 100k faces, 512 ~ 1M. "
                                     "For an exact face count, follow with DecimateMesh."),
                IO.DynamicCombo.Input("sign_mode", options=sign_mode_options, display_name="sign_mode",
                                      tooltip="udf: robust to messy/non-manifold input. sdf: clean single "
                                              "surface with QEF sharp-feature recovery, but needs consistent winding."),
                IO.Float.Input("band", default=1.0, min=0.5, max=4.0, step=0.1, advanced=True,
                               tooltip="Narrow-band width in voxel units. In UDF mode also offsets the surface."),
                IO.Float.Input("project_back", default=0.0, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="Lerp verts toward the original surface (0 = pure DC, 1 = snapped)."),
                IO.Boolean.Input("fix_poles", default=False, advanced=True,
                                 tooltip="Collapse valence-3 vertex pairs (DC T-junction artifact)."),
                IO.Int.Input("smooth_iters", default=0, min=0, max=20,
                             tooltip="Taubin smoothing iters (0 = off). 2-3 cleans DC stairstepping; higher rounds off QEF edges."),
                IO.Float.Input("drop_small_components", default=0.01, min=0.0, max=0.5, step=0.005, advanced=True,
                               tooltip="Drop components below this fraction of the largest's face count. 0 disables."),
                IO.Int.Input("precluster_max_verts", default=8_000_000, min=0, max=50_000_000, advanced=True,
                             tooltip="Cap input vertex count before the field queries, inputs above this are cluster-decimated to it first. Prevents OOM on huge meshes."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mesh, resolution, sign_mode, band,
                project_back, fix_poles, smooth_iters,
                drop_small_components, precluster_max_verts):
        mode = sign_mode.get("sign_mode", "udf")
        # mode-specific sub-widgets (absent → defaults)
        qef = bool(sign_mode.get("qef", True))
        manifold = bool(sign_mode.get("manifold", False))
        drop_inverted_components = bool(sign_mode.get("drop_inverted_components", False))
        drop_enclosed_components = bool(sign_mode.get("drop_enclosed_components", False))

        # ComfyUI passes meshes on CPU (remesh far faster on GPU); compute on device, return on original.
        compute_device = comfy.model_management.get_torch_device()
        counts = {"in": 0, "out": 0}

        def _fn(v, f, c):
            counts["in"] += int(f.shape[0])
            try:
                src_device = v.device
                vv = v.to(compute_device).float()
                ff = f.to(compute_device).to(torch.int64)
                cc = c.to(compute_device).float() if c is not None else None

                # cluster-decimate very large inputs before field queries
                if precluster_max_verts > 0 and vv.shape[0] > precluster_max_verts:
                    vv, ff, cc = qem_cluster_decimate(
                        vv, ff, target_verts=int(precluster_max_verts), colors=cc)

                # Fixed [-0.5,0.5] cube domain (matches cumesh/TRELLIS2); scale ≈ 1.0 any resolution.
                rs_scale = (resolution + 3.0 * band) / resolution
                rs_center = torch.zeros(3, dtype=vv.dtype, device=compute_device)

                rv, rf, rc = remesh_narrow_band_dc(
                    vv, ff,
                    resolution=int(resolution),
                    band=float(band), project_back=float(project_back),
                    qef=qef, sign_mode=mode,
                    manifold=manifold, fix_poles=bool(fix_poles),
                    smooth_iters=int(smooth_iters),
                    drop_small_components=float(drop_small_components),
                    drop_inverted_components=drop_inverted_components,
                    drop_enclosed_components=drop_enclosed_components,
                    scale=rs_scale, center=rs_center, colors=cc)

                v = rv.to(src_device)
                f = rf.to(src_device)
                c = rc.to(src_device) if rc is not None else None
            except Exception as e:
                logging.warning(f"RemeshMesh: remesh failed, passing mesh through unchanged: {e!r}")
            counts["out"] += int(f.shape[0])
            return v, f, c

        result = _process_mesh_batch(mesh, _fn)

        # Display the face change on the node
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                _fmt_face_change(counts["in"], counts["out"]), cls.hidden.unique_id)

        return result


def _pack_uv_meshes(vs, fs, uvs, colors):
    """Pack per-item (verts, faces, uvs[, colors]) into a MESH; stack if single, else pad."""
    if len(vs) == 1:
        m = Types.MESH(vertices=vs[0].unsqueeze(0), faces=fs[0].unsqueeze(0), uvs=uvs[0].unsqueeze(0))
        if colors is not None:
            m.vertex_colors = colors[0].unsqueeze(0)
        return m
    bsz = len(vs)
    dev = vs[0].device
    maxv = max(v.shape[0] for v in vs)
    maxf = max(f.shape[0] for f in fs)
    pv = vs[0].new_zeros((bsz, maxv, 3))
    pf = fs[0].new_zeros((bsz, maxf, 3))
    pu = uvs[0].new_zeros((bsz, maxv, 2))
    for i, (v, f, u) in enumerate(zip(vs, fs, uvs)):
        pv[i, :v.shape[0]] = v
        pf[i, :f.shape[0]] = f
        pu[i, :u.shape[0]] = u
    vc = torch.tensor([v.shape[0] for v in vs], device=dev, dtype=torch.int64)
    fc = torch.tensor([f.shape[0] for f in fs], device=dev, dtype=torch.int64)
    m = Types.MESH(vertices=pv, faces=pf, uvs=pu, vertex_counts=vc, face_counts=fc)
    if colors is not None:
        pc = colors[0].new_zeros((bsz, maxv, colors[0].shape[1]))
        for i, c in enumerate(colors):
            pc[i, :c.shape[0]] = c
        m.vertex_colors = pc
    return m


def _uv_weld_vertices(v, f, weld_distance):
    """Merge coincident verts; returns (welded_v, welded_f, welded_to_orig); last None if no welding."""
    v_np = v.cpu().numpy()
    f_np = f.cpu().numpy()
    if v_np.size == 0:
        return v, f, None
    extent = float(np.linalg.norm(v_np.max(axis=0) - v_np.min(axis=0)))
    tol = weld_distance if weld_distance > 0.0 else 1e-5 * extent
    if tol <= 0.0:
        return v, f, None
    keys = np.round(v_np / tol).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_unique = int(inv.max()) + 1
    if n_unique >= v_np.shape[0]:
        return v, f, None
    v_welded = np.zeros((n_unique, 3), dtype=np.float32)
    counts = np.zeros(n_unique, dtype=np.int64)
    np.add.at(v_welded, inv, v_np)
    np.add.at(counts, inv, 1)
    v_welded /= counts[:, None]
    welded_to_orig = np.empty(n_unique, dtype=np.int64)
    welded_to_orig[inv] = np.arange(v_np.shape[0], dtype=np.int64)
    v_new = torch.from_numpy(v_welded).to(v.dtype).to(v.device)
    f_new = torch.from_numpy(inv[f_np]).to(f.dtype).to(f.device)
    return v_new, f_new, welded_to_orig


def _uv_unwrap(positions, indices, segmenter, resolution, padding, weld_distance):
    """UV-unwrap a single mesh; returns (vmapping, indices, uvs); vmapping maps each output
    vertex to an input vertex (seam verts duplicated)."""
    v_in = positions.to(torch.float32)
    f_in = indices.to(torch.long).reshape(-1, 3)
    v_in, f_in, welded_to_orig = _uv_weld_vertices(v_in, f_in, weld_distance)

    # drop degenerate faces (repeated index; corrupt edge adjacency)
    degen = ((f_in[:, 0] == f_in[:, 1]) | (f_in[:, 1] == f_in[:, 2]) | (f_in[:, 2] == f_in[:, 0]))
    if bool(degen.any()):
        f_in = f_in[~degen]

    mesh = _uv_mesh.build_mesh(v_in, f_in)
    ff = mesh.face_face
    if ff.numel() and float((ff >= 0).float().mean().item()) < 0.25:
        logging.warning("[uv_unwrap] mesh face-adjacency < 25% — vertices appear un-welded "
                        "(triangle soup); UV charts will be per-face. Raise weld_distance.")

    if segmenter == "pec":
        if mesh.faces.device.type != "cuda":
            raise RuntimeError("segmenter='pec' requires a CUDA mesh; use 'adaptive' for CPU.")
        face_chart = _uv_seg.cluster_charts_pec(mesh, target_chart_count=0, max_cost=1.0)
    elif segmenter == "adaptive":
        face_chart = _uv_seg.segment_charts(mesh, max_cost=2.0, target_chart_count=0)
    else:
        raise ValueError(f"unknown segmenter '{segmenter}'. valid: pec, adaptive")

    n_charts = int(face_chart.max().item()) + 1 if face_chart.numel() else 0
    areas_cpu = _uv_mesh.chart_3d_areas(mesh.face_area, face_chart, n_charts).detach().cpu()

    # per-chart loop on CPU/numpy to avoid per-chart GPU sync
    face_chart_np = face_chart.cpu().numpy()
    faces_np = mesh.faces.cpu().numpy()
    vertices_np = mesh.vertices.cpu().numpy()
    face_face_np = mesh.face_face.cpu().numpy()
    sorted_face_idx_np = np.argsort(face_chart_np, kind="stable")
    chart_counts_np = np.bincount(face_chart_np, minlength=n_charts)
    chart_offsets_np = np.empty(n_charts + 1, dtype=np.int64)
    chart_offsets_np[0] = 0
    np.cumsum(chart_counts_np, out=chart_offsets_np[1:])

    all_chart_uvs, all_chart_3d_areas, all_chart_uv_areas, all_chart_faces = [], [], [], []
    chart_records = []
    for c in range(n_charts):
        gfi_np = sorted_face_idx_np[chart_offsets_np[c]:chart_offsets_np[c + 1]]
        chart_faces_global = faces_np[gfi_np]
        used_verts_np = np.unique(chart_faces_global)
        local_faces_np = np.searchsorted(used_verts_np, chart_faces_global)
        local_verts_np = vertices_np[used_verts_np]
        ff_global = face_face_np[gfi_np]
        ff_safe = np.maximum(ff_global, 0)
        nb_chart = np.where(ff_global >= 0, face_chart_np[ff_safe], -1)
        keep = (ff_global >= 0) & (nb_chart == c)
        local_neighbor = np.searchsorted(gfi_np, ff_safe)
        local_ff_np = np.where(keep, local_neighbor, -1)

        lf = torch.from_numpy(local_faces_np)
        uvs = _uv_param.parametrize_chart(
            torch.from_numpy(local_verts_np), lf, torch.from_numpy(local_ff_np))
        ua, ub, uc = uvs[lf[:, 0]], uvs[lf[:, 1]], uvs[lf[:, 2]]
        uv_area_sum = float(0.5 * (
            (ub[:, 0] - ua[:, 0]) * (uc[:, 1] - ua[:, 1])
            - (uc[:, 0] - ua[:, 0]) * (ub[:, 1] - ua[:, 1])).abs().sum().item())
        chart_records.append({"local_faces": lf, "vmap": torch.from_numpy(used_verts_np),
                              "global_face_idx": torch.from_numpy(gfi_np)})
        all_chart_uvs.append(uvs)
        all_chart_3d_areas.append(float(areas_cpu[c].item()))
        all_chart_uv_areas.append(uv_area_sum)
        all_chart_faces.append(lf)

    # auto-tune texel density toward `resolution` (~0.62 pack fill)
    total_3d_area = sum(all_chart_3d_areas) or 1.0
    target_dim = float(resolution) if resolution > 0 else 1024.0
    tex_per_unit = math.sqrt((target_dim * target_dim) * 0.62 / total_3d_area)

    placements, atlas_w, atlas_h = _uv_pack.pack_bitmap(
        all_chart_uvs, all_chart_3d_areas, all_chart_uv_areas, all_chart_faces,
        texels_per_unit=tex_per_unit, padding_texels=padding)
    placed = _uv_pack.apply_placements(all_chart_uvs, placements, atlas_w, atlas_h)

    n_in_faces = mesh.faces.shape[0]
    out_indices = np.zeros((n_in_faces, 3), dtype=np.int64)
    out_uvs_list, out_vmap_list, v_cursor = [], [], 0
    for c, rec in enumerate(chart_records):
        vmap_np = rec["vmap"].cpu().numpy()
        local_faces_np = rec["local_faces"].cpu().numpy()
        global_face_idx = rec["global_face_idx"].cpu().numpy()
        out_uvs_list.append(placed[c].cpu().numpy())
        if welded_to_orig is not None:
            vmap_np = welded_to_orig[vmap_np]
        out_vmap_list.append(vmap_np)
        out_indices[global_face_idx] = local_faces_np + v_cursor
        v_cursor += vmap_np.shape[0]

    vmapping_out = np.concatenate(out_vmap_list) if out_vmap_list else np.empty(0, dtype=np.int64)
    uvs_out = np.concatenate(out_uvs_list) if out_uvs_list else np.empty((0, 2), dtype=np.float32)
    return vmapping_out, out_indices, uvs_out


class UnwrapMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="UnwrapMesh",
            display_name="Unwrap Mesh UVs",
            category="latent/3d",
            description=(
                "Generates a UV atlas (pure-torch, no xatlas): segments the surface into charts, "
                "parameterizes each, packs into a [0,1] atlas. Seam verts duplicated. Run after "
                "DecimateMesh/RemeshMesh, before texture baking."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Combo.Input("segmenter", options=["pec", "adaptive"], default="pec",
                               tooltip="pec: fast parallel-edge-collapse charting (CUDA; CPU falls back to "
                                       "adaptive). adaptive: CPU, slower."),
                IO.Int.Input("resolution", default=1024, min=0, max=8192, step=256,
                             tooltip="Target atlas resolution for texel-density auto-scale (0 = fit-to-content)."),
                IO.Int.Input("padding", default=1, min=0, max=16,
                             tooltip="Texel padding between charts."),
                IO.Float.Input("weld_distance", default=0.0, min=0.0, max=1.0, step=0.0001,
                               tooltip="Coincident-vert merge radius as a fraction of mesh extent (0 = auto). "
                                       "Raise to ~0.001 if you get per-triangle charts (unwelded input)."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mesh, segmenter, resolution, padding, weld_distance):
        compute_device = comfy.model_management.get_torch_device()
        seg = segmenter
        if seg == "pec" and compute_device.type != "cuda":
            seg = "adaptive"
        seg_device = compute_device if seg == "pec" else torch.device("cpu")

        is_list = isinstance(mesh.vertices, list)
        is_batched = not is_list and mesh.vertices.ndim == 3
        bsz = len(mesh.vertices) if is_list else (mesh.vertices.shape[0] if is_batched else 1)
        bar = comfy.utils.ProgressBar(bsz)

        out_v, out_f, out_uv, out_c = [], [], [], []
        for i in range(bsz):
            if is_list or is_batched:
                vi, fi = mesh.vertices[i], mesh.faces[i]
                ci = None
                vc = getattr(mesh, "vertex_colors", None)
                if vc is not None:
                    ci = vc[i] if (isinstance(vc, list) or vc.ndim == 3) else vc
            else:
                vi, fi = mesh.vertices, mesh.faces
                ci = getattr(mesh, "vertex_colors", None)

            src_device = vi.device
            vnp = vi.detach().cpu().numpy().astype(np.float32)
            extent = float(np.linalg.norm(vnp.max(0) - vnp.min(0))) if vnp.shape[0] else 0.0
            weld_abs = weld_distance * extent if weld_distance > 0.0 else 0.0

            vmapping, indices, uvs = _uv_unwrap(
                vi.to(seg_device).float(), fi.to(seg_device).long(),
                seg, int(resolution), int(padding), weld_abs)
            uvs = uvs.copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]                       # UV y flipped vs trimesh

            out_v.append(torch.from_numpy(vnp[vmapping]).to(src_device))
            out_f.append(torch.from_numpy(indices).to(device=src_device, dtype=torch.long))
            out_uv.append(torch.from_numpy(uvs.astype(np.float32)).to(src_device))
            if ci is not None:
                cnp = ci.detach().cpu().numpy()
                out_c.append(torch.from_numpy(np.ascontiguousarray(cnp[vmapping])).to(src_device))
            bar.update(1)

        out_mesh = _pack_uv_meshes(out_v, out_f, out_uv, out_c if out_c else None)
        if getattr(mesh, "texture", None) is not None:
            out_mesh.texture = mesh.texture

        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                f"UV: {_fmt_count(out_v[0].shape[0])} verts / {_fmt_count(out_f[0].shape[0])} faces"
                f" · atlas ~{resolution}px",
                cls.hidden.unique_id)
        return IO.NodeOutput(out_mesh)


def _uv_sorted_edge_keys(indices: np.ndarray):
    """Sorted undirected edge keys; returns (sorted_keys, face_id, lo, hi, first_mask)."""
    a = indices.ravel().astype(np.int64)
    b = np.roll(indices, -1, axis=1).ravel().astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    V = int(indices.max()) + 1
    key = lo * V + hi
    order = np.argsort(key, kind="stable")
    sk = key[order]
    fid = (np.arange(a.size, dtype=np.int64) // 3)[order]
    first = np.ones(sk.size, dtype=bool)
    first[1:] = sk[1:] != sk[:-1]
    return sk, fid, lo[order], hi[order], first


def _uv_faces_to_chart_ids(indices: np.ndarray) -> np.ndarray:
    """Chart = connected component of faces sharing a (non-seam-duplicated) UV vertex."""
    F = indices.shape[0]
    if F == 0:
        return np.empty(0, dtype=np.int64)
    _sk, fid, _lo, _hi, first = _uv_sorted_edge_keys(indices)
    group_id = np.cumsum(first) - 1
    starts = np.nonzero(first)[0]
    rows = fid[starts[group_id[~first]]]
    cols = fid[~first]
    if rows.size == 0:
        return np.arange(F, dtype=np.int64)
    adj = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(F, F))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


_UV_TAB20 = np.array([
    [0.121568627, 0.466666667, 0.705882353], [0.682352941, 0.780392157, 0.909803922],
    [1.000000000, 0.498039216, 0.054901961], [1.000000000, 0.733333333, 0.470588235],
    [0.172549020, 0.627450980, 0.172549020], [0.596078431, 0.874509804, 0.541176471],
    [0.839215686, 0.152941176, 0.156862745], [1.000000000, 0.596078431, 0.588235294],
    [0.580392157, 0.403921569, 0.741176471], [0.772549020, 0.690196078, 0.835294118],
    [0.549019608, 0.337254902, 0.294117647], [0.768627451, 0.611764706, 0.580392157],
    [0.890196078, 0.466666667, 0.760784314], [0.968627451, 0.713725490, 0.823529412],
    [0.498039216, 0.498039216, 0.498039216], [0.780392157, 0.780392157, 0.780392157],
    [0.737254902, 0.741176471, 0.133333333], [0.858823529, 0.858823529, 0.552941176],
    [0.090196078, 0.745098039, 0.811764706], [0.619607843, 0.854901961, 0.898039216],
], dtype=np.float32)


def _uv_palette(n: int) -> np.ndarray:
    rng = np.random.RandomState(42)
    perm = rng.permutation(max(1, n))
    out = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        out[i] = _UV_TAB20[perm[i % len(perm)] % 20]
    return out


def _uv_render_atlas(uvs_np, indices_np, resolution, device,
                     bg=(0.13, 0.13, 0.13), edge=(0.0, 0.0, 0.0)):
    """Tile-based torch rasterizer of the UV atlas (charts colored, borders outlined); (H,W,3)."""
    w = h = int(resolution)
    chart_ids_np = _uv_faces_to_chart_ids(indices_np)
    uvs = torch.from_numpy(uvs_np).to(device=device, dtype=torch.float32)
    indices = torch.from_numpy(indices_np).to(device=device, dtype=torch.long)
    chart_ids = torch.from_numpy(chart_ids_np).to(device=device, dtype=torch.long)

    img = torch.tensor(bg, dtype=torch.float32, device=device).expand(h, w, 3).contiguous()
    if indices.numel() == 0:
        return img

    n_charts = int(chart_ids.max().item()) + 1 if chart_ids.numel() else 1
    colors = torch.from_numpy(_uv_palette(n_charts)).to(device=device, dtype=torch.float32)

    uv_px = uvs.clone()
    uv_px[:, 0] = uv_px[:, 0].clamp(0.0, 1.0) * (w - 1)
    uv_px[:, 1] = uv_px[:, 1].clamp(0.0, 1.0) * (h - 1)

    tri = uv_px[indices]
    x0 = tri[:, 0, 0]
    y0 = tri[:, 0, 1]
    x1 = tri[:, 1, 0]
    y1 = tri[:, 1, 1]
    x2 = tri[:, 2, 0]
    y2 = tri[:, 2, 1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    nondegen = denom.abs() > 1e-20

    xmin = torch.minimum(torch.minimum(x0, x1), x2).floor().clamp_(0, w - 1).long()
    xmax = torch.maximum(torch.maximum(x0, x1), x2).ceil().clamp_(0, w - 1).long()
    ymin = torch.minimum(torch.minimum(y0, y1), y2).floor().clamp_(0, h - 1).long()
    ymax = torch.maximum(torch.maximum(y0, y1), y2).ceil().clamp_(0, h - 1).long()

    # full point-in-tri over all pairs is O(H*W*F); tile and test only bbox-overlapping tris
    TILE = 64
    eps = 1e-6
    for ty in range(0, h, TILE):
        ty_end = min(ty + TILE, h)
        for tx in range(0, w, TILE):
            tx_end = min(tx + TILE, w)
            tri_mask = (nondegen & (xmin < tx_end) & (xmax >= tx)
                        & (ymin < ty_end) & (ymax >= ty))
            if not tri_mask.any():
                continue
            idx = torch.nonzero(tri_mask, as_tuple=True)[0]
            ys = torch.arange(ty, ty_end, dtype=torch.float32, device=device) + 0.5
            xs = torch.arange(tx, tx_end, dtype=torch.float32, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            sub_x0 = x0[idx][:, None, None]
            sub_y0 = y0[idx][:, None, None]
            sub_x1 = x1[idx][:, None, None]
            sub_y1 = y1[idx][:, None, None]
            sub_x2 = x2[idx][:, None, None]
            sub_y2 = y2[idx][:, None, None]
            sub_den = denom[idx][:, None, None]
            bx = ((sub_y1 - sub_y2) * (xx - sub_x2) + (sub_x2 - sub_x1) * (yy - sub_y2)) / sub_den
            by = ((sub_y2 - sub_y0) * (xx - sub_x2) + (sub_x0 - sub_x2) * (yy - sub_y2)) / sub_den
            bz = 1.0 - bx - by
            inside = (bx >= -eps) & (by >= -eps) & (bz >= -eps)
            if not inside.any():
                continue
            hit_any = inside.any(dim=0)
            best_tri = idx[inside.int().argmax(dim=0)]
            tile_color = colors[chart_ids[best_tri]]
            tile_img = img[ty:ty_end, tx:tx_end]
            tile_img[hit_any] = tile_color[hit_any]
            img[ty:ty_end, tx:tx_end] = tile_img

    # chart outlines: UV-space borders are open boundaries (edges with 1 incident face)
    _sk, _fid, lo, hi, first = _uv_sorted_edge_keys(indices_np)
    starts = np.nonzero(first)[0]
    counts = np.diff(np.append(starts, first.size))
    boundary = counts == 1
    uv_cpu = uv_px.cpu().numpy()
    px_xs, px_ys = [], []
    for a, b in zip(lo[starts[boundary]], hi[starts[boundary]]):
        p0 = uv_cpu[a]
        p1 = uv_cpu[b]
        steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1)
        if steps <= 1:
            continue
        ts = np.linspace(0.0, 1.0, steps)
        xs = (p0[0] + (p1[0] - p0[0]) * ts).astype(np.int32)
        ys = (p0[1] + (p1[1] - p0[1]) * ts).astype(np.int32)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        px_xs.append(xs[valid])
        px_ys.append(ys[valid])
    if px_xs:
        xs_all = torch.from_numpy(np.concatenate(px_xs)).to(device=device, dtype=torch.long)
        ys_all = torch.from_numpy(np.concatenate(px_ys)).to(device=device, dtype=torch.long)
        img[ys_all, xs_all] = torch.tensor(edge, dtype=torch.float32, device=device)

    return img


class RenderUVAtlas(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RenderUVAtlas",
            display_name="Render UV Atlas",
            category="latent/3d",
            description=("Renders a mesh's UV layout as an image (each chart a distinct color, "
                         "outlined at borders). Run UnwrapMesh first."),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("resolution", default=1024, min=64, max=4096, step=64),
            ],
            outputs=[IO.Image.Output("image")],
        )

    @classmethod
    def execute(cls, mesh, resolution):
        uvs_t = getattr(mesh, "uvs", None)
        if uvs_t is None:
            raise RuntimeError("mesh has no UVs to render. Run UnwrapMesh first.")
        uvs_np = uvs_t.detach().cpu().numpy()
        if uvs_np.ndim == 3:
            uvs_np = uvs_np[0]
        f = mesh.faces
        if torch.is_tensor(f):
            f = f.detach().cpu().numpy()
        if f.ndim == 3:
            f = f[0]
        f = np.ascontiguousarray(f, dtype=np.int64)
        uvs_np = np.ascontiguousarray(uvs_np, dtype=np.float32)
        device = comfy.model_management.get_torch_device()
        img = _uv_render_atlas(uvs_np, f, int(resolution), device)
        return IO.NodeOutput(img.detach().cpu().unsqueeze(0))


class FillHoles(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FillHoles",
            display_name="Fill Holes",
            category="latent/3d",
            description=(
                "Fills holes up to a max perimeter, preserving existing geometry/UVs (only patch "
                "tris added). GPU-vectorised with auto-corrected winding and loop-averaged centroid "
                "colors; CPU walker fallback on non-CUDA."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("max_perimeter", default=0.03, min=0.0, step=0.0001,
                               tooltip="Max hole perimeter to fill. 0 disables."),
                IO.Float.Input("weld_epsilon_rel", default=1e-5, min=0.0, step=1e-6,
                               tooltip="Pre-weld tolerance (fraction of bbox diagonal); boundary detection "
                                       "needs welded verts. 0 skips."),
                IO.Int.Input("max_verts", default=16, min=3, max=1024,
                             tooltip="Cap boundary verts per cycle; centroid-fan only works for small "
                                     "near-planar holes. Keep ≤16."),
                IO.Boolean.Input("fill_chains", default=False,
                                 tooltip="Also fill open chains (not just cycles). Noisy; OFF matches cumesh."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, max_perimeter, weld_epsilon_rel, max_verts, fill_chains):
        def _fn(v, f, c):
            if max_perimeter > 0:
                v, f, c = fill_holes_v2_fn(
                    v, f, max_perimeter=max_perimeter, colors=c,
                    weld_epsilon_rel=weld_epsilon_rel,
                    fill_chains=fill_chains,
                    max_verts=max_verts,
                )
            return v, f, c
        return _process_mesh_batch(mesh, _fn)


class WeldVertices(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="WeldVertices",
            display_name="Weld Vertices",
            category="latent/3d",
            description=(
                "Merge coincident vertices via L_inf grid quantization. Use when a mesh comes in "
                "unwelded (per-face verts, no shared edges) — pre-pass before FillHoles, "
                "DecimateMesh, or any topology-aware op. Colors averaged per cluster."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("epsilon_rel", default=1e-5, min=0.0, step=1e-6,
                               tooltip="Weld tolerance (fraction of bbox diagonal). 1e-5 for float dedup; "
                                       "1e-3 for visibly-close-but-distinct verts."),
                IO.Float.Input("epsilon_abs", default=0.0, min=0.0, step=1e-6,
                               tooltip="Absolute weld tolerance (overrides epsilon_rel when > 0)."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, epsilon_rel, epsilon_abs):
        eps = epsilon_abs if epsilon_abs > 0 else None
        def _fn(v, f, c):
            v, f, c, n = weld_vertices_fn(v, f, epsilon=eps, epsilon_rel=epsilon_rel, colors=c)
            if n > 0:
                logging.info(f"[WeldVertices] merged {n} verts ({v.shape[0]} remain)")
            return v, f, c
        return _process_mesh_batch(mesh, _fn)


def merge_meshes(meshes):
    """Concatenate Types.MESH list into one (B=1) mesh: cumulative face-index offset,
    missing uvs/colors padded (zeros/white), texture from the first input that has one
    (later dropped — single-primitive glb can't carry multiple atlases)."""
    if not meshes:
        raise ValueError("merge_meshes: need at least one mesh")

    def _b0(t):
        return t[0] if t.ndim == 3 else t

    any_uvs = any(getattr(m, "uvs", None) is not None for m in meshes)
    any_colors = any(getattr(m, "vertex_colors", None) is not None for m in meshes)

    verts_list, faces_list, uvs_list, colors_list = [], [], [], []
    texture = None
    offset = 0
    for m in meshes:
        # Coerce to CPU so CUDA-side (MoGe) meshes merge cleanly with our outputs.
        v = _b0(m.vertices).cpu()
        f = _b0(m.faces).cpu()
        verts_list.append(v)
        faces_list.append(f + offset)
        offset += v.shape[0]
        if any_uvs:
            mu = getattr(m, "uvs", None)
            uvs_list.append(_b0(mu).cpu() if mu is not None else v.new_zeros((v.shape[0], 2)))
        if any_colors:
            mc = getattr(m, "vertex_colors", None)
            if mc is not None:
                c = _b0(mc).cpu()
            else:
                c = v.new_ones((v.shape[0], 3))
            colors_list.append(c)
        mt = getattr(m, "texture", None)
        if mt is not None:
            if texture is None:
                texture = mt.cpu()
            else:
                logging.warning("MergeMeshes: dropping extra texture from input; only one texture is kept.")

    merged_verts = torch.cat(verts_list, dim=0).unsqueeze(0)
    merged_faces = torch.cat(faces_list, dim=0).unsqueeze(0)
    merged_uvs = torch.cat(uvs_list, dim=0).unsqueeze(0) if any_uvs else None
    merged_colors = torch.cat(colors_list, dim=0).unsqueeze(0) if any_colors else None

    return Types.MESH(
        vertices=merged_verts,
        faces=merged_faces,
        uvs=merged_uvs,
        vertex_colors=merged_colors,
        texture=texture,
    )


class MergeMeshes(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        autogrow_template = IO.Autogrow.TemplatePrefix(
            IO.Mesh.Input("mesh"), prefix="mesh", min=2, max=50,
        )
        return IO.Schema(
            node_id="MergeMeshes",
            display_name="Merge Meshes",
            category="latent/3d",
            description=(
                "Concatenate N meshes into one by offsetting face indices and stacking verts, "
                "faces, uvs, and colors."
            ),
            inputs=[
                IO.Autogrow.Input("meshes", template=autogrow_template),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, meshes: IO.Autogrow.Type) -> IO.NodeOutput:
        return IO.NodeOutput(merge_meshes(list(meshes.values())))


class PostProcessMeshExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            FillHoles,
            WeldVertices,
            DecimateMesh,
            RemeshMesh,
            UnwrapMesh,
            RenderUVAtlas,
            PaintMesh,
            BakeTextureFromVoxel,
            MeshTextureToImage,
            ApplyTextureToMesh,
            BakeNormalMapFromMesh,
            BakeAmbientOcclusion,
            SetMeshMaterial,
            MergeMeshes,
        ]


async def comfy_entrypoint() -> PostProcessMeshExtension:
    return PostProcessMeshExtension()

import torch
import numpy as np
import math
from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
import copy
import comfy.utils
import comfy.model_management
from comfy_extras.qem_decimate.qem_core import simplify as qem_decimate_simplify, QEMConfig
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
    # Voxel field may carry the full PBR set (base_color, metallic, roughness,
    # alpha); vertex colors only use base_color RGB.
    if v_colors.shape[-1] > 3:
        v_colors = v_colors[:, :3]

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


# =============================================================================
# Texture baking from sparse voxel volume.
#
# Pipeline: xatlas UV unwrap → OpenGL UV-space rasterize to position map →
# nearest-voxel color sample per texel → cv2.inpaint to fill UV seams →
# attach texture + UVs to the Mesh for SaveGLB to serialize.
#
# Uses comfy_extras.nodes_glsl.GLContext for OpenGL context (already handles
# GLFW / EGL / OSMesa backend selection); xatlas for UV parameterization.
# =============================================================================

_GL_COMPILE_PROGRAM_CACHE_KEY = "_bake_texture_program_cache"


def _gl_compile_program(gl, vert_src: str, frag_src: str):
    """Compile and link a minimal vert+frag GL program. Caller owns the GLuint
    and must glDeleteProgram when done."""
    def _check_shader(s, kind):
        if not gl.glGetShaderiv(s, gl.GL_COMPILE_STATUS):
            log = gl.glGetShaderInfoLog(s).decode(errors="replace")
            gl.glDeleteShader(s)
            raise RuntimeError(f"GL {kind} shader compile failed: {log}")

    vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
    gl.glShaderSource(vs, vert_src)
    gl.glCompileShader(vs)
    _check_shader(vs, "vertex")
    fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
    gl.glShaderSource(fs, frag_src)
    gl.glCompileShader(fs)
    _check_shader(fs, "fragment")
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        log = gl.glGetProgramInfoLog(prog).decode(errors="replace")
        gl.glDeleteProgram(prog)
        raise RuntimeError(f"GL program link failed: {log}")
    return prog


# Position-passthrough shader. Vertex maps UV → clip space; fragment outputs the
# interpolated world-space vertex position (with alpha=1 marking valid texels).
_BAKE_VERT_SRC = """
#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_uv;
out vec3 v_pos;
void main() {
    v_pos = a_pos;
    gl_Position = vec4(a_uv * 2.0 - 1.0, 0.0, 1.0);
}
"""

_BAKE_FRAG_SRC = """
#version 330 core
in vec3 v_pos;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_pos, 1.0);
}
"""


def _bake_position_map(verts_np, faces_np, uvs_np, texture_size):
    """Rasterize unwrapped mesh in UV space; return (position_map, mask).
    position_map: (H, W, 3) float32 — interpolated 3D position per texel.
    mask:         (H, W)    bool    — valid (covered) texels.

    Uses comfy_extras.nodes_glsl.GLContext, which lazily picks GLFW/EGL/OSMesa."""
    from comfy_extras.nodes_glsl import GLContext, _import_opengl

    GLContext()  # ensure backend is initialized + current
    gl = _import_opengl()

    # PyOpenGL's high-level wrappers for the buffer/draw/readback functions
    # store array refs in OpenGL.contextdata, which on EGL contexts triggers
    # "Attempt to retrieve context when no valid context". Use the raw C
    # entry points (OpenGL.raw.*) instead — they skip the bookkeeping.
    import ctypes as _ctypes
    from OpenGL.raw.GL.VERSION.GL_1_1 import (
        glReadPixels as _raw_glReadPixels,
        glTexImage2D as _raw_glTexImage2D,
        glDrawElements as _raw_glDrawElements,
    )
    from OpenGL.raw.GL.VERSION.GL_1_5 import glBufferData as _raw_glBufferData
    from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _raw_glVertexAttribPointer

    H = W = int(texture_size)
    fbo = color_tex = vbo = ibo = vao = prog = None
    try:
        # Interleaved [pos.x, pos.y, pos.z, uv.x, uv.y] per vertex (stride=20 bytes).
        verts32 = np.ascontiguousarray(verts_np, dtype=np.float32)
        uvs32 = np.ascontiguousarray(uvs_np, dtype=np.float32)
        faces32 = np.ascontiguousarray(faces_np, dtype=np.uint32)
        vbo_data = np.ascontiguousarray(np.concatenate([verts32, uvs32], axis=-1), dtype=np.float32)

        prog = _gl_compile_program(gl, _BAKE_VERT_SRC, _BAKE_FRAG_SRC)

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        _raw_glBufferData(gl.GL_ARRAY_BUFFER, int(vbo_data.nbytes),
                          vbo_data.ctypes.data_as(_ctypes.c_void_p), gl.GL_STATIC_DRAW)

        ibo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ibo)
        _raw_glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, int(faces32.nbytes),
                          faces32.ctypes.data_as(_ctypes.c_void_p), gl.GL_STATIC_DRAW)

        gl.glEnableVertexAttribArray(0)
        _raw_glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 20, _ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        _raw_glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 20, _ctypes.c_void_p(12))

        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        color_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        _raw_glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, W, H,
                          0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, color_tex, 0)
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"FBO incomplete (status=0x{status:x})")

        gl.glViewport(0, 0, W, H)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glUseProgram(prog)
        _raw_glDrawElements(gl.GL_TRIANGLES, int(faces32.size), gl.GL_UNSIGNED_INT, None)
        gl.glFinish()

        # Pre-allocate readback buffer and pass it as a pointer so PyOpenGL
        # doesn't try to allocate one through its array-handler machinery.
        arr = np.empty((H, W, 4), dtype=np.float32)
        _raw_glReadPixels(0, 0, W, H, gl.GL_RGBA, gl.GL_FLOAT,
                          arr.ctypes.data_as(_ctypes.c_void_p))
        # Do NOT flipud here. Our shader places UV(0,0) at FBO bottom-left
        # (clip(-1,-1)), and glReadPixels returns bottom-row-first, so arr[0]
        # already holds the UV v=0 data. glTF samples PNG with row 0 = upper-left
        # = UV v=0, so storing arr as-is gives a consistent mapping. Flipping
        # would invert V and make every sample come from the wrong row.
        position_map = arr[..., :3]
        mask = arr[..., 3] > 0.5
        return position_map, mask
    finally:
        if fbo is not None:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glDeleteFramebuffers(1, [fbo])
        if color_tex is not None:
            gl.glDeleteTextures(1, [color_tex])
        if vbo is not None:
            gl.glDeleteBuffers(1, [vbo])
        if ibo is not None:
            gl.glDeleteBuffers(1, [ibo])
        if vao is not None:
            gl.glBindVertexArray(0)
            gl.glDeleteVertexArrays(1, [vao])
        if prog is not None:
            gl.glUseProgram(0)
            gl.glDeleteProgram(prog)


def _trilinear_sample_sparse(positions, voxel_coords_np, color_np, resolution):
    """Normalized trilinear interpolation of a SPARSE voxel attribute field.

    The official o_voxel.to_glb trilinear-samples a *dense* attribute volume; here
    the field is sparse (only surface voxels carry values), so a plain trilinear
    would bleed zeros from empty cells. Instead we accumulate, per query, only the
    occupied corners among the 8 surrounding voxels and renormalize by their
    weights — i.e. trilinear over the occupied subset. Voxel centres sit at integer
    coords c with world position c/resolution - 0.5.

    Returns (vals [K, C] float64, ok [K] bool). `ok` is False where none of the 8
    corners is occupied (caller falls back to nearest there)."""
    R = int(resolution)
    origin = -0.5
    voxel_size = 1.0 / R
    # Cell-CENTER convention: voxel coord c sits at world origin + (c+0.5)*voxel_size,
    # matching the official flex_gemm grid_sample_3d (its trilinear weight centers
    # integer coord c at query c+0.5). The `- 0.5` puts integer gc on voxel centres
    # so the 8 trilinear corners bracket the query correctly. Omitting it samples
    # half a voxel toward the corner — colour bleed at boundaries / thin features.
    gc = (positions.astype(np.float64) - origin) / voxel_size - 0.5  # continuous voxel-index coords
    base = np.floor(gc).astype(np.int64)                       # [K,3] lower corner
    frac = gc - base                                           # [K,3] in [0,1)

    vc = voxel_coords_np.astype(np.int64)
    occ_keys = (vc[:, 0] * R + vc[:, 1]) * R + vc[:, 2]        # linear key per occupied voxel
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
                idx = order[ins]                              # original voxel index (garbage where !matched)
                w = np.where(matched, wx * wy * wz, 0.0)[:, None]
                acc += w * color_np[idx]                      # w=0 cancels the garbage rows
                wsum += w
    ok = wsum[:, 0] > 1e-8
    vals = np.zeros((K, C), dtype=np.float64)
    vals[ok] = acc[ok] / wsum[ok]
    return vals, ok


def _nearest_voxel_sample_gpu(positions, voxel_coords_np, color_np, resolution):
    """GPU nearest-occupied-voxel lookup for surface points. Voxels sit on a
    regular integer grid (coord c ↔ world c/R-0.5), so the nearest voxel to a
    query is round((p+0.5)*R) plus a 3³ neighbour check — an O(1)-per-query grid
    lookup (sorted-key binary search), ~10-30× faster than a cKDTree over millions
    of voxels and ~identical. Returns (vals [K,C] float32, found [K] bool); `found`
    is False for the rare query whose nearest occupied voxel is >1 cell away (the
    caller falls back to a cKDTree on just those)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = int(resolution)
    P = torch.from_numpy(np.ascontiguousarray(positions)).to(dev).float()
    VC = torch.from_numpy(np.ascontiguousarray(voxel_coords_np)).to(dev).long()
    col = torch.from_numpy(np.ascontiguousarray(color_np)).to(dev).float()
    M, K, C = VC.shape[0], P.shape[0], col.shape[1]
    key = (VC[:, 0] * R + VC[:, 1]) * R + VC[:, 2]
    skey, order = key.sort()

    def _search(idx, radius):
        """Nearest occupied voxel within ±radius cells, for query subset P[idx]."""
        Ps = P[idx]
        # Cell-CENTER convention: voxel c is centred at (c+0.5)/R - 0.5 in world,
        # so the coord nearest a point is round((p+0.5)*R - 0.5) (matches the
        # official grid_sample_3d). The distance test below uses the same centre.
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

    all_idx = torch.arange(K, device=dev)
    best_i = torch.zeros(K, dtype=torch.long, device=dev)
    found = torch.zeros(K, dtype=torch.bool, device=dev)
    # Pass 1: radius 1 (3³) over everything — catches ~all surface texels cheaply.
    bi1, fnd1 = _search(all_idx, 1)
    best_i[all_idx] = bi1
    found[all_idx] = fnd1
    # Pass 2: wider radius on ONLY the few misses (avoids ever building a cKDTree
    # over millions of voxels just for a handful of >1-cell-away points).
    miss = torch.nonzero(~found, as_tuple=True)[0]
    if miss.numel() > 0:
        bi2, fnd2 = _search(miss, 4)
        best_i[miss] = bi2
        found[miss] = fnd2
    vals = col[best_i]
    return vals.cpu().numpy(), found.cpu().numpy()


def _sample_voxel_attrs_per_texel(position_map, mask, voxel_coords, voxel_colors, resolution,
                                  mode="trilinear"):
    """For every masked texel, sample the voxel field and return ALL its attribute
    channels. Returns (H, W, C) float32 in [0, 1] where C is the voxel feature
    width (3 for plain color, 6 for full PBR).

    mode="trilinear" — normalized trilinear over occupied voxels (the default; matches
    the official o_voxel.to_glb path), with nearest fallback for texels whose 8
    surrounding voxels are all empty. This is the only mode the nodes expose now.
    mode="nearest"  — nearest-voxel; kept as an internal/dev lever (blocky)."""
    H, W, _ = position_map.shape
    color_np = voxel_colors.detach().cpu().numpy().astype(np.float32)
    C = color_np.shape[-1]
    out = np.zeros((H, W, C), dtype=np.float32)
    if not mask.any():
        return out

    origin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
    voxel_size = 1.0 / float(resolution)
    coords_np = voxel_coords.detach().cpu().numpy()
    # Cell-CENTER convention (+0.5 voxel), matching the official grid_sample_3d and
    # the _trilinear/_nearest paths above; this cKDTree only serves the rare
    # >cell-radius nearest fallback but must use the same world mapping.
    voxel_pos = (coords_np.astype(np.float32) + 0.5) * voxel_size + origin
    valid_positions = position_map[mask]

    def _nearest(query):
        # GPU grid lookup; cKDTree only for the rare >1-cell miss.
        vals, found = _nearest_voxel_sample_gpu(query, coords_np, color_np, resolution)
        if not found.all():
            tree = scipy.spatial.cKDTree(voxel_pos)
            _, nearest_idx = tree.query(query[~found], k=1, workers=-1)
            vals[~found] = color_np[nearest_idx]
        return vals

    if mode == "trilinear":
        vals, ok = _trilinear_sample_sparse(valid_positions, coords_np, color_np, resolution)
        if not ok.all():
            # Texels with no occupied neighbour fall back to nearest.
            vals[~ok] = _nearest(valid_positions[~ok])
        out[mask] = np.clip(vals, 0.0, 1.0).astype(np.float32)
    else:
        out[mask] = np.clip(_nearest(valid_positions), 0.0, 1.0)
    return out


def _closest_point_on_triangles(p, a, b, c):
    """Vectorized exact closest point on triangles (Ericson, Real-Time Collision
    Detection §5.1.5). p/a/b/c are [..., 3]; returns [..., 3]. Handles all
    vertex/edge/face Voronoi regions, applied highest-priority-last via where."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def u(x):  # broadcast a scalar-per-element weight to [...,1]
        return x.unsqueeze(-1)

    # face region (default)
    denom = 1.0 / (va + vb + vc).clamp_min(1e-20)
    v = vb * denom
    w = vc * denom
    res = a + ab * u(v) + ac * u(w)
    # edge BC
    den_bc = (d4 - d3) + (d5 - d6)
    w_bc = (d4 - d3) / den_bc.clamp_min(1e-20)
    res = torch.where(u((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)),
                      b + (c - b) * u(w_bc), res)
    # edge AC
    w_ac = d2 / (d2 - d6).clamp_min(1e-20)
    res = torch.where(u((vb <= 0) & (d2 >= 0) & (d6 <= 0)), a + ac * u(w_ac), res)
    # vertex C
    res = torch.where(u((d6 >= 0) & (d5 <= d6)), c, res)
    # edge AB
    v_ab = d1 / (d1 - d3).clamp_min(1e-20)
    res = torch.where(u((vc <= 0) & (d1 >= 0) & (d3 <= 0)), a + ab * u(v_ab), res)
    # vertex B
    res = torch.where(u((d3 >= 0) & (d4 <= d3)), b, res)
    # vertex A
    res = torch.where(u((d1 <= 0) & (d2 <= 0)), a, res)
    return res


def _msb_int64(x):
    """floor(log2(x)) elementwise for int64 x >= 1 (bit-search, no float)."""
    r = torch.zeros_like(x); xx = x.clone()
    for s in (32, 16, 8, 4, 2, 1):
        sh = xx >> s; m = sh > 0
        r = torch.where(m, r + s, r); xx = torch.where(m, sh, xx)
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
    """Linear BVH (Karras 2012) over triangle AABBs, pure torch, NO external deps.

    21-bit-per-axis Morton sort of triangle centroids -> parallel radix-tree
    construction -> bottom-up node AABBs. Internal nodes are indexed 0..T-2, leaves
    are encoded as LEAF+i (i in 0..T-1) where leaf i holds triangle `order[i]`.
    Returns a dict with node AABBs (nmin,nmax over 2T entries), child links
    (left,right), the leaf->triangle map `order`, LEAF offset and T.

    A real tree (not a uniform grid) is what makes the closest-point query prune
    empty space and dense clusters, so it stays fast on huge, non-uniform references
    where the grid's ring search blows up — i.e. the cuMesh BVH approach, in torch."""
    dev = tri.device; T = tri.shape[0]
    amin = tri.amin(1); amax = tri.amax(1); cent = (amin + amax) * 0.5
    lo = cent.amin(0); hi = cent.amax(0); span = (hi - lo).clamp_min(1e-12)
    q = (((cent - lo) / span).clamp(0, 1) * float((1 << 21) - 1)).long()
    morton = (_morton_expand21(q[:, 0]) << 2 | _morton_expand21(q[:, 1]) << 1 | _morton_expand21(q[:, 2])).long()
    order = torch.argsort(morton); msort = morton[order]

    # delta(i,j): length of the common prefix of the (morton, index) keys of leaves
    # i and j (index breaks ties so duplicate Morton codes still split); -1 if OOB.
    def delta(i, j):
        ok = (j >= 0) & (j < T); jj = j.clamp(0, T - 1)
        x = msort[i] ^ msort[jj]; same = x == 0
        cp = torch.where(same, torch.full_like(x, 63), 62 - _msb_int64(x.clamp_min(1)))
        xi = i ^ jj
        cpi = torch.where(xi == 0, torch.full_like(x, 32), 31 - _msb_int64(xi.clamp_min(1)))
        return torch.where(ok, cp + torch.where(same, cpi, torch.zeros_like(cp)), torch.full_like(x, -1))

    I = torch.arange(T - 1, device=dev)
    dplus = delta(I, I + 1); dminus = delta(I, I - 1)
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
    l = torch.zeros_like(I); t = lmax.clone()
    while True:
        t = t // 2
        if int(t.max()) == 0:
            break
        cond = delta(I, I + (l + t) * direction) > dmin
        l = torch.where(cond, l + t, l)
    j = I + l * direction
    first = torch.minimum(I, j); last = torch.maximum(I, j)
    # split position: binary search on delta within [first, last]
    dnode = delta(first, last)
    s = torch.zeros_like(I); div = torch.full_like(I, 2); rng = last - first
    while True:
        step = (rng + div - 1) // div
        cond = delta(first, (first + s + step).clamp(max=T - 1)) > dnode
        s = torch.where(cond, s + step, s)
        if int(step.max()) <= 1:
            cond1 = delta(first, (first + s + 1).clamp(max=T - 1)) > dnode
            s = torch.where(cond1, s + 1, s)
            break
        div = div * 2
    gamma = first + s; LEAF = T
    left = torch.where(gamma == first, LEAF + gamma, gamma)
    right = torch.where(gamma + 1 == last, LEAF + gamma + 1, gamma + 1)

    # node AABBs: leaves seeded, internal unioned bottom-up over a few passes (a
    # balanced tree settles in ~log2(T) passes; the cap is a safety bound).
    nmin = torch.empty((2 * T, 3), device=dev); nmax = torch.empty((2 * T, 3), device=dev)
    nmin[LEAF:] = amin[order]; nmax[LEAF:] = amax[order]
    setm = torch.zeros(2 * T, dtype=torch.bool, device=dev); setm[LEAF:] = True
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


def _closest_points_on_mesh_bvh(Q, tri, bvh, max_stack=64):
    """Exact closest surface point per query, via per-query stack traversal of the
    triangle BVH (nearest-child-first for tight pruning), pure torch. Returns [N,3].

    Each while-iteration advances all still-active queries by one node; the active
    set shrinks fast, so even a few thousand iterations are cheap big GPU kernels.
    `max_stack` bounds the per-query stack (= tree height); overflow is counted and
    warned (a handful of texels could be slightly off) rather than silently wrong."""
    dev = Q.device; N = Q.shape[0]
    LEAF = bvh['LEAF']; nmin = bvh['nmin']; nmax = bvh['nmax']
    left = bvh['left']; right = bvh['right']; order = bvh['order']
    stack = torch.full((N, max_stack), -1, dtype=torch.long, device=dev)
    sp = torch.ones(N, dtype=torch.long, device=dev); stack[:, 0] = 0
    best = torch.full((N,), 1e30, device=dev); bestp = Q.clone()
    active = torch.arange(N, device=dev); overflow = 0

    def aabb_d2(node, q):
        d = (nmin[node] - q).clamp_min(0) + (q - nmax[node]).clamp_min(0)
        return (d * d).sum(-1)

    while active.numel() > 0:
        a = active; qa = Q[a]
        node = stack[a, sp[a] - 1]; sp[a] = sp[a] - 1
        within = aabb_d2(node, qa) < best[a]
        isleaf = node >= LEAF
        lv = within & isleaf
        if bool(lv.any()):
            ga = a[lv]; tt = tri[order[node[lv] - LEAF]]
            cp = _closest_point_on_triangles(qa[lv], tt[:, 0], tt[:, 1], tt[:, 2])
            d2 = ((cp - qa[lv]) ** 2).sum(-1)
            upd = d2 < best[ga]; gu = ga[upd]; best[gu] = d2[upd]; bestp[gu] = cp[upd]
        iv = within & ~isleaf
        if bool(iv.any()):
            gi = a[iv]; qi = qa[iv]; lc = left[node[iv]]; rc = right[node[iv]]
            dl = aabb_d2(lc, qi); dr = aabb_d2(rc, qi)
            near = torch.where(dl <= dr, lc, rc); far = torch.where(dl <= dr, rc, lc)
            s0 = sp[gi]
            stack[gi, s0.clamp(max=max_stack - 1)] = far; sp[gi] = (s0 + 1).clamp(max=max_stack)
            s1 = sp[gi]; overflow += int((s1 >= max_stack).sum())
            stack[gi, s1.clamp(max=max_stack - 1)] = near; sp[gi] = (s1 + 1).clamp(max=max_stack)
        active = a[sp[a] > 0]
    if overflow:
        logging.warning(f"[back-project] BVH stack overflow on {overflow} pushes "
                        f"(max_stack={max_stack}); a few texels may be slightly off — "
                        f"raise max_stack if this is large.")
    return bestp


def _back_project_positions(position_map, mask, ref_v, ref_f):
    """Snap each covered texel's interpolated position onto the reference mesh's true
    surface, so the voxel field is sampled at full surface detail instead of along
    flat triangle chords (the cause of faceted/pixelized bakes on coarse meshes).
    Mirrors o_voxel.to_glb step 7c but with NO cumesh/scipy/trimesh dependency: a
    pure-torch linear BVH (`_build_triangle_bvh`) + exact closest-point traversal,
    the same approach as cuMesh's cuBVH. Returns a new position_map with the covered
    texels replaced."""
    valid = np.ascontiguousarray(position_map[mask].astype(np.float32))
    if valid.shape[0] == 0:
        return position_map

    import time as _time
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rv = ref_v.detach().to(dev).float()
    rf = ref_f.detach().to(dev).long()
    tri = rv[rf]
    Q = torch.from_numpy(valid).to(dev)

    _t = _time.perf_counter()
    bvh = _build_triangle_bvh(tri)
    _tb = _time.perf_counter()
    bp = _closest_points_on_mesh_bvh(Q, tri, bvh)
    logging.info(f"[back-project] BVH build {_tb - _t:.1f}s + traverse "
                 f"{_time.perf_counter() - _tb:.1f}s ({rf.shape[0]} ref tris, "
                 f"{valid.shape[0]} texels)")

    out = position_map.copy()
    out[mask] = bp.detach().cpu().numpy().astype(position_map.dtype)
    return out


def _jfa_fill_gpu(img01, mask):
    """Fill every uncovered texel with its nearest covered texel's value via GPU
    Jump Flooding (O(log n) passes) — a fast nearest-fill replacement for
    cv2.inpaint on UV seam/gutter filling. img01 [H,W,C] float, mask [H,W] bool
    (True = covered). Returns [H,W,C] float. ~6× faster than cv2 Telea per map."""
    if not mask.any():
        return img01
    dev = "cuda"
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
    """Fill the UV-gutter texels around covered charts so seam sampling doesn't
    pull in black. GPU Jump Flooding (nearest fill) when CUDA is available, else
    cv2 Telea inpaint. `inpaint_radius<=0` disables; the radius only affects the
    cv2 fallback (JFA fills every uncovered texel by nearest)."""
    if inpaint_radius <= 0:
        return img01
    if torch.cuda.is_available():
        return _jfa_fill_gpu(img01, mask)
    import cv2
    u8 = (img01 * 255.0).clip(0, 255).astype(np.uint8)
    u8 = cv2.inpaint(u8, ((~mask).astype(np.uint8)) * 255, int(inpaint_radius), cv2.INPAINT_TELEA)
    if u8.ndim == 2:
        u8 = u8[..., None]
    return u8.astype(np.float32) / 255.0


def bake_texture_from_voxel_fn(vertices, faces, voxel_coords, voxel_colors,
                               resolution, texture_size, inpaint_radius=3,
                               fast_unwrap=True, existing_uvs=None,
                               normalize_uvs=True, sample_mode="trilinear",
                               reference=None, pbar=None):
    """Bake a baseColor (+ optional metallicRoughness) texture for
    `vertices/faces`, rasterizing in UV space and nearest-voxel-sampling each
    texel from the provided sparse colored voxel volume.

    If `existing_uvs` (N, 2) is given, it is used directly and xatlas is
    skipped — bakes onto the mesh's current UV layout without re-unwrapping.
    Otherwise xatlas computes a fresh atlas (verts/faces may grow at seams).

    Returns (out_vertices, out_faces, out_uvs, out_texture, out_mr).

    `fast_unwrap=True` configures xatlas with permissive chart options so it
    finishes in a reasonable time on large meshes — at the cost of less even
    UV distribution. Set False to use xatlas defaults (slow on >100k faces).

    Progress: drives a local tqdm over its 5 stages (unwrap → rasterize →
    back-project → sample → finalize) and, if a comfy `pbar` (ProgressBar) is
    passed, ticks it once per stage too — so callers should size it as 5 per
    bake."""
    import time

    # 5-stage progress: tqdm (console) + optional comfy ProgressBar (UI). _tick is
    # called exactly once at each stage boundary, including no-op stages (e.g. no
    # back-projection), so the comfy pbar stays aligned at 5 ticks per bake.
    try:
        from tqdm import tqdm as _tqdm
        _tq = _tqdm(total=5, desc="Bake texture", leave=False)
    except Exception:
        _tq = None

    def _tick(name):
        if _tq is not None:
            _tq.set_postfix_str(name)
            _tq.update(1)
        if pbar is not None:
            pbar.update(1)

    v_np = vertices.detach().cpu().numpy().astype(np.float32)
    f_np = faces.detach().cpu().numpy().astype(np.uint32)
    fcount = int(f_np.shape[0])
    t0 = time.perf_counter()

    if existing_uvs is not None:
        # Bake onto the mesh's current UVs — no xatlas, no seam-splitting.
        uv_np = existing_uvs.detach().cpu().numpy().astype(np.float32)
        if uv_np.shape[0] != v_np.shape[0]:
            raise ValueError(
                f"BakeTextureFromVoxel: existing UVs ({uv_np.shape[0]}) must be 1:1 "
                f"with vertices ({v_np.shape[0]})."
            )
        uv_min = uv_np.min(axis=0)
        uv_max = uv_np.max(axis=0)
        oob = int(((uv_np < 0.0) | (uv_np > 1.0)).any(axis=1).sum())
        logging.info(f"[BakeTextureFromVoxel] using existing UVs: {v_np.shape[0]} verts, "
                     f"{fcount} faces (xatlas skipped)")
        logging.info(f"[BakeTextureFromVoxel]   UV range: u[{uv_min[0]:.3f},{uv_max[0]:.3f}] "
                     f"v[{uv_min[1]:.3f},{uv_max[1]:.3f}]  out-of-[0,1] verts: {oob}/{uv_np.shape[0]}")
        out_of_unit = (uv_min.min() < -1e-4) or (uv_max.max() > 1.0001)
        if normalize_uvs and out_of_unit:
            # Uniform fit of the UV bbox into [0,1] (preserves chart aspect ratios).
            # Handles packers that overflow the unit square slightly. NOT a UDIM
            # de-tiler — a true multi-tile layout would get squashed; warn if the
            # span is large enough to look like tiling.
            extent = float((uv_max - uv_min).max())
            span = max(float(uv_max[0] - uv_min[0]), float(uv_max[1] - uv_min[1]))
            if span > 1.5:
                logging.warning(
                    f"[BakeTextureFromVoxel] UV span {span:.2f} looks like a tiled/UDIM "
                    f"layout; uniform-fitting it into [0,1] will overlap tiles. "
                    f"Re-unwrap instead (use_existing_uvs=False)."
                )
            if extent > 0:
                uv_np = ((uv_np - uv_min) / extent).astype(np.float32)
                logging.info(f"[BakeTextureFromVoxel]   normalized UVs into [0,1] "
                             f"(uniform scale 1/{extent:.4f})")
        new_verts, new_faces, new_uvs = v_np, f_np, uv_np
    else:
        import xatlas
        if fcount > 300_000:
            logging.warning(
                f"[BakeTextureFromVoxel] mesh has {fcount} faces — xatlas chart "
                f"decomposition is CPU-bound and may take many minutes. Consider "
                f"decimating to under ~200k faces before baking."
            )
        logging.info(f"[BakeTextureFromVoxel] xatlas unwrap: {v_np.shape[0]} verts, {fcount} faces")
        if fast_unwrap and hasattr(xatlas, "Atlas"):
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            logging.info(f"[BakeTextureFromVoxel]   add_mesh: {time.perf_counter() - t0:.1f}s")
            gen_kwargs = {}
            applied = []
            # ChartOptions: looser growth → larger / fewer charts → faster.
            if hasattr(xatlas, "ChartOptions"):
                co = xatlas.ChartOptions()
                for attr, val in (
                    ("max_iterations", 1),
                    ("max_cost", 8.0),
                    ("normal_deviation_weight", 1.0),
                    ("roundness_weight", 0.0),
                    ("straightness_weight", 0.0),
                    ("normal_seam_weight", 1.0),
                    ("texture_seam_weight", 0.0),
                    ("use_input_mesh_uvs", False),
                ):
                    if hasattr(co, attr):
                        setattr(co, attr, val)
                        applied.append(f"chart.{attr}")
                gen_kwargs["chart_options"] = co
            # PackOptions.bruteForce defaults to True — tries many rotations per
            # chart and is the single biggest contributor to pack time on small
            # meshes. Off it loses ~5-15% packing efficiency but runs ~5× faster.
            if hasattr(xatlas, "PackOptions"):
                po = xatlas.PackOptions()
                for attr, val in (
                    ("bruteForce", False),
                    ("brute_force", False),   # snake_case alias on some builds
                    ("create_image", False),
                    ("createImage", False),
                    ("padding", 2),
                ):
                    if hasattr(po, attr):
                        setattr(po, attr, val)
                        applied.append(f"pack.{attr}")
                gen_kwargs["pack_options"] = po
            logging.info(f"[BakeTextureFromVoxel]   options applied: {applied}")
            tgen = time.perf_counter()
            try:
                atlas.generate(**gen_kwargs)
            except TypeError as e:
                logging.warning(f"[BakeTextureFromVoxel]   generate(**kwargs) rejected ({e}); retrying with defaults")
                atlas.generate()
            logging.info(f"[BakeTextureFromVoxel]   generate: {time.perf_counter() - tgen:.1f}s")
            tget = time.perf_counter()
            vmapping, indices, uvs = atlas[0]
            logging.info(f"[BakeTextureFromVoxel]   retrieve: {time.perf_counter() - tget:.1f}s")
        else:
            vmapping, indices, uvs = xatlas.parametrize(v_np, f_np)

        logging.info(f"[BakeTextureFromVoxel] xatlas total {time.perf_counter() - t0:.1f}s "
                     f"({vmapping.shape[0]} verts after seams)")
        new_verts = v_np[vmapping]
        new_faces = indices.astype(np.uint32)
        new_uvs = uvs.astype(np.float32)

    _tick("unwrap")

    t1 = time.perf_counter()
    position_map, mask = _bake_position_map(new_verts, new_faces, new_uvs, texture_size)
    logging.info(f"[BakeTextureFromVoxel] GL rasterize {texture_size}² in {time.perf_counter() - t1:.1f}s "
                 f"({int(mask.sum())}/{mask.size} texels covered)")
    _tick("rasterize")

    if reference is not None:
        # Back-project texel positions onto the original dense surface before
        # sampling — the o_voxel.to_glb step that makes the bake smooth on coarse
        # meshes (instead of sampling along flat triangle chords).
        tb = time.perf_counter()
        position_map = _back_project_positions(position_map, mask, reference[0], reference[1])
        logging.info(f"[BakeTextureFromVoxel] BVH back-project in {time.perf_counter() - tb:.1f}s")
    _tick("back-project")

    t2 = time.perf_counter()
    attrs = _sample_voxel_attrs_per_texel(
        position_map, mask, voxel_coords, voxel_colors, resolution, mode=sample_mode,
    )
    logging.info(f"[BakeTextureFromVoxel] voxel sample in {time.perf_counter() - t2:.1f}s "
                 f"({attrs.shape[-1]} channels)")
    _tick("sample")

    # Split into PBR maps. Layout matches upstream pbr_attr_layout:
    #   0:3 base_color, 3 metallic, 4 roughness, 5 alpha.
    C = attrs.shape[-1]
    base_color = attrs[..., 0:3]
    has_pbr = C >= 5
    metallic = attrs[..., 3:4] if C >= 4 else None
    roughness = attrs[..., 4:5] if C >= 5 else None
    # alpha channel exists at index 5 but we keep meshes opaque (upstream uses
    # alpha_mode=OPAQUE in the remesh path); plumb later if needed.

    t3 = time.perf_counter()
    base_color = _seam_fill(np.ascontiguousarray(base_color), mask, inpaint_radius)
    mr_image = None
    if has_pbr:
        # glTF metallicRoughness: R unused, G=roughness, B=metallic.
        mr = np.concatenate([np.zeros_like(roughness), roughness, metallic], axis=-1)
        mr_image = _seam_fill(np.ascontiguousarray(mr), mask, inpaint_radius)
    if inpaint_radius > 0:
        logging.info(f"[BakeTextureFromVoxel] inpaint in {time.perf_counter() - t3:.1f}s")

    device = vertices.device
    out_v = torch.from_numpy(new_verts).to(device=device, dtype=torch.float32)
    out_f = torch.from_numpy(new_faces.astype(np.int32)).to(device=device, dtype=torch.int32)
    out_uvs = torch.from_numpy(new_uvs).to(device=device, dtype=torch.float32)
    out_tex = torch.from_numpy(np.ascontiguousarray(base_color)).to(device=device, dtype=torch.float32)
    out_mr = (torch.from_numpy(np.ascontiguousarray(mr_image)).to(device=device, dtype=torch.float32)
              if mr_image is not None else None)
    _tick("finalize")
    if _tq is not None:
        _tq.close()
    return out_v, out_f, out_uvs, out_tex, out_mr


def _per_vertex_normals(verts_np, faces_np):
    """Area-weighted per-vertex normals (unit length) for a triangle mesh."""
    v = verts_np.astype(np.float64)
    f = faces_np.astype(np.int64)
    # Un-normalized face normals are area-weighted (cross product magnitude = 2*area),
    # so accumulating them onto vertices gives an area-weighted vertex normal.
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn = vn / np.clip(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12, None)
    return vn.astype(np.float32)


def bake_texture_multiview_fn(vertices, faces, voxel_coords, voxel_colors, resolution,
                              texture_size, views, blend_temperature=0.25,
                              inpaint_radius=3, fast_unwrap=True, existing_uvs=None,
                              normalize_uvs=True, sample_mode="trilinear"):
    """Bake a baseColor texture by projecting view photos onto the mesh.

    Reuses bake_texture_from_voxel_fn for the xatlas unwrap + the nearest-voxel
    fallback colour, then overlays photo colour on every covered+visible texel:
    each texel's world position/normal is projected into each view, occlusion is
    resolved with a texel z-buffer, and the views are blended weighted by how
    directly each camera faces the surface. Texels seen by no view keep the voxel
    colour. The seam inpaint runs last, over the composited result.

    `views`: list of dicts {image[H,W,3] in [0,1], azimuth_deg, transform_matrix[4,4],
    camera_angle_x (scalar tensor), image_resolution}. All Pixal3D views share the
    one front camera and differ only by azimuth.

    Returns (verts, faces, uvs, tex, mr) — same shape contract as
    bake_texture_from_voxel_fn, so the node attaches them identically."""
    from comfy.ldm.trellis2 import multiview_bake as mvbake

    # Voxel bake → unwrapped geometry + fallback colour (inpaint deferred to the end).
    out_v, out_f, out_uvs, voxel_tex, voxel_mr = bake_texture_from_voxel_fn(
        vertices, faces, voxel_coords, voxel_colors, resolution=resolution,
        texture_size=texture_size, inpaint_radius=0, fast_unwrap=fast_unwrap,
        existing_uvs=existing_uvs, normalize_uvs=normalize_uvs, sample_mode=sample_mode)

    v_np = out_v.detach().cpu().numpy().astype(np.float32)
    f_np = out_f.detach().cpu().numpy().astype(np.uint32)
    uv_np = out_uvs.detach().cpu().numpy().astype(np.float32)

    # Per-texel world position + normal (the GL baker outputs any per-vertex vec3).
    position_map, mask = _bake_position_map(v_np, f_np, uv_np, texture_size)
    normal_map, _ = _bake_position_map(_per_vertex_normals(v_np, f_np), f_np, uv_np, texture_size)

    device = out_v.device
    base = voxel_tex.detach().cpu().numpy().copy()
    if mask.any() and views:
        pos = torch.from_numpy(np.ascontiguousarray(position_map[mask])).to(device)
        nrm = torch.from_numpy(np.ascontiguousarray(normal_map[mask])).to(device)
        fallback = torch.from_numpy(np.ascontiguousarray(base[mask])).to(device)
        view_objs = [{
            "image": vw["image"].to(device),
            "azimuth_deg": vw["azimuth_deg"],
            "transform_matrix": vw["transform_matrix"].to(device),
            "camera_angle_x": vw["camera_angle_x"].to(device),
            "image_resolution": vw["image_resolution"],
        } for vw in views]
        rgb, _seen = mvbake.composite_views(pos, nrm, view_objs, fallback, blend_temperature)
        base[mask] = rgb.detach().cpu().numpy()

    base = _seam_fill(np.ascontiguousarray(base), mask, inpaint_radius)

    out_tex = torch.from_numpy(np.ascontiguousarray(base)).to(device=device, dtype=torch.float32)
    return out_v, out_f, out_uvs, out_tex, voxel_mr


class BakeTextureFromVoxel(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BakeTextureFromVoxel",
            display_name="Bake Texture From Voxel",
            category="latent/3d",
            description=(
                "Bakes PBR textures onto the mesh's existing UV layout by rasterizing it "
                "in UV space via OpenGL (ComfyUI's PyOpenGL backend) and trilinear-sampling "
                "the input sparse voxel volume. Does NOT unwrap — connect a UV unwrap node "
                "(e.g. Trellis2OfficialUnwrap or TorchXatlasUVWrap) upstream. Produces a "
                "baseColor texture, plus a metallicRoughness texture when the voxel field "
                "carries the full PBR set (6 channels). Returns a Mesh with `uvs`, `texture`, "
                "and `metallic_roughness` attached — SaveGLB serializes them as real "
                "baseColorTexture / metallicRoughnessTexture maps. UVs that spill outside "
                "[0,1] are uniformly fit back into the unit square."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Voxel.Input("voxel_colors"),
                IO.Int.Input("texture_size", default=1024, min=64, max=8192,
                             tooltip="Square texture resolution. Larger = sharper but slower / bigger file."),
                IO.Mesh.Input("reference_mesh", optional=True,
                              tooltip=(
                                  "Optional original (dense, pre-decimation) mesh. If connected, each "
                                  "texel is back-projected onto its true surface before sampling — the "
                                  "o_voxel.to_glb step that removes faceted/pixelized baking on coarse "
                                  "meshes. Pure scipy+torch, no extra deps.")),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, voxel_colors, texture_size, reference_mesh=None):
        # Seam-gutter inpaint radius is hardcoded to 3 (matches the official to_glb);
        # it's an on/off-grade knob — Telea fills the whole gutter regardless of value.
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
            out_verts, out_faces, out_uvs, out_tex, out_mr = [], [], [], [], []
            # 5 stage ticks per item (see bake_texture_from_voxel_fn); skipped items
            # tick all 5 so the bar stays aligned.
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
                bv, bf, bu, bt, bmr = bake_texture_from_voxel_fn(
                    v_i, f_i, item_coords, item_colors,
                    resolution=resolution, texture_size=texture_size,
                    inpaint_radius=inpaint_radius,
                    existing_uvs=ev_i, reference=ref_i, pbar=pbar,
                )
                out_verts.append(bv); out_faces.append(bf); out_uvs.append(bu)
                out_tex.append(bt); out_mr.append(bmr)
            if not out_verts:
                return IO.NodeOutput(mesh)
            # Local pack_variable_mesh_batch doesn't take uvs/texture; build the
            # packed mesh ourselves so we can attach both. UVs are 1:1 with verts.
            packed = pack_variable_mesh_batch(out_verts, out_faces)
            max_v = packed.vertices.shape[1]
            packed_uvs = out_uvs[0].new_zeros((len(out_uvs), max_v, 2))
            for i, u in enumerate(out_uvs):
                packed_uvs[i, :u.shape[0]] = u
            packed.uvs = packed_uvs
            packed.texture = torch.stack(out_tex, dim=0)
            if all(mr is not None for mr in out_mr):
                packed.metallic_roughness = torch.stack(out_mr, dim=0)
            return IO.NodeOutput(packed)

        # Single-item path.
        v0 = mesh.vertices.squeeze(0)
        f0 = mesh.faces.squeeze(0)
        ev0 = mesh_uvs.squeeze(0)
        ref0 = None
        if reference_mesh is not None:
            ref0 = (reference_mesh.vertices.squeeze(0), reference_mesh.faces.squeeze(0))
        pbar = comfy.utils.ProgressBar(5)  # 5 stage ticks (see bake_texture_from_voxel_fn)
        bv, bf, bu, bt, bmr = bake_texture_from_voxel_fn(
            v0, f0, coords, colors,
            resolution=resolution, texture_size=texture_size,
            inpaint_radius=inpaint_radius,
            existing_uvs=ev0, reference=ref0, pbar=pbar,
        )
        out_mesh = Types.MESH(
            vertices=bv.unsqueeze(0), faces=bf.unsqueeze(0),
            uvs=bu.unsqueeze(0), texture=bt.unsqueeze(0),
        )
        if bmr is not None:
            out_mesh.metallic_roughness = bmr.unsqueeze(0)
        return IO.NodeOutput(out_mesh)


class MeshTextureToImage(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshTextureToImage",
            display_name="Mesh Texture to Image",
            category="latent/3d",
            description=(
                "Extracts a mesh's baked textures as IMAGE outputs for preview/save. "
                "base_color is the baseColor map; metallic_roughness is the packed "
                "glTF MR map (R unused, G=roughness, B=metallic) — black if the mesh "
                "has no PBR texture."
            ),
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[
                IO.Image.Output(display_name="base_color"),
                IO.Image.Output(display_name="metallic_roughness"),
                IO.Image.Output(display_name="metallic"),
                IO.Image.Output(display_name="roughness"),
            ],
        )

    @classmethod
    def execute(cls, mesh):
        def _as_image(tex):
            # Mesh textures are (B, H, W, 3) float in [0, 1] — already IMAGE layout.
            if tex is None:
                return None
            t = tex.float().clamp(0.0, 1.0).cpu()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t

        base = _as_image(getattr(mesh, "texture", None))
        mr = _as_image(getattr(mesh, "metallic_roughness", None))

        if base is None:
            raise ValueError(
                "MeshTextureToImage: mesh has no baseColor texture. Run "
                "BakeTextureFromVoxel first (PaintMesh only sets vertex colors, not a texture)."
            )
        if mr is None:
            mr = torch.zeros_like(base)
        # Split the packed glTF MR map into single-channel grayscale previews:
        # G=roughness, B=metallic. Replicated to 3 channels so they display
        # as proper grayscale IMAGEs.
        metallic = mr[..., 2:3].expand(-1, -1, -1, 3).contiguous()
        roughness = mr[..., 1:2].expand(-1, -1, -1, 3).contiguous()
        return IO.NodeOutput(base, mr, metallic, roughness)


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


def _fill_holes_v2_diagnostic(verts, faces, max_perimeter):
    """Topology dump for debugging missed-hole cases. Logs edge count
    distribution, cycle count, and per-cycle (size, perimeter)."""
    device = verts.device
    V = verts.shape[0]
    F = faces.shape[0]
    e_all = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    e_sorted, _ = e_all.sort(dim=1)
    packed = e_sorted[:, 0].long() * V + e_sorted[:, 1].long()
    unique_packed, counts = torch.unique(packed, return_counts=True)

    n_boundary = int((counts == 1).sum().item())
    n_interior = int((counts == 2).sum().item())
    n_nonmanifold = int((counts >= 3).sum().item())
    nm_max = int(counts.max().item()) if counts.numel() > 0 else 0
    nm_share_breakdown = []
    if n_nonmanifold > 0:
        # Show top-5 non-manifold counts
        nm_counts = counts[counts >= 3]
        unique_nm, cnt_nm = torch.unique(nm_counts, return_counts=True)
        for c, n in zip(unique_nm.tolist(), cnt_nm.tolist()):
            nm_share_breakdown.append(f"{n} edges×{c}faces")

    logging.info(f"[FillHolesV2 diag] V={V} F={F} | "
                 f"boundary(cnt==1)={n_boundary}  interior(cnt==2)={n_interior}  "
                 f"non-manifold(cnt>=3)={n_nonmanifold} (max={nm_max})")
    if nm_share_breakdown:
        logging.info(f"[FillHolesV2 diag] non-manifold breakdown: {', '.join(nm_share_breakdown[:5])}")

    if n_boundary == 0:
        logging.info("[FillHolesV2 diag] no boundary edges → no cycles to fill")
        return

    # Walk components same as production path (bidir-prop, by-vertex count).
    boundary_packed = unique_packed[counts == 1]
    is_b = torch.isin(packed, boundary_packed)
    b_directed = e_all[is_b]
    src = b_directed[:, 0].long()
    tgt = b_directed[:, 1].long()

    labels = torch.arange(V, dtype=torch.long, device=device)
    for _ in range(64):
        edge_min = torch.minimum(labels[src], labels[tgt])
        new_labels = labels.clone()
        new_labels.scatter_reduce_(0, src, edge_min, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(0, tgt, edge_min, reduce="amin", include_self=True)
        new_labels = new_labels[new_labels]
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

    edge_component = labels[src]
    unique_components, component_idx = torch.unique(edge_component, return_inverse=True)
    L = unique_components.shape[0]
    edge_len = (verts[src] - verts[tgt]).norm(dim=-1)
    perim = torch.zeros(L, dtype=verts.dtype, device=device)
    perim.scatter_add_(0, component_idx, edge_len)
    edge_count = torch.bincount(component_idx, minlength=L)

    pair_keys = torch.unique(torch.cat([
        component_idx.long() * V + src,
        component_idx.long() * V + tgt,
    ]))
    pair_c = pair_keys // V
    vert_count = torch.bincount(pair_c, minlength=L)

    # Open chain = vert_count == edge_count + 1; closed cycle = vert_count == edge_count.
    is_chain = (vert_count == edge_count + 1)
    is_cycle = (vert_count == edge_count) & (vert_count > 0)
    is_other = ~(is_chain | is_cycle)

    # Match production filter (cycles only, default fill_chains=False, default max_verts=16).
    MAX_VERTS_DEFAULT = 16
    CENTROID_FAN_THRESHOLD = 8
    cycle_perim_ok = is_cycle & (perim < max_perimeter)
    cycle_size_ok = is_cycle & (vert_count >= 3) & (vert_count <= MAX_VERTS_DEFAULT)
    actually_kept = is_cycle & (vert_count >= 3) & (vert_count <= MAX_VERTS_DEFAULT) & (perim < max_perimeter)

    # Triangulation strategy split.
    vfan = actually_kept & (vert_count <= CENTROID_FAN_THRESHOLD)
    cfan = actually_kept & (vert_count > CENTROID_FAN_THRESHOLD)
    vfan_tris = int((vert_count[vfan] - 2).sum().item())   # N-2 tris per N-vert cycle
    cfan_tris = int(vert_count[cfan].sum().item())          # N tris per N-vert cycle
    cfan_new_verts = int(cfan.sum().item())                  # 1 centroid per centroid-fan component

    logging.info(f"[FillHolesV2 diag] components={L}  "
                 f"cycles={int(is_cycle.sum().item())}  chains={int(is_chain.sum().item())}  "
                 f"non-simple={int(is_other.sum().item())}")
    logging.info(f"[FillHolesV2 diag] (with default filter: cycles only, verts in [3,{MAX_VERTS_DEFAULT}], perim<{max_perimeter})")
    logging.info(f"[FillHolesV2 diag] actually kept={int(actually_kept.sum().item())}  "
                 f"cycle rejected by perim={int((is_cycle & ~cycle_perim_ok).sum().item())}  "
                 f"cycle rejected by verts={int((is_cycle & ~cycle_size_ok).sum().item())}")
    logging.info(f"[FillHolesV2 diag] vertex-fan: {int(vfan.sum().item())} cycles → {vfan_tris} tris (no new verts)")
    logging.info(f"[FillHolesV2 diag] centroid-fan: {int(cfan.sum().item())} cycles → {cfan_tris} tris + {cfan_new_verts} new verts")

    # Cycle vert-count distribution
    if is_cycle.any():
        from collections import Counter
        cycle_sizes = vert_count[is_cycle].tolist()
        sc = Counter(cycle_sizes)
        # show buckets: 3, 4, 5, 6, 7-10, 11-20, 21-50, 51+
        buckets = {"3":0,"4":0,"5":0,"6":0,"7-10":0,"11-20":0,"21-50":0,"51+":0}
        for s, n in sc.items():
            if s == 3: buckets["3"] += n
            elif s == 4: buckets["4"] += n
            elif s == 5: buckets["5"] += n
            elif s == 6: buckets["6"] += n
            elif s <= 10: buckets["7-10"] += n
            elif s <= 20: buckets["11-20"] += n
            elif s <= 50: buckets["21-50"] += n
            else: buckets["51+"] += n
        logging.info(f"[FillHolesV2 diag] cycle vert-count buckets: {buckets}")

    if is_cycle.any():
        cycle_perims = perim[is_cycle].cpu().tolist()
        head = sorted(cycle_perims, reverse=True)[:10]
        logging.info(f"[FillHolesV2 diag] top-10 cycle perimeters: "
                     f"{['%.4f' % p for p in head]}")


def _fill_holes_v2_gpu(verts, faces, max_perimeter, colors=None, fill_chains=False, max_verts=16):
    # Bidirectional connected-component labeling on the undirected boundary
    # subgraph. Fixes the original pointer-doubling bug where chains starting at
    # the lowest-id vertex never propagated their label backward, producing
    # spurious size-1/2 fragments (see qem_core._propagate_face_labels for
    # the same pattern applied to face adjacency).
    #
    # By default we only close TRUE cycles (each boundary vert has degree 2 in
    # the component). Chains tend to be either real surface boundaries or
    # fragments of a cycle broken by non-manifold edges — fan-filling them with
    # an arbitrary centroid produces overlapping/noisy geometry. Pass
    # fill_chains=True to opt in to chain closure.
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

    # Each boundary edge -> its component id. After bidir-prop, labels[src] == labels[tgt].
    edge_component = labels[src]
    unique_components, component_idx = torch.unique(edge_component, return_inverse=True)
    L = unique_components.shape[0]
    edge_count = torch.bincount(component_idx, minlength=L)
    edge_len = (verts[src] - verts[tgt]).norm(dim=-1)
    perim = torch.zeros(L, dtype=dtype, device=device)
    perim.scatter_add_(0, component_idx, edge_len)

    # Unique boundary-vertex set per component, to count verts and place centroids.
    # Pack (component, vert) into one key; dedup via torch.unique.
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

    # Identify closed cycles: every boundary vert in the component has exactly
    # degree 2 in the boundary subgraph. Equivalent: vert_count == edge_count.
    is_cycle_component = (vert_count == edge_count) & (vert_count > 0)

    # Filter: keep cycles (always) and chains (only if fill_chains=True), under perim limit.
    # Also cap vert_count: fan-from-centroid only triangulates correctly for small,
    # near-planar cycles. Larger holes produce overlapping geometry because the
    # centroid lands far from any surface.
    size_ok = (vert_count >= 3) & (vert_count <= max_verts) & (perim < max_perimeter)
    if fill_chains:
        keep_component = size_ok
    else:
        keep_component = is_cycle_component & size_ok
    if not keep_component.any():
        return verts, faces, colors, 0
    # Only centroid-fan components actually allocate a new vertex slot.
    # We pre-compute their indices here so the triangulation step below has them ready.
    use_centroid_per_comp_pre = keep_component & (vert_count > 8)  # threshold mirrored below
    centroid_long = use_centroid_per_comp_pre.long()
    centroid_idx_per_comp = V + centroid_long.cumsum(0) - 1

    # Triangulate kept components. Two strategies:
    #
    #   Vertex-fan (small cycles): pick one boundary vert as apex, connect to all
    #   non-adjacent boundary edges. N verts -> N-2 triangles, no inserted vertex.
    #   Apex stays on the existing surface, so no off-surface centroid → no overlap.
    #   Right choice for 6-vert dual-grid pinches around interior verts.
    #
    #   Centroid-fan (large cycles): insert a new vertex at the boundary centroid,
    #   fan from it. N triangles. Only safe if the cycle is close to planar.
    #   We fall back to centroid-fan above `centroid_fan_threshold` verts where
    #   vertex-fan would produce excessively skinny triangles.
    CENTROID_FAN_THRESHOLD = 8  # tune: lower = more vertex-fan, higher = more centroid-fan

    # Edge kept mask
    edge_kept = keep_component[component_idx]
    edge_comp = component_idx[edge_kept]
    kept_src = src[edge_kept]
    kept_tgt = tgt[edge_kept]

    # Per-edge tag: which strategy does its component use?
    use_centroid_per_comp = keep_component & (vert_count > CENTROID_FAN_THRESHOLD)
    use_centroid_per_edge = use_centroid_per_comp[edge_comp]

    fan_pieces = []
    # ---- Centroid-fan branch (only for components > threshold) ----
    if use_centroid_per_edge.any():
        kept_centroid = centroid_idx_per_comp[edge_comp[use_centroid_per_edge]]
        fan_pieces.append(torch.stack([
            kept_tgt[use_centroid_per_edge],
            kept_src[use_centroid_per_edge],
            kept_centroid,
        ], dim=1).to(faces.dtype))

    # ---- Vertex-fan branch (small cycles, no centroid inserted) ----
    use_vertex_fan_per_comp = keep_component & (vert_count <= CENTROID_FAN_THRESHOLD)
    if use_vertex_fan_per_comp.any():
        # For each vertex-fan component, pick the smallest-id boundary vert as apex
        # (deterministic & matches labels[*] = smallest after bidir-prop).
        # Then emit edges in component as fan tris (apex, src, tgt) EXCEPT for
        # the two edges incident to the apex (those would be degenerate).
        apex_per_comp = labels[unique_components]   # labels[u]==u after convergence
        # Edges that DON'T touch their component's apex
        vf_mask = use_vertex_fan_per_comp[edge_comp]
        if vf_mask.any():
            vf_src = kept_src[vf_mask]
            vf_tgt = kept_tgt[vf_mask]
            vf_comp = edge_comp[vf_mask]
            vf_apex = apex_per_comp[vf_comp]
            # Skip edges that include the apex (apex==src or apex==tgt → degenerate tri).
            non_apex = (vf_src != vf_apex) & (vf_tgt != vf_apex)
            fan_pieces.append(torch.stack([
                vf_tgt[non_apex], vf_src[non_apex], vf_apex[non_apex],
            ], dim=1).to(faces.dtype))

    fan_faces = torch.cat(fan_pieces, dim=0) if fan_pieces else torch.empty((0, 3), dtype=faces.dtype, device=device)

    # Open chains: close them with a closing triangle ONLY for centroid-fan
    # components (vertex-fan chains would need a different closing strategy).
    # In practice fill_chains=False makes this a no-op since chains aren't kept.
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
    """Merge coincident vertices via L_inf grid quantization.
    Ported from custom_nodes/qem_simplify/qem_core.py:_weld_vertices.

    `epsilon`: absolute L_inf distance; verts within this collapse together.
               If None, `epsilon_rel * bbox_diag` is used.
    Attributes (colors) are averaged across each cluster.

    Returns (new_verts, new_faces, new_colors, n_welded)."""
    if vertices.ndim == 3:
        v_out, f_out, c_out = [], [], [] if colors is not None else None
        total = 0
        for i in range(vertices.shape[0]):
            ci = colors[i] if colors is not None else None
            v_i, f_i, c_i, n = weld_vertices_fn(vertices[i], faces[i], epsilon, epsilon_rel, ci)
            v_out.append(v_i); f_out.append(f_i); total += n
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


def fill_holes_v2_fn(vertices, faces, max_perimeter=0.03, colors=None, weld_epsilon_rel=1e-5, fill_chains=False, max_verts=16, diagnostic=False):
    """Batched wrapper for the v2 GPU hole-filler. CPU tensors get pulled
    onto CUDA when available; otherwise fall back to the v1 (CPU walker) fn.

    Pre-welds vertices via `weld_vertices_fn(epsilon_rel=weld_epsilon_rel)` —
    boundary detection requires shared edges, which requires welded verts.
    Already-welded meshes early-exit cheaply. Set `weld_epsilon_rel=0` to skip."""
    if vertices.ndim == 3:
        v_list, f_list, c_list = [], [], [] if colors is not None else None
        pbar = comfy.utils.ProgressBar(vertices.shape[0])
        for i in range(vertices.shape[0]):
            ci = colors[i] if colors is not None else None
            v_i, f_i, c_i = fill_holes_v2_fn(vertices[i], faces[i], max_perimeter, ci, weld_epsilon_rel, fill_chains, max_verts, diagnostic)
            v_list.append(v_i); f_list.append(f_i)
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
    # Adaptive weld: a properly welded triangle surface has V/F ≈ 0.5 (closed)
    # to ~1.0 (with boundaries). V/F > 1 means most faces still share no verts
    # and hole-fill would emit one bogus closing tri per face. We double the
    # weld epsilon until V/F < WELDED_THRESHOLD or we hit WELD_CAP.
    if weld_epsilon_rel > 0:
        eps = float(weld_epsilon_rel)
        WELD_CAP = 1e-2          # ≈ 10 voxels at 1024-res — aggressive but bounded
        WELDED_THRESHOLD = 1.0   # V/F below this is "welded enough" for hole-fill
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
            logging.info(f"[FillHolesV2] pre-welded {total_welded} verts, V/F={ratio:.2f}{tag}")
        if ratio >= WELDED_THRESHOLD:
            logging.warning(
                f"[FillHolesV2] even at weld epsilon_rel={WELD_CAP} the mesh stays "
                f"unwelded (V/F={ratio:.2f}, want < {WELDED_THRESHOLD}). Source mesh has "
                f"duplicate verts at distances >{WELD_CAP}× bbox; fix upstream "
                f"(decimate node settings) or run WeldVertices manually with a larger epsilon."
            )
    # Diag runs AFTER welding so its topology numbers match what the filler sees.
    if diagnostic and vertices.device.type == "cuda" and faces.numel() > 0:
        _fill_holes_v2_diagnostic(vertices, faces, max_perimeter)
    if vertices.device.type == "cuda":
        out_v, out_f, out_c, _ = _fill_holes_v2_gpu(vertices, faces, max_perimeter, colors, fill_chains, max_verts)
        return out_v, out_f, out_c
    # CPU fallback: re-use the v1 walker (no attribute prop, but topologically equivalent
    # for manifold boundary; v2 GPU is the path that actually matters for pixal3d output).
    out_v, out_f = fill_holes_fn(vertices, faces, max_perimeter=max_perimeter)
    return out_v, out_f, colors


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


def fix_face_orientation(vertices, faces, reference_normals=None):
    num_faces = faces.shape[0]
    if num_faces == 0:
        return faces

    device = faces.device
    corrected = faces.clone()
    max_vert = vertices.shape[0]

    # Manifold edge adjacency: pair faces that share an edge (run length 2 after
    # canonicalizing + sorting edge hashes).
    idx = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.int64, device=device)
    edges = corrected[:, idx]  # (num_faces, 3, 2) directed
    edges_canon = torch.sort(edges, dim=2)[0].view(-1, 2)
    edge_hash = edges_canon[:, 0] * max_vert + edges_canon[:, 1]
    hash_sorted, sort_idx = torch.sort(edge_hash)
    start = torch.cat([torch.ones(1, dtype=torch.bool, device=device),
                       hash_sorted[1:] != hash_sorted[:-1]])
    unique_starts = torch.nonzero(start, as_tuple=True)[0]
    unique_ends = torch.cat([unique_starts[1:],
                             torch.tensor([hash_sorted.numel()], device=device)])
    manifold_starts = unique_starts[(unique_ends - unique_starts) == 2]

    if manifold_starts.numel() > 0:
        f_a = sort_idx[manifold_starts] // 3
        f_b = sort_idx[manifold_starts + 1] // 3
        le_a = sort_idx[manifold_starts] % 3
        le_b = sort_idx[manifold_starts + 1] % 3
        opposite = (edges[f_a, le_a] == edges[f_b, le_b].flip(dims=[1])).all(dim=1)

        # Connected components via scipy (fast C), replacing a per-face Python BFS.
        import scipy.sparse
        import scipy.sparse.csgraph
        fa_np = f_a.cpu().numpy(); fb_np = f_b.cpu().numpy()
        graph = scipy.sparse.coo_matrix(
            (np.ones(fa_np.shape[0] * 2, dtype=np.int8),
             (np.concatenate([fa_np, fb_np]), np.concatenate([fb_np, fa_np]))),
            shape=(num_faces, num_faces))
        num_components, comp = scipy.sparse.csgraph.connected_components(graph, directed=False)
        component_id = torch.from_numpy(comp.astype(np.int64)).to(device)

        # Within-component consistent winding. A QEM output from a consistently wound
        # source is already consistent (every shared edge is traversed oppositely) ->
        # no flips needed, the common fast path. Otherwise propagate a parity flip
        # across the dual graph by vectorized label relaxation (min-root carrying
        # parity), instead of the old per-face CPU BFS.
        if not bool(opposite.all()):
            nf = ~opposite
            src = torch.cat([f_a, f_b]); dst = torch.cat([f_b, f_a]); nfd = torch.cat([nf, nf])
            root = torch.arange(num_faces, device=device)
            par = torch.zeros(num_faces, dtype=torch.bool, device=device)
            for _ in range(num_faces + 8):  # breaks at graph diameter; cap is a backstop
                cand_root = root[src]; cand_par = par[src] ^ nfd
                new_root = root.clone()
                new_root.scatter_reduce_(0, dst, cand_root, reduce='amin', include_self=True)
                changed = new_root < root
                if not bool(changed.any()):
                    break
                apply = changed[dst] & (cand_root == new_root[dst])
                par[dst[apply]] = cand_par[apply]
                root = new_root
            if bool(par.any()):
                corrected[par] = corrected[par][:, [0, 2, 1]]
    else:
        component_id = torch.arange(num_faces, device=device)
        num_components = num_faces

    v0 = vertices[corrected[:, 0]]
    v1 = vertices[corrected[:, 1]]
    v2 = vertices[corrected[:, 2]]

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = face_normals / (torch.norm(face_normals, dim=-1, keepdim=True) + 1e-8)

    if reference_normals is not None:
        n0 = reference_normals[corrected[:, 0]]
        n1 = reference_normals[corrected[:, 1]]
        n2 = reference_normals[corrected[:, 2]]
        ref_normals = (n0 + n1 + n2) / 3.0
        ref_normals = ref_normals / (torch.norm(ref_normals, dim=-1, keepdim=True) + 1e-8)

        votes = (face_normals * ref_normals).sum(dim=-1)

        outward_votes_comp = torch.zeros(num_components, dtype=torch.int64, device=device)
        inward_votes_comp = torch.zeros(num_components, dtype=torch.int64, device=device)

        outward_votes_comp.scatter_add_(0, component_id, (votes > 0).to(torch.int64))
        inward_votes_comp.scatter_add_(0, component_id, (votes < 0).to(torch.int64))

        n_faces_comp_int = torch.zeros(num_components, dtype=torch.int64, device=device)
        n_faces_comp_int.scatter_add_(0, component_id, torch.ones(num_faces, dtype=torch.int64, device=device))

        thresholds = torch.maximum(torch.ones_like(n_faces_comp_int), n_faces_comp_int // 10)
        should_flip_comp = inward_votes_comp > outward_votes_comp + thresholds
    else:
        # Vectorized 3-Axis Extreme Majority Vote (Geometrically Infallible)
        face_centroids = (v0 + v1 + v2) / 3.0

        votes_by_axis = []
        for axis in range(3):
            coords = face_centroids[:, axis]

            # Double stable sort acts as a vectorized lexsort on (coords, component_id)
            sort_idx2 = torch.argsort(coords, stable=True)
            sort_idx2 = sort_idx2[torch.argsort(component_id[sort_idx2], stable=True)]

            # Find group boundaries to get the extreme outer face along this axis per component
            comp_id_sorted = component_id[sort_idx2]
            group_ends = torch.nonzero(comp_id_sorted[1:] != comp_id_sorted[:-1], as_tuple=True)[0]
            group_ends = torch.cat([group_ends, torch.tensor([len(comp_id_sorted) - 1], device=device)])

            extreme_face_indices = sort_idx2[group_ends]
            extreme_normals = face_normals[extreme_face_indices]

            # Normal's component along the respective axis should be positive
            votes_by_axis.append(extreme_normals[:, axis] > 0)

        stacked_votes = torch.stack(votes_by_axis, dim=0)
        should_flip_comp = stacked_votes.sum(dim=0) < 2  # False if at least 2 axes agree outward

    should_flip_face = should_flip_comp[component_id]
    if should_flip_face.any():
        corrected[should_flip_face] = corrected[should_flip_face][:, [0, 2, 1]]

    return corrected


def unweld_and_offset_mesh(vertices, faces, colors=None, z_offset=1e-4):
    is_batched = vertices.ndim == 3
    device = vertices.device

    if is_batched:
        B = vertices.shape[0]
        F = faces.shape[1]

        # 1. Advanced index broadcast to pull all faces in parallel without any Python loops
        batch_idx = torch.arange(B, device=device).view(-1, 1, 1)
        v_faces = vertices[batch_idx, faces]  # shape (B, F, 3, 3)

        v0, v1, v2 = v_faces[:, :, 0], v_faces[:, :, 1], v_faces[:, :, 2]

        # 2. Compute face normals
        fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
        fn = fn / (torch.norm(fn, dim=-1, keepdim=True) + 1e-8)

        # 3. Translate directly along the face normals in parallel
        offset_verts = v_faces + fn.unsqueeze(2) * z_offset
        out_v = offset_verts.reshape(B, -1, 3)

        # 4. Generate identical faces for all batches using constant expansion (O(1))
        f_single = torch.arange(F * 3, device=device).reshape(-1, 3)
        out_f = f_single.unsqueeze(0).expand(B, -1, -1)

        if colors is not None:
            c_faces = colors[batch_idx, faces]
            out_c = c_faces.reshape(B, -1, colors.shape[-1])
            return out_v, out_f, out_c
        return out_v, out_f

    # --- Unbatched (Single Mesh) ---
    v_faces = vertices[faces]  # shape (F, 3, 3)
    v0, v1, v2 = v_faces[:, 0], v_faces[:, 1], v_faces[:, 2]

    # Compute face normals
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / (torch.norm(fn, dim=-1, keepdim=True) + 1e-8)

    # Offset each face's private vertices along its face normal
    offset_verts = v_faces + fn.unsqueeze(1) * z_offset
    offset_verts = offset_verts.reshape(-1, 3)

    # Generate sequential face indices for the unwelded vertices
    f_unwelded = torch.arange(faces.shape[0] * 3, device=vertices.device).reshape(-1, 3)

    if colors is not None:
        c_faces = colors[faces]
        c_unwelded = c_faces.reshape(-1, colors.shape[-1])
        return offset_verts, f_unwelded, c_unwelded

    return offset_verts, f_unwelded, None

class DecimateMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        # placement_mode picks how the merged vertex is positioned, and which extra
        # quality knobs are surfaced (DynamicCombo: the qem sub-widgets only appear
        # when 'qem' is selected).
        placement_options = [
            IO.DynamicCombo.Option(key="midpoint", inputs=[]),
            IO.DynamicCombo.Option(key="qem", inputs=[
                IO.Float.Input("line_quadric_weight", default=0.0, min=0.0, max=100.0, step=0.1,
                               tooltip="Weight of the per-edge line quadric (squared distance to the edge "
                                       "line). Biases collapses to preserve sharp ridges/valleys. 0 = off."),
                IO.Float.Input("feature_edge_quadric_weight", default=0.0, min=0.0, max=1000.0, step=1.0,
                               tooltip="Extra quadric weight on dihedral feature edges (creases). Higher = "
                                       "more aggressively preserves hard edges. 0 = off."),
                IO.Float.Input("feature_edge_min_dihedral_deg", default=30.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Minimum dihedral angle (degrees) for an edge to count as a feature "
                                       "edge for feature_edge_quadric_weight."),
                IO.Boolean.Input("clamp_v_to_edge", default=True,
                                 tooltip="Project the QEM-optimal position onto the collapsed edge segment. "
                                         "Prevents inward-cascade drift on curved surfaces."),
            ]),
        ]
        return IO.Schema(
            node_id="DecimateMesh",
            display_name="Decimate Mesh",
            category="latent/3d",
            description=(
                "Simplifies a mesh to a target face count using QEM, on the active compute "
                "device. 'midpoint' placement uses the cumesh-faithful preset (best quality, "
                "preserves thin features / hair). 'qem' places each merged vertex at the QEM "
                "optimum and exposes line/feature-edge quadric controls. Output stays welded "
                "so it smooth-shades."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("target_face_count", default=200_000, min=0, max=50_000_000,
                             tooltip="Target maximum number of faces. Set to 0 to disable."),
                IO.DynamicCombo.Input("placement_mode", options=placement_options,
                                      display_name="placement_mode",
                                      tooltip="midpoint: cumesh-faithful preset (recommended). "
                                              "qem: QEM-optimal placement with line/feature-edge controls."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, mesh, target_face_count, placement_mode):
        mode = placement_mode.get("placement_mode", "midpoint")
        if mode == "qem":
            # QEM-optimum placement + ratio driver; everything else inherits the defaults.
            cfg = QEMConfig(
                placement_mode="qem",
                line_quadric_weight=float(placement_mode.get("line_quadric_weight", 0.0)),
                feature_edge_quadric_weight=float(placement_mode.get("feature_edge_quadric_weight", 0.0)),
                feature_edge_min_dihedral_deg=float(placement_mode.get("feature_edge_min_dihedral_deg", 30.0)),
                clamp_v_to_edge=bool(placement_mode.get("clamp_v_to_edge", True)),
            )
        else:
            cfg = QEMConfig()  # midpoint placement + threshold driver (the defaults)

        # ComfyUI passes meshes on CPU; the QEM is ~30x slower there. Run on the
        # selected compute device and return on the mesh's original device.
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

        # Send progress text to display the face reduction on the node
        if cls.hidden.unique_id:
            PromptServer.instance.send_progress_text(
                f"faces: {counts['in']} -> {counts['out']}", cls.hidden.unique_id)

        return result


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

class FillHolesV2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FillHolesV2",
            display_name="Fill Holes (v2)",
            category="latent/3d",
            description=(
                "GPU-vectorised hole-filling via directed-half-edge pointer-doubling. "
                "Drop-in alternative to FillHoles for comparison: same max_perimeter "
                "cutoff and fan-from-centroid triangulation, but no Python loop, "
                "auto-correct winding from face direction, and centroid colors are "
                "averaged from the loop instead of left zero."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("max_perimeter", default=0.03, min=0.0, step=0.0001,
                               tooltip="Maximum hole perimeter to fill. Set to 0 to disable."),
                IO.Float.Input("weld_epsilon_rel", default=1e-5, min=0.0, step=1e-6,
                               tooltip=(
                                   "Pre-weld tolerance as a fraction of the bbox diagonal. "
                                   "Boundary detection needs welded verts; already-welded meshes "
                                   "early-exit free. Set to 0 to skip pre-weld."
                               )),
                IO.Int.Input("max_verts", default=16, min=3, max=1024,
                             tooltip=(
                               "Cap the boundary-vertex count per cycle. Fan-from-centroid "
                               "only triangulates correctly for small, near-planar holes — "
                               "larger cycles produce overlapping geometry because the centroid "
                               "lands far from any surface. Keep low (≤16) for clean fills."
                             )),
                IO.Boolean.Input("fill_chains", default=False,
                                 tooltip=(
                                   "Also fill open boundary chains (not just closed cycles) "
                                   "by closing them with a fan + closing triangle. "
                                   "Often produces noisy/overlapping geometry on real meshes "
                                   "because chains are usually genuine surface boundaries or "
                                   "fragments of cycles broken by non-manifold edges. Leave OFF "
                                   "to match cumesh/upstream behavior."
                               )),
                IO.Boolean.Input("verbose", default=False,
                                 tooltip="Log topology diagnostics (edge counts, cycles found, reject reasons) for debugging."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh, max_perimeter, weld_epsilon_rel, max_verts, fill_chains, verbose):
        def _fn(v, f, c):
            if max_perimeter > 0:
                v, f, c = fill_holes_v2_fn(
                    v, f, max_perimeter=max_perimeter, colors=c,
                    weld_epsilon_rel=weld_epsilon_rel,
                    fill_chains=fill_chains,
                    max_verts=max_verts,
                    diagnostic=verbose,
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
                "Merge coincident vertices via L_inf grid quantization. Use when a "
                "mesh comes in unwelded (every face has its own 3 verts, no shared edges) "
                "— pre-pass before FillHoles, DecimateMesh, or any topology-aware op. "
                "Per-vertex colors are averaged across each merged cluster."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("epsilon_rel", default=1e-5, min=0.0, step=1e-6,
                               tooltip="Weld tolerance as a fraction of the bbox diagonal. "
                                       "1e-5 is enough for floating-point dedup; raise to "
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
    """Concatenate a list of Types.MESH into a single (B=1) mesh.

    Vertices, faces (with cumulative index offset), uvs, and vertex_colors are
    concatenated. If only some inputs carry uvs/vertex_colors, the missing sides
    are padded — zeros for uvs, white (1.0) for vertex_colors — so the merged
    primitive has a uniform attribute set. Texture is taken from the first input
    that has one; later textures are dropped with a warning (single-primitive glb
    can't carry multiple texture atlases without baking).
    """
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
        # Mesh tensors are normalized to CPU by our producer nodes; coerce defensively
        # so MoGe-side meshes (which may land on CUDA) merge cleanly with our outputs.
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
                "Concatenate N meshes into a single mesh by offsetting face indices "
                "and stacking vertices, faces, uvs, and vertex colors. Useful for combining a "
                "Pixal3D-reconstructed object (via Pixal3DAlignObject) with a MoGe scene "
                "background (via MoGePointMapToMesh) into one GLB."
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
            FillHolesV2,
            WeldVertices,
            DecimateMesh,
            PaintMesh,
            BakeTextureFromVoxel,
            MeshTextureToImage,
            MergeMeshes,
        ]


async def comfy_entrypoint() -> PostProcessMeshExtension:
    return PostProcessMeshExtension()

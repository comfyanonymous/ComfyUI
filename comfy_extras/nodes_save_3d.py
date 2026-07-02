"""Save-side 3D nodes: mesh packing/slicing helpers + GLB writer + SaveGLB node."""

import copy
import json
import logging
import math
import os
import struct
from io import BytesIO
from typing import TypedDict

import numpy as np
from PIL import Image
import torch
from typing_extensions import override

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, IO, Types


def pack_variable_mesh_batch(vertices, faces, colors=None, uvs=None, texture=None, unlit=False,
                             normals=None, metallic_roughness=None, tangents=None, normal_map=None,
                             occlusion_in_mr=False, material=None, emissive=None):
    # Pack per-item tensors into padded batches, stashing per-item lengths as runtime attrs.
    # colors/uvs/normals/tangents are 1:1 with vertices (padded to max_vertices); texture/
    # metallic_roughness/normal_map are (B,H,W,*) image stacks passed through unchanged.
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

    packed_colors = None
    if colors is not None:
        packed_colors = colors[0].new_zeros((batch_size, max_vertices, colors[0].shape[1]))
        for i, c in enumerate(colors):
            assert c.shape[0] == vertices[i].shape[0], (
                f"vertex_colors[{i}] has {c.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_colors[i, :c.shape[0]] = c

    packed_uvs = None
    if uvs is not None:
        packed_uvs = uvs[0].new_zeros((batch_size, max_vertices, uvs[0].shape[1]))
        for i, u in enumerate(uvs):
            assert u.shape[0] == vertices[i].shape[0], (
                f"uvs[{i}] has {u.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_uvs[i, :u.shape[0]] = u

    packed_normals = None
    if normals is not None:
        packed_normals = normals[0].new_zeros((batch_size, max_vertices, normals[0].shape[1]))
        for i, nrm in enumerate(normals):
            assert nrm.shape[0] == vertices[i].shape[0], (
                f"normals[{i}] has {nrm.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_normals[i, :nrm.shape[0]] = nrm

    packed_tangents = None
    if tangents is not None:
        packed_tangents = tangents[0].new_zeros((batch_size, max_vertices, tangents[0].shape[1]))
        for i, tn in enumerate(tangents):
            assert tn.shape[0] == vertices[i].shape[0], (
                f"tangents[{i}] has {tn.shape[0]} entries, expected {vertices[i].shape[0]} (1:1 with vertices)"
            )
            packed_tangents[i, :tn.shape[0]] = tn

    return Types.MESH(packed_vertices, packed_faces,
                      uvs=packed_uvs, vertex_colors=packed_colors, texture=texture,
                      metallic_roughness=metallic_roughness,
                      vertex_counts=vertex_counts, face_counts=face_counts, unlit=unlit,
                      normals=packed_normals, tangents=packed_tangents,
                      normal_map=normal_map, occlusion_in_mr=occlusion_in_mr,
                      material=material, emissive=emissive)


def get_mesh_batch_item(mesh, index):
    # Returns (vertices, faces, colors, uvs) for batch index, slicing to real lengths
    # if the mesh carries per-item counts (variable-size batch).
    v_colors = mesh.vertex_colors
    v_uvs = mesh.uvs
    v_normals = mesh.normals
    if mesh.vertex_counts is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = v_colors[index, :vertex_count] if v_colors is not None else None
        uvs = v_uvs[index, :vertex_count] if v_uvs is not None else None
        normals = v_normals[index, :vertex_count] if v_normals is not None else None
        return vertices, faces, colors, uvs, normals

    colors = v_colors[index] if v_colors is not None else None
    uvs = v_uvs[index] if v_uvs is not None else None
    normals = v_normals[index] if v_normals is not None else None
    return mesh.vertices[index], mesh.faces[index], colors, uvs, normals


def _smooth_vertex_normals(vertices_np, faces_np):
    """Area-weighted per-vertex normals (unit length), fully smooth — no vertex splitting.

    Un-normalized face normals (the raw cross product) have magnitude 2*area, so
    accumulating them onto their vertices yields an area-weighted average."""
    tris = vertices_np[faces_np]                                  # (M, 3, 3)
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals = np.zeros((vertices_np.shape[0], 3), dtype=np.float64)
    for k in range(3):
        np.add.at(normals, faces_np[:, k], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 1e-12, lens, 1.0)
    return normals.astype(np.float32)


def _compute_vertex_normals(vertices_np, faces_np, crease_angle=None):
    """Compute per-vertex normals, returning (vertices, faces_uint32, normals, remap).

    crease_angle is None (or >= 180) -> fully smooth normals; vertices/faces are
    returned unchanged and remap is None.

    Otherwise vertices are split along edges whose dihedral angle exceeds
    crease_angle (degrees) so hard creases stay sharp while smooth regions still
    interpolate. remap maps each output vertex back to its source index, so the
    caller can duplicate any per-vertex attributes (uvs / colors) to match."""
    faces_i = faces_np.astype(np.int64)
    if crease_angle is None or crease_angle >= 180.0:
        return (vertices_np, faces_i.astype(np.uint32),
                _smooth_vertex_normals(vertices_np, faces_i), None)

    M = faces_i.shape[0]
    tris = vertices_np[faces_i]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    areas = np.linalg.norm(face_n, axis=1, keepdims=True)
    face_unit = face_n / np.where(areas > 1e-12, areas, 1.0)
    cos_thresh = math.cos(math.radians(crease_angle))

    # Union faces that share an edge whose dihedral angle is below the crease
    # threshold; each connected component becomes one smoothing group.
    parent = list(range(M))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_faces = {}
    for fi in range(M):
        a, b, c = int(faces_i[fi, 0]), int(faces_i[fi, 1]), int(faces_i[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((u, v) if u < v else (v, u), []).append(fi)
    for fl in edge_faces.values():
        if len(fl) == 2 and float(np.dot(face_unit[fl[0]], face_unit[fl[1]])) >= cos_thresh:
            ra, rb = find(fl[0]), find(fl[1])
            if ra != rb:
                parent[ra] = rb

    # Emit one output vertex per (original vertex, smoothing group) pair.
    new_index = {}
    remap = []
    out_faces = np.empty((M, 3), dtype=np.int64)
    for fi in range(M):
        g = find(fi)
        for k in range(3):
            ov = int(faces_i[fi, k])
            key = (ov, g)
            ni = new_index.get(key)
            if ni is None:
                ni = len(remap)
                new_index[key] = ni
                remap.append(ov)
            out_faces[fi, k] = ni

    remap = np.asarray(remap, dtype=np.int64)
    normals = np.zeros((remap.shape[0], 3), dtype=np.float64)
    for k in range(3):
        np.add.at(normals, out_faces[:, k], face_n)
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lens > 1e-12, lens, 1.0)
    return (vertices_np[remap], out_faces.astype(np.uint32), normals.astype(np.float32), remap)


def save_glb(vertices, faces, filepath=None, metadata=None,
             uvs=None, vertex_colors=None, texture_image=None,
             metallic_roughness_image=None, unlit=False,
             normals=None, normal_map_image=None, tangents=None, occlusion_in_mr=False,
             material=None, emissive_image=None):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 3) - The face indices (triangle faces)
    filepath: str - Output filepath (should end with .glb). None returns the GLB bytes instead of writing.
    metadata: dict - Optional asset.extras metadata
    uvs: torch.Tensor of shape (N, 2) - Optional per-vertex texture coordinates
    vertex_colors: torch.Tensor of shape (N, 3) or (N, 4) - Optional per-vertex colors in [0, 1]
    texture_image: PIL.Image - Optional baseColor texture, embedded as PNG
    metallic_roughness_image: PIL.Image - Optional glTF metallicRoughness texture
        (R unused, G=roughness, B=metallic), embedded as PNG
    normals: torch.Tensor of shape (N, 3) - Optional per-vertex normals, written as the
        glTF NORMAL attribute. When omitted, NO normals are written and viewers fall back
        to flat (per-face) shading — use the MeshSmoothNormals node to generate them.
    normal_map_image: PIL.Image - Optional tangent-space normal map (glTF/OpenGL +Y),
        written as the material normalTexture. Needs TEXCOORD_0.
    tangents: torch.Tensor of shape (N, 4) - Optional per-vertex tangents (xyz + handedness w),
        written as the glTF TANGENT attribute. Without it viewers derive tangents in-shader.
    occlusion_in_mr: bool - When True, R of metallic_roughness_image holds AO (ORM packing) and
        occlusionTexture is pointed at that same image.
    material: dict - Optional scalar overrides from SetMeshMaterial (base_color_factor,
        metallic/roughness_factor with <0 = auto, emissive_factor/strength, normal_scale,
        occlusion_strength, double_sided).
    emissive_image: PIL.Image - Optional emissive (glow) texture, written as emissiveTexture.
    """

    # Convert tensors to numpy arrays
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    faces_signed = faces.cpu().numpy().astype(np.int64)
    uvs_np = uvs.cpu().numpy().astype(np.float32) if uvs is not None else None
    colors_np = vertex_colors.cpu().numpy().astype(np.float32) if vertex_colors is not None else None
    if colors_np is not None:
        colors_np = np.clip(colors_np, 0.0, 1.0)

    n_verts = vertices_np.shape[0]
    if n_verts == 0:
        raise ValueError("save_glb: vertices is empty")
    if faces_signed.size > 0:
        fmin = int(faces_signed.min())
        fmax = int(faces_signed.max())
        if fmin < 0 or fmax >= n_verts:
            raise ValueError(
                f"save_glb: face index out of range [0, {n_verts}): min={fmin}, max={fmax}"
            )
    if uvs_np is not None and uvs_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: uvs has {uvs_np.shape[0]} entries but vertex count is {n_verts}"
        )
    if colors_np is not None and colors_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: vertex_colors has {colors_np.shape[0]} entries but vertex count is {n_verts}"
        )

    normals_np = normals.cpu().numpy().astype(np.float32) if normals is not None else None
    if normals_np is not None and normals_np.shape[0] != n_verts:
        raise ValueError(
            f"save_glb: normals has {normals_np.shape[0]} entries but vertex count is {n_verts}"
        )
    tangents_np = tangents.cpu().numpy().astype(np.float32) if tangents is not None else None
    if tangents_np is not None and tangents_np.shape != (n_verts, 4):
        raise ValueError(
            f"save_glb: tangents must be (N, 4) with N={n_verts}, got {tuple(tangents_np.shape)}"
        )
    faces_np = faces_signed.astype(np.uint32)
    texture_png_bytes = None
    if texture_image is not None:
        buf = BytesIO()
        texture_image.save(buf, format="PNG")
        texture_png_bytes = buf.getvalue()
    mr_png_bytes = None
    if metallic_roughness_image is not None:
        buf = BytesIO()
        metallic_roughness_image.save(buf, format="PNG")
        mr_png_bytes = buf.getvalue()
    nm_png_bytes = None
    if normal_map_image is not None:
        buf = BytesIO()
        normal_map_image.save(buf, format="PNG")
        nm_png_bytes = buf.getvalue()
    em_png_bytes = None
    if emissive_image is not None:
        buf = BytesIO()
        emissive_image.save(buf, format="PNG")
        em_png_bytes = buf.getvalue()

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()
    uvs_buffer = uvs_np.tobytes() if uvs_np is not None else b""
    colors_buffer = colors_np.tobytes() if colors_np is not None else b""
    normals_buffer = normals_np.tobytes() if normals_np is not None else b""
    tangents_buffer = tangents_np.tobytes() if tangents_np is not None else b""
    texture_buffer = texture_png_bytes if texture_png_bytes is not None else b""
    mr_buffer = mr_png_bytes if mr_png_bytes is not None else b""
    nm_buffer = nm_png_bytes if nm_png_bytes is not None else b""
    em_buffer = em_png_bytes if em_png_bytes is not None else b""

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    # Blob order in one place; offsets accumulated in a pass so adding a buffer is one entry.
    _blobs = [
        ("vertices", vertices_buffer), ("indices", indices_buffer), ("uvs", uvs_buffer),
        ("colors", colors_buffer), ("normals", normals_buffer), ("tangents", tangents_buffer),
        ("texture", texture_buffer), ("mr", mr_buffer), ("nm", nm_buffer), ("em", em_buffer),
    ]
    byte_offset = {}
    acc = 0
    parts = []
    for name, b in _blobs:
        padded = pad_to_4_bytes(b)
        byte_offset[name] = acc
        acc += len(padded)
        parts.append(padded)
    buffer_data = b"".join(parts)

    vertices_byte_length = len(vertices_buffer)
    indices_byte_length = len(indices_buffer)
    vertices_byte_offset = byte_offset["vertices"]
    indices_byte_offset = byte_offset["indices"]
    uvs_byte_offset = byte_offset["uvs"]
    colors_byte_offset = byte_offset["colors"]
    normals_byte_offset = byte_offset["normals"]
    tangents_byte_offset = byte_offset["tangents"]
    texture_byte_offset = byte_offset["texture"]
    mr_byte_offset = byte_offset["mr"]
    nm_byte_offset = byte_offset["nm"]
    em_byte_offset = byte_offset["em"]

    buffer_views = [
        {
            "buffer": 0,
            "byteOffset": vertices_byte_offset,
            "byteLength": vertices_byte_length,
            "target": 34962  # ARRAY_BUFFER
        },
        {
            "buffer": 0,
            "byteOffset": indices_byte_offset,
            "byteLength": indices_byte_length,
            "target": 34963  # ELEMENT_ARRAY_BUFFER
        }
    ]
    accessors = [
        {
            "bufferView": 0,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(vertices_np),
            "type": "VEC3",
            "max": vertices_np.max(axis=0).tolist(),
            "min": vertices_np.min(axis=0).tolist()
        },
        {
            "bufferView": 1,
            "byteOffset": 0,
            "componentType": 5125,  # UNSIGNED_INT
            "count": faces_np.size,
            "type": "SCALAR"
        }
    ]
    primitive_attributes = {"POSITION": 0}

    if uvs_np is not None and len(uvs_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": uvs_byte_offset,
            "byteLength": len(uvs_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(uvs_np),
            "type": "VEC2",
        })
        primitive_attributes["TEXCOORD_0"] = accessor_idx

    if colors_np is not None and len(colors_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": colors_byte_offset,
            "byteLength": len(colors_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,
            "count": len(colors_np),
            "type": "VEC3" if colors_np.shape[1] == 3 else "VEC4",
        })
        primitive_attributes["COLOR_0"] = accessor_idx

    if normals_np is not None and len(normals_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": normals_byte_offset,
            "byteLength": len(normals_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(normals_np),
            "type": "VEC3",
        })
        primitive_attributes["NORMAL"] = accessor_idx

    if tangents_np is not None and len(tangents_np) > 0:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": tangents_byte_offset,
            "byteLength": len(tangents_buffer),
            "target": 34962
        })
        accessor_idx = len(accessors)
        accessors.append({
            "bufferView": len(buffer_views) - 1,
            "byteOffset": 0,
            "componentType": 5126,  # FLOAT
            "count": len(tangents_np),
            "type": "VEC4",  # xyz tangent + w handedness (glTF TANGENT)
        })
        primitive_attributes["TANGENT"] = accessor_idx

    primitive = {
        "attributes": primitive_attributes,
        "indices": 1,
        "mode": 4  # TRIANGLES
    }

    images = []
    textures = []
    samplers = []
    materials = []
    extensions_used = []

    def add_image_texture(png_byte_offset, png_byte_length):
        """Append an embedded PNG image + a texture referencing it; return the texture index."""
        buffer_views.append({"buffer": 0, "byteOffset": png_byte_offset, "byteLength": png_byte_length})
        images.append({"bufferView": len(buffer_views) - 1, "mimeType": "image/png"})
        if not samplers:
            samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071})
        textures.append({"source": len(images) - 1, "sampler": 0})
        return len(textures) - 1

    has_uv = "TEXCOORD_0" in primitive_attributes
    if unlit and texture_png_bytes is None:
        # Flat, light-independent shading (KHR_materials_unlit): COLOR_0 is shown as-is, matching how a
        # gaussian splat renders (emissive). Without this the viewer lights the mesh and washes the colours.
        if nm_png_bytes is not None or em_png_bytes is not None or occlusion_in_mr or material is not None:
            logging.warning(
                "save_glb: unlit material ignores normal/occlusion/emissive maps and SetMeshMaterial "
                "overrides — those are PBR-lit features. Disable unlit to export them.")
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        })
        extensions_used.append("KHR_materials_unlit")
        primitive["material"] = 0
    else:
        pbr = {
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "baseColorFactor": [0.22, 0.22, 0.22, 1.0],   # neutral-gray fallback for bare geometry only
        }
        if texture_png_bytes is not None and has_uv:
            pbr["baseColorTexture"] = {"index": add_image_texture(texture_byte_offset, len(texture_buffer)), "texCoord": 0}

        if (texture_png_bytes is not None and has_uv) or "COLOR_0" in primitive_attributes:
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            pbr["roughnessFactor"] = 1.0

        if mr_png_bytes is not None and has_uv:
            mr_texture_index = add_image_texture(mr_byte_offset, len(mr_buffer))
            pbr["metallicRoughnessTexture"] = {"index": mr_texture_index, "texCoord": 0}
            # When a metallicRoughness texture is present, the factors scale it; use 1.0
            # so the texture values pass through unchanged (glTF convention).
            pbr["metallicFactor"] = 1.0
            pbr["roughnessFactor"] = 1.0

        mat = material if isinstance(material, dict) else {}
        # Scalar overrides from SetMeshMaterial (factor < 0 means "leave auto").
        if mat.get("base_color_factor") is not None:
            pbr["baseColorFactor"] = [float(x) for x in mat["base_color_factor"]]
        if mat.get("metallic_factor", -1.0) >= 0.0:
            pbr["metallicFactor"] = float(mat["metallic_factor"])
        if mat.get("roughness_factor", -1.0) >= 0.0:
            pbr["roughnessFactor"] = float(mat["roughness_factor"])

        material = {
            "pbrMetallicRoughness": pbr,
            "doubleSided": bool(mat.get("double_sided", True)),
        }
        if occlusion_in_mr and mr_png_bytes is not None and has_uv:
            # ORM packing: occlusionTexture reuses the MR image (glTF reads its R channel).
            material["occlusionTexture"] = {"index": mr_texture_index, "texCoord": 0,
                                            "strength": float(mat.get("occlusion_strength", 1.0))}
        if nm_png_bytes is not None and has_uv:
            material["normalTexture"] = {"index": add_image_texture(nm_byte_offset, len(nm_buffer)),
                                         "texCoord": 0, "scale": float(mat.get("normal_scale", 1.0))}

        emissive_factor = [float(x) for x in mat.get("emissive_factor", [0.0, 0.0, 0.0])]
        emissive_strength = float(mat.get("emissive_strength", 1.0))
        has_em_tex = em_png_bytes is not None and has_uv
        if any(c > 0.0 for c in emissive_factor) or has_em_tex:
            # glTF multiplies emissiveFactor × texture, so a texture with no color would go black;
            # default the factor to white in that case.
            if has_em_tex and not any(c > 0.0 for c in emissive_factor):
                emissive_factor = [1.0, 1.0, 1.0]
            material["emissiveFactor"] = [min(1.0, c) for c in emissive_factor]
            if has_em_tex:
                material["emissiveTexture"] = {"index": add_image_texture(em_byte_offset, len(em_buffer)),
                                               "texCoord": 0}
            if emissive_strength != 1.0:
                material.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": emissive_strength}
                if "KHR_materials_emissive_strength" not in extensions_used:
                    extensions_used.append("KHR_materials_emissive_strength")

        materials.append(material)
        primitive["material"] = 0

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI"},
        "buffers": [{"byteLength": len(buffer_data)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    if images:
        gltf["images"] = images
    if samplers:
        gltf["samplers"] = samplers
    if textures:
        gltf["textures"] = textures
    if materials:
        gltf["materials"] = materials
    if extensions_used:
        gltf["extensionsUsed"] = extensions_used

    if metadata:
        gltf["asset"]["extras"] = metadata

    # Convert the JSON to bytes
    gltf_json = json.dumps(gltf).encode('utf8')

    def pad_json_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b' ' * padding_length

    gltf_json_padded = pad_json_to_4_bytes(gltf_json)

    # Create the GLB header (a 4-byte ASCII magic identifier glTF)
    glb_header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(gltf_json_padded) + 8 + len(buffer_data))

    # Create JSON chunk header (chunk type 0)
    json_chunk_header = struct.pack('<II', len(gltf_json_padded), 0x4E4F534A)  # "JSON" in little endian

    # Create BIN chunk header (chunk type 1)
    bin_chunk_header = struct.pack('<II', len(buffer_data), 0x004E4942)  # "BIN\0" in little endian

    glb = b"".join([glb_header, json_chunk_header, gltf_json_padded, bin_chunk_header, buffer_data])
    if filepath is None:
        return glb                       # in-memory GLB bytes (e.g. for a File3D object)
    with open(filepath, 'wb') as f:
        f.write(glb)
    return filepath


def mesh_item_to_glb_bytes(mesh, index, metadata=None):
    """Serialize one batch item of a MESH to in-memory GLB bytes, carrying every PBR attribute
    (uvs, colors, normals, texture, ORM/occlusion, normal map + tangents, emissive, material).
    Returns None for an empty item. Shared by SaveGLB (per item) and MeshToFile3D."""
    vertices_i, faces_i, v_colors, uvs_i, normals_i = get_mesh_batch_item(mesh, index)
    if vertices_i.shape[0] == 0 or faces_i.shape[0] == 0:
        return None

    def _img(attr):
        t = getattr(mesh, attr, None)
        if t is None:
            return None
        a = (t[index].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
        assert a.ndim == 3 and a.shape[-1] == 3, f"{attr} must be (B, H, W, 3), got {tuple(t.shape)}"
        return Image.fromarray(a, mode="RGB")

    tangents_b = mesh.tangents
    tangents_i = tangents_b[index, :vertices_i.shape[0]] if tangents_b is not None else None
    return save_glb(
        vertices_i, faces_i, None, metadata,
        uvs=uvs_i,
        vertex_colors=v_colors,
        texture_image=_img("texture"),
        metallic_roughness_image=_img("metallic_roughness"),
        unlit=mesh.unlit,
        normals=normals_i,
        normal_map_image=_img("normal_map"),
        tangents=tangents_i,
        occlusion_in_mr=mesh.occlusion_in_mr,
        material=mesh.material,
        emissive_image=_img("emissive"),
    )


class SaveGLB(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveGLB",
            display_name="Save 3D Model",
            search_aliases=["export 3d model", "save mesh"],
            category="3d",
            essentials_category="Basics",
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    IO.Mesh.Input("mesh"),
                    types=[
                        IO.File3DGLB,
                        IO.File3DGLTF,
                        IO.File3DOBJ,
                        IO.File3DFBX,
                        IO.File3DSTL,
                        IO.File3DUSDZ,
                        IO.File3DPLY,
                        IO.File3DSPLAT,
                        IO.File3DSPZ,
                        IO.File3DKSPLAT,
                        IO.File3DSplatAny,
                        IO.File3DPointCloudAny,
                        IO.File3DAny,
                    ],
                    tooltip="Mesh or 3D file to save",
                ),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo]
        )

    @classmethod
    def execute(cls, mesh: Types.MESH | Types.File3D, filename_prefix: str) -> IO.NodeOutput:
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results = []

        metadata = {}
        if not args.disable_metadata:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt)
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        if isinstance(mesh, Types.File3D):
            # Handle File3D input - save BytesIO data to output folder
            ext = mesh.format or "glb"
            f = f"{filename}_{counter:05}_.{ext}"
            mesh.save_to(os.path.join(full_output_folder, f))
            results.append({
                "filename": f,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1
        else:
            # Handle Mesh input - save vertices and faces as GLB; carry optional UVs / colors / texture.
            for i in range(mesh.vertices.shape[0]):
                glb = mesh_item_to_glb_bytes(mesh, i, metadata)
                if glb is None:
                    logging.warning(f"SaveGLB: skipping empty mesh at batch index {i}")
                    continue
                f = f"{filename}_{counter:05}_.glb"
                with open(os.path.join(full_output_folder, f), "wb") as fh:
                    fh.write(glb)
                results.append({
                    "filename": f,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
        return IO.NodeOutput(ui={"3d": results})


class MeshToFile3D(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshToFile3D",
            display_name="Create 3D File (from Mesh)",
            search_aliases=["mesh to glb", "mesh to file", "export mesh"],
            category="3d",
            description="Serialize a mesh to a GLB File3D object for Save / Preview 3D nodes, "
                        "carrying its UVs, colors, normals, texture, normal/occlusion/emissive "
                        "maps and material. Supports one item per batch only.",
            inputs=[IO.Mesh.Input("mesh")],
            outputs=[IO.File3DGLB.Output(display_name="model_3d")],
        )

    @classmethod
    def execute(cls, mesh) -> IO.NodeOutput:
        if mesh.vertices.shape[0] > 1:
            logging.warning("MeshToFile3D supports one item per batch only. Got %d; using first.",
                            mesh.vertices.shape[0])
        glb = mesh_item_to_glb_bytes(mesh, 0)
        if glb is None:
            raise ValueError("MeshToFile3D: mesh is empty (no vertices/faces).")
        return IO.NodeOutput(Types.File3D(BytesIO(glb), file_format="glb"))


class RotateMesh(IO.ComfyNode):
    class ModeValues(TypedDict, total=False):
        mode: str
        angle_x: float
        angle_y: float
        angle_z: float
        qw: float
        qx: float
        qy: float
        qz: float

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RotateMesh",
            display_name="Rotate Mesh",
            category="3d",
            description=(
                "Rotate a mesh. Euler XYZ applies X then Y then Z about the world axes (degrees). "
                "Quaternion is (w, x, y, z), auto-normalized."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.DynamicCombo.Input(
                    "mode",
                    options=[
                        IO.DynamicCombo.Option("euler_xyz", [
                            IO.Float.Input("angle_x", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the X axis in degrees."),
                            IO.Float.Input("angle_y", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the Y axis in degrees."),
                            IO.Float.Input("angle_z", default=0.0, min=-360.0, max=360.0, step=0.1,
                                           tooltip="Rotation around the Z axis in degrees."),
                        ]),
                        IO.DynamicCombo.Option("quaternion", [
                            IO.Float.Input("qw", default=1.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qx", default=0.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qy", default=0.0, min=-1.0, max=1.0, step=0.001),
                            IO.Float.Input("qz", default=0.0, min=-1.0, max=1.0, step=0.001),
                        ]),
                    ],
                ),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh: Types.MESH, mode: ModeValues) -> IO.NodeOutput:
        mode_name = mode["mode"]
        if mode_name == "euler_xyz":
            ax = math.radians(mode["angle_x"])
            ay = math.radians(mode["angle_y"])
            az = math.radians(mode["angle_z"])
            if ax == 0.0 and ay == 0.0 and az == 0.0:
                return IO.NodeOutput(mesh)
            cx, sx = math.cos(ax), math.sin(ax)
            cy, sy = math.cos(ay), math.sin(ay)
            cz, sz = math.cos(az), math.sin(az)
            R_rows = [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy,     sx * cy,                cx * cy],
            ]
        elif mode_name == "quaternion":
            qw, qx, qy, qz = mode["qw"], mode["qx"], mode["qy"], mode["qz"]
            n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if n < 1e-8:
                raise ValueError("RotateMesh: quaternion has zero magnitude")
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
            if qw == 1.0 and qx == 0.0 and qy == 0.0 and qz == 0.0:
                return IO.NodeOutput(mesh)
            R_rows = [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
            ]
        else:
            raise ValueError(f"RotateMesh: unknown mode {mode_name!r}")

        def rotate(v: torch.Tensor) -> torch.Tensor:
            R = torch.tensor(R_rows, device=v.device, dtype=v.dtype)
            return v @ R.T

        out = copy.copy(mesh)
        if isinstance(mesh.vertices, list):
            out.vertices = [rotate(v) for v in mesh.vertices]
        else:
            out.vertices = rotate(mesh.vertices)
        # Normals are directions; rotate them too (R is orthogonal) so they stay valid.
        nrm = mesh.normals
        if nrm is not None:
            out.normals = [rotate(n) for n in nrm] if isinstance(nrm, list) else rotate(nrm)
        return IO.NodeOutput(out)


class MeshSmoothNormals(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="MeshSmoothNormals",
            display_name="Smooth Mesh Normals",
            category="3d",
            description=(
                "Compute smooth per-vertex normals and attach them to the mesh. Meshes "
                "without normals are shaded flat (per-face) by glTF viewers; this makes "
                "them shade smoothly. With crease_angle below 180, edges sharper than the "
                "threshold are kept hard by splitting vertices along them."
            ),
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Float.Input("crease_angle", default=180.0, min=0.0, max=180.0, step=1.0,
                               tooltip="Edges whose dihedral angle exceeds this (degrees) stay "
                                       "hard (vertices are split). 180 = fully smooth; lower "
                                       "preserves sharp edges (e.g. ~30-60 for hard-surface)."),
            ],
            outputs=[IO.Mesh.Output("mesh")],
        )

    @classmethod
    def execute(cls, mesh: Types.MESH, crease_angle: float) -> IO.NodeOutput:
        crease = None if crease_angle >= 180.0 else float(crease_angle)
        batch_size = mesh.vertices.shape[0]

        if crease is None:
            # Fully smooth: topology is unchanged, so just attach a normals tensor that
            # matches the existing (possibly zero-padded) vertex layout and keep all fields.
            normals_padded = torch.zeros_like(mesh.vertices)
            for i in range(batch_size):
                v_i, f_i, _, _, _ = get_mesh_batch_item(mesh, i)
                if v_i.shape[0] == 0 or f_i.shape[0] == 0:
                    continue
                n_i = _smooth_vertex_normals(v_i.cpu().numpy().astype(np.float32),
                                             f_i.cpu().numpy().astype(np.int64))
                normals_padded[i, :n_i.shape[0]] = torch.from_numpy(n_i).to(mesh.vertices)
            out = copy.copy(mesh)
            out.normals = normals_padded
            return IO.NodeOutput(out)

        # Crease split changes per-item vertex counts -> rebuild as a variable-size batch.
        tangents_b = mesh.tangents
        v_list, f_list, n_list = [], [], []
        c_list = [] if mesh.vertex_colors is not None else None
        u_list = [] if mesh.uvs is not None else None
        t_list = [] if tangents_b is not None else None
        for i in range(batch_size):
            v_i, f_i, c_i, u_i, _ = get_mesh_batch_item(mesh, i)
            if v_i.shape[0] == 0 or f_i.shape[0] == 0:
                continue
            dev = v_i.device
            vo, fo, no, remap = _compute_vertex_normals(
                v_i.cpu().numpy().astype(np.float32),
                f_i.cpu().numpy().astype(np.int64), crease)
            remap_t = torch.from_numpy(remap)
            v_list.append(torch.from_numpy(vo).to(dev, mesh.vertices.dtype))
            f_list.append(torch.from_numpy(fo.astype(np.int64)).to(dev, mesh.faces.dtype))
            n_list.append(torch.from_numpy(no).to(dev, mesh.vertices.dtype))
            if c_list is not None:
                c_list.append(c_i[remap_t.to(c_i.device)])
            if u_list is not None:
                u_list.append(u_i[remap_t.to(u_i.device)])
            if t_list is not None:
                # Remap (not recompute) so TANGENT keeps the baked basis; split verts copy theirs.
                t_i = tangents_b[i, :v_i.shape[0]]
                t_list.append(t_i[remap_t.to(t_i.device)])
        if not v_list:
            return IO.NodeOutput(mesh)
        out = pack_variable_mesh_batch(
            v_list, f_list, colors=c_list, uvs=u_list,
            texture=mesh.texture, unlit=mesh.unlit,
            normals=n_list, metallic_roughness=mesh.metallic_roughness,
            tangents=t_list, normal_map=mesh.normal_map,
            occlusion_in_mr=mesh.occlusion_in_mr,
            material=mesh.material, emissive=mesh.emissive)
        return IO.NodeOutput(out)


class Save3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SaveGLB, MeshToFile3D, RotateMesh, MeshSmoothNormals]


async def comfy_entrypoint() -> Save3DExtension:
    return Save3DExtension()

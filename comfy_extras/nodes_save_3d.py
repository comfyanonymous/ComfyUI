"""Save-side 3D nodes: mesh packing/slicing helpers + GLB writer + SaveGLB
node, plus pose-data exporters (BuildPoseGLB / SavePoseBVH) that accept either
SAM3DBody Predict's MHR pose data or external-rig pose data from Kimodo."""

import json
import logging
import os
import struct
from io import BytesIO

import numpy as np
from PIL import Image
import torch
from typing_extensions import override

import folder_paths
from comfy.cli_args import args
from comfy_api.latest import ComfyExtension, IO, Types

from comfy_extras.sam3d_body.export.bvh import build_bvh
from comfy_extras.sam3d_body.export.glb_openpose import build_glb_openpose
from comfy_extras.sam3d_body.export.glb_skeletal import build_glb_skeletal


MHRPoseData = IO.Custom("MHR_POSE_DATA")
KimodoPoseData = IO.Custom("KIMODO_POSE_DATA")
SAM3DBodyModel = IO.Custom("SAM3D_BODY_MODEL")


def pack_variable_mesh_batch(vertices, faces, colors=None, uvs=None, texture=None, unlit=False):
    # Pack lists of (Nᵢ, *) vertex/face/color/uv tensors into padded batched tensors,
    # stashing per-item lengths as runtime attrs so consumers can recover the real slice.
    # colors and uvs are 1:1 with vertices, so they're padded to max_vertices and read with vertex_counts.
    # texture is (B, H, W, 3) — passed through unchanged
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

    return Types.MESH(packed_vertices, packed_faces,
                      uvs=packed_uvs, vertex_colors=packed_colors, texture=texture,
                      vertex_counts=vertex_counts, face_counts=face_counts, unlit=unlit)


def get_mesh_batch_item(mesh, index):
    # Returns (vertices, faces, colors, uvs) for batch index, slicing to real lengths
    # if the mesh carries per-item counts (variable-size batch).
    v_colors = getattr(mesh, "vertex_colors", None)
    v_uvs = getattr(mesh, "uvs", None)
    if getattr(mesh, "vertex_counts", None) is not None:
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = v_colors[index, :vertex_count] if v_colors is not None else None
        uvs = v_uvs[index, :vertex_count] if v_uvs is not None else None
        return vertices, faces, colors, uvs

    colors = v_colors[index] if v_colors is not None else None
    uvs = v_uvs[index] if v_uvs is not None else None
    return mesh.vertices[index], mesh.faces[index], colors, uvs


def save_glb(vertices, faces, filepath, metadata=None,
             uvs=None, vertex_colors=None, texture_image=None, unlit=False):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 3) - The face indices (triangle faces)
    filepath: str - Output filepath (should end with .glb)
    metadata: dict - Optional asset.extras metadata
    uvs: torch.Tensor of shape (N, 2) - Optional per-vertex texture coordinates
    vertex_colors: torch.Tensor of shape (N, 3) or (N, 4) - Optional per-vertex colors in [0, 1]
    texture_image: PIL.Image - Optional baseColor texture, embedded as PNG
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
    faces_np = faces_signed.astype(np.uint32)
    texture_png_bytes = None
    if texture_image is not None:
        buf = BytesIO()
        texture_image.save(buf, format="PNG")
        texture_png_bytes = buf.getvalue()

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()
    uvs_buffer = uvs_np.tobytes() if uvs_np is not None else b""
    colors_buffer = colors_np.tobytes() if colors_np is not None else b""
    texture_buffer = texture_png_bytes if texture_png_bytes is not None else b""

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
    indices_buffer_padded = pad_to_4_bytes(indices_buffer)
    uvs_buffer_padded = pad_to_4_bytes(uvs_buffer)
    colors_buffer_padded = pad_to_4_bytes(colors_buffer)
    texture_buffer_padded = pad_to_4_bytes(texture_buffer)

    buffer_data = b"".join([
        vertices_buffer_padded,
        indices_buffer_padded,
        uvs_buffer_padded,
        colors_buffer_padded,
        texture_buffer_padded,
    ])

    vertices_byte_length = len(vertices_buffer)
    vertices_byte_offset = 0
    indices_byte_length = len(indices_buffer)
    indices_byte_offset = len(vertices_buffer_padded)
    uvs_byte_offset = indices_byte_offset + len(indices_buffer_padded)
    colors_byte_offset = uvs_byte_offset + len(uvs_buffer_padded)
    texture_byte_offset = colors_byte_offset + len(colors_buffer_padded)

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
    if unlit and texture_png_bytes is None:
        # Flat, light-independent shading (KHR_materials_unlit): COLOR_0 is shown as-is, matching how a
        # gaussian splat renders (emissive). Without this the viewer lights the mesh and washes the colours.
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
            "extensions": {"KHR_materials_unlit": {}},
            "doubleSided": True,
        })
        extensions_used.append("KHR_materials_unlit")
        primitive["material"] = 0
    if texture_png_bytes is not None and "TEXCOORD_0" in primitive_attributes:
        buffer_views.append({
            "buffer": 0,
            "byteOffset": texture_byte_offset,
            "byteLength": len(texture_buffer),
        })
        images.append({"bufferView": len(buffer_views) - 1, "mimeType": "image/png"})
        samplers.append({"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071})
        textures.append({"source": 0, "sampler": 0})
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0, "texCoord": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        })
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

    # Write the GLB file
    with open(filepath, 'wb') as f:
        f.write(glb_header)
        f.write(json_chunk_header)
        f.write(gltf_json_padded)
        f.write(bin_chunk_header)
        f.write(buffer_data)

    return filepath


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
            texture_b = getattr(mesh, "texture", None)
            texture_np = None
            if texture_b is not None:
                texture_np = (texture_b.clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
                assert texture_np.ndim == 4 and texture_np.shape[-1] == 3, (
                    f"texture must be (B, H, W, 3) RGB, got shape {tuple(texture_np.shape)}"
                )
            for i in range(mesh.vertices.shape[0]):
                vertices_i, faces_i, v_colors, uvs_i = get_mesh_batch_item(mesh, i)
                if vertices_i.shape[0] == 0 or faces_i.shape[0] == 0:
                    logging.warning(f"SaveGLB: skipping empty mesh at batch index {i}")
                    continue
                tex_img = Image.fromarray(texture_np[i], mode="RGB") if texture_np is not None else None
                f = f"{filename}_{counter:05}_.glb"
                save_glb(vertices_i, faces_i, os.path.join(full_output_folder, f), metadata,
                         uvs=uvs_i,
                         vertex_colors=v_colors,
                         texture_image=tex_img,
                         unlit=getattr(mesh, "unlit", False))
                results.append({
                    "filename": f,
                    "subfolder": subfolder,
                    "type": "output"
                })
                counter += 1
        return IO.NodeOutput(ui={"3d": results})


def rainbow_tilt_inputs():
    """Shared rainbow-shader tilt inputs (used by Render and ToGLB schemas)."""
    return [
        IO.Float.Input(
            "rainbow_tilt_z", default=-35.0, min=-90.0, max=90.0, step=0.5,
            tooltip="Rotate rainbow jet axis around Z (forward). Differentiates left/right.",
        ),
        IO.Float.Input(
            "rainbow_tilt_x", default=0.0, min=-90.0, max=90.0, step=0.5,
            tooltip="Rotate rainbow jet axis around X (right). Differentiates front/back.",
        ),
    ]


def camera_translation_input():
    """Shared camera_translation combo (BuildPoseGLB + SavePoseBVH)."""
    return IO.Combo.Input(
        "camera_translation",
        options=["off", "centered", "absolute"],
        default="off",
        tooltip=(
            "Bake pred_cam_t into the root's translation "
            "'off' = bind position "
            "'centered' = delta from frame 0 "
            "'absolute' = raw (Z is camera depth — usually meters away)."
        ),
    )


class BuildPoseGLB(IO.ComfyNode):
    """Convert pose_data to an in-memory animated GLB"""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BuildPoseGLB",
            display_name="Build Pose GLB",
            description="Convert pose data to an animated GLB",
            category="3d",
            inputs=[
                IO.MultiType.Input(
                    "pose_data", types=[MHRPoseData, KimodoPoseData],
                    tooltip=("MHR pose data from SAM3DBody_Predict, Kimodo. "),
                ),
                SAM3DBodyModel.Input("sam3d_body_model", optional=True),
                IO.DynamicCombo.Input(
                    "mesh_style",
                    options=[
                        IO.DynamicCombo.Option("body_mesh", [
                            IO.DynamicCombo.Input(
                                "bone_vis",
                                options=[
                                    IO.DynamicCombo.Option("off", []),
                                    IO.DynamicCombo.Option("octahedrons", [
                                        IO.Float.Input(
                                            "bone_vis_radius_m",
                                            default=0.02, min=0.005, max=0.5, step=0.005, advanced=True,
                                            tooltip="Radius in m (sphere radius / octahedron half-width).",
                                        ),
                                        IO.Combo.Input(
                                            "bone_vis_color",
                                            options=["white", "rainbow_y"],
                                            default="rainbow_y",
                                            tooltip=(
                                                "Per-bone vertex colors (unlit material). "
                                                "'white' = none, 'rainbow_y' = head→toe jet."
                                            ),
                                        ),
                                    ]),
                                ],
                                tooltip=("Bone vis shape, rigidly skinned to each joint. "),
                            ),
                            IO.DynamicCombo.Input(
                                "shader",
                                options=[
                                    IO.DynamicCombo.Option("default", []),
                                    IO.DynamicCombo.Option("rainbow", [
                                        *rainbow_tilt_inputs(),
                                        IO.Float.Input(
                                            "person_palette_falloff",
                                            default=0.6, min=0.1, max=1.0, step=0.05,
                                            tooltip="Per-person desaturation: each track gets (1 - falloff^k) pastel mix.",
                                        ),
                                    ]),
                                    IO.DynamicCombo.Option("rainbow_face_normal", [
                                        *rainbow_tilt_inputs(),
                                        IO.Float.Input(
                                            "person_palette_falloff",
                                            default=0.6, min=0.1, max=1.0, step=0.05,
                                            tooltip="Per-person desaturation: each track gets (1 - falloff^k) pastel mix.",
                                        ),
                                    ]),
                                    IO.DynamicCombo.Option("rainbow_face_semantic", [
                                        *rainbow_tilt_inputs(),
                                        IO.Float.Input(
                                            "person_palette_falloff",
                                            default=0.6, min=0.1, max=1.0, step=0.05,
                                            tooltip="Per-person desaturation: each track gets (1 - falloff^k) pastel mix.",
                                        ),
                                    ]),
                                ],
                                tooltip=(
                                    "Bake per-vertex colors matching the Render node's shaders "
                                    "(COLOR_0 + KHR_materials_unlit). 'default' = no colors."
                                ),
                            ),
                        ]),
                        IO.DynamicCombo.Option("bones_only", [
                            IO.DynamicCombo.Input(
                                "bone_vis",
                                options=[
                                    IO.DynamicCombo.Option("octahedrons", [
                                        IO.Float.Input(
                                            "bone_vis_radius_m",
                                            default=0.02, min=0.005, max=0.5, step=0.005, advanced=True,
                                            tooltip="Radius in m (sphere radius / octahedron half-width).",
                                        ),
                                        IO.Combo.Input(
                                            "bone_vis_color",
                                            options=["white", "rainbow_y"],
                                            default="rainbow_y",
                                            tooltip=(
                                                "Per-bone vertex colors (unlit material). "
                                                "'white' = none, 'rainbow_y' = head→toe jet."
                                            ),
                                        ),
                                    ]),
                                ],
                                tooltip=(
                                    "Bone vis shape, rigidly skinned to each joint. "
                                    "'octahedrons' = Blender-style directional bones (joint → "
                                    "primary child)."
                                ),
                            ),
                        ]),
                        IO.DynamicCombo.Option("openpose", [
                            IO.Float.Input(
                                "marker_radius_m", default=0.010, min=0.005, max=0.1, step=0.001, advanced=True,
                                tooltip="Sphere radius in m.",
                            ),
                            IO.Float.Input(
                                "stick_radius_m", default=0.008, min=0.002, max=0.05, step=0.001, advanced=True,
                                tooltip="Limb half-width in m. Auto-clamped to bone_length x 0.1.",
                            ),
                            IO.Boolean.Input(
                                "include_hands", default=False,
                                tooltip=(
                                    "Append 21+21 OpenPose hands (wrist + 5 fingers x 4 joints, "
                                    "base→tip) sourced from pred_keypoints_3d."
                                ),
                            ),
                            IO.Float.Input(
                                "hand_marker_radius_m", default=0.005, min=0.001, max=0.1, step=0.001, advanced=True,
                                tooltip="Hand sphere radius in m.",
                            ),
                            IO.Float.Input(
                                "hand_stick_radius_m", default=0.003, min=0.001, max=0.05, step=0.001, advanced=True,
                                tooltip="Hand limb half-width in m.",
                            ),
                            IO.Combo.Input(
                                "face_style",
                                options=["disabled", "full", "eyes_mouth"],
                                default="disabled",
                                tooltip=(
                                    "Face-contour landmarks sampled from pred_vertices at fixed "
                                    "head-mesh vertex IDs (needs canonical_colors on pose_data). "
                                    "'full' = all ~30 points; 'eyes_mouth' = eyes + outer lips only."
                                ),
                            ),
                            IO.Float.Input(
                                "face_marker_radius_m", default=0.0, min=0.0, max=0.05, step=0.0005, advanced=True,
                                tooltip="Face dot radius. 0 = auto = 0.3 x marker_radius_m.",
                            ),
                        ]),
                        IO.DynamicCombo.Option("scail", [
                            IO.Float.Input(
                                "stick_radius_m", default=0.022, min=0.002, max=0.1, step=0.001, advanced=True,
                                tooltip=(
                                    "Cylinder radius in m. Bones are open cylinders at constant "
                                    "radius; joint spheres (auto-sized to match) cap the open ends. "
                                    "SCAIL reference = 0.0215 m."
                                ),
                            ),
                            IO.Float.Input(
                                "marker_radius_m", default=0.0, min=0.0, max=0.1, step=0.001, advanced=True,
                                tooltip="Joint sphere radius. 0 = auto = stick_radius_m (flush cap).",
                            ),
                            IO.Float.Input(
                                "material_roughness", default=0.3, min=0.0, max=1.0, step=0.05, advanced=True,
                                tooltip="PBR roughness. SCAIL ref = 0.3. 1 = matte; 0 = chrome.",
                            ),
                            IO.Boolean.Input(
                                "include_hands", default=False,
                                tooltip="Append 21+21 hand keypoints + capsule sticks per track.",
                            ),
                            IO.Float.Input(
                                "hand_marker_radius_m", default=0.005, min=0.001, max=0.05, step=0.001, advanced=True,
                                tooltip="Hand sphere radius in m.",
                            ),
                            IO.Float.Input(
                                "hand_stick_radius_m", default=0.003, min=0.001, max=0.05, step=0.001, advanced=True,
                                tooltip="Hand cylinder radius in m.",
                            ),
                            IO.Combo.Input(
                                "face_style",
                                options=["disabled", "full", "eyes_mouth"],
                                default="disabled",
                                tooltip=(
                                    "Face-contour landmarks sampled from pred_vertices (needs "
                                    "canonical_colors on pose_data). 'full' = all ~30 points; "
                                    "'eyes_mouth' = eyes + outer lips only."
                                ),
                            ),
                        ]),
                    ],
                    tooltip=(
                        "'body_mesh' = real Armature (127 bones, skinning, TRS keyframes, 72 face morphs; needs model). "
                        "'bones_only' = bone-shape primitives at each joint (preview armature). "
                        "'openpose' = OpenPose-18 3D skeleton from keypoints "
                        "'scail' = SCAIL 3D capsule rig (open cylinders capped flush by joint spheres)."
                    ),
                ),
                IO.Int.Input(
                    "bone_smooth_window",
                    default=0, min=0, max=51, step=2,
                    tooltip=(
                        "Gaussian smoothing window on per-bone rotation keyframes / keypoint "
                        "tracks. 0 = off. 7-15 calms spins/jitter where upstream Smooth misses spikes."
                    ),
                ),
                IO.Float.Input(
                    "fps", default=24.0, min=1.0, max=240.0, step=1.0,
                    tooltip="Animation frame rate.",
                ),
                camera_translation_input(),
                IO.Int.Input(
                    "track_index", default=-1, min=-1, max=15,
                    tooltip="-1 = all tracks; ≥0 = single track.",
                ),
            ],
            outputs=[IO.File3DGLB.Output("glb")],
        )

    @classmethod
    def execute(cls, pose_data, mesh_style, sam3d_body_model=None, bone_smooth_window=0, fps=24.0, camera_translation="off", track_index=-1) -> IO.NodeOutput:
        mesh_style = mesh_style or {"mesh_style": "body_mesh"}
        mode_key = mesh_style["mesh_style"]
        # `shader` is nested in body_mesh; absent for bones_only.
        shader_dict = mesh_style.get("shader") or {}
        shader_key = shader_dict.get("shader", "default")
        common = dict(
            fps=float(fps),
            camera_translation=str(camera_translation),
            track_index=int(track_index),
            shader=str(shader_key),
            rainbow_tilt_x_deg=float(shader_dict.get("rainbow_tilt_x", 0.0)),
            rainbow_tilt_z_deg=float(shader_dict.get("rainbow_tilt_z", 0.0)),
            person_palette_falloff=float(shader_dict.get("person_palette_falloff", 0.6)),
        )
        if mode_key in ("body_mesh", "bones_only"):
            # External rigs (e.g. ComfyUI-Kimodo) supply pose_data["_skeleton_override"]
            # so the GLB writer reads rig/bind/skin from there instead of MHR.
            has_external_rig = isinstance(pose_data, dict) and ("_skeleton_override" in pose_data)
            if sam3d_body_model is None and not has_external_rig:
                raise ValueError(
                    f"BuildPoseGLB: '{mode_key}' mode needs the `sam3d_body_model` input OR a "
                    "`_skeleton_override` dict in pose_data. Connect the SAM3DBody model "
                    "or feed pose_data from a node that supplies the override (e.g. KimodoSample)."
                )
            default_shape = "off" if mode_key == "body_mesh" else "octahedrons"
            bone_vis_dict = mesh_style.get("bone_vis", {"bone_vis": default_shape})
            bone_vis = str(bone_vis_dict.get("bone_vis", default_shape))
            bone_vis_radius_m = float(bone_vis_dict.get("bone_vis_radius_m", 0.04))
            bone_vis_color = str(bone_vis_dict.get("bone_vis_color", "white"))
            glb_bytes = build_glb_skeletal(
                pose_data, sam3d_body_model,
                bone_smooth_window=int(bone_smooth_window),
                bone_vis=bone_vis,
                bone_vis_radius_m=bone_vis_radius_m,
                bone_vis_color=bone_vis_color,
                include_body_mesh=(mode_key == "body_mesh"),
                **common,
            )
        elif mode_key == "openpose":
            # Rig-independent: sourced from pred_keypoints_3d. face_source='rig'
            # additionally reads canonical_colors for head-mesh vertex IDs.
            glb_bytes = build_glb_openpose(
                pose_data,
                fps=float(fps),
                camera_translation=str(camera_translation),
                track_index=int(track_index),
                marker_radius_m=float(mesh_style.get("marker_radius_m", 0.025)),
                stick_radius_m=float(mesh_style.get("stick_radius_m", 0.008)),
                include_hands=bool(mesh_style.get("include_hands", False)),
                hand_marker_radius_m=float(mesh_style.get("hand_marker_radius_m", 0.005)),
                hand_stick_radius_m=float(mesh_style.get("hand_stick_radius_m", 0.003)),
                face_style=str(mesh_style.get("face_style", "disabled")),
                face_marker_radius_m=float(mesh_style.get("face_marker_radius_m", 0.0)),
                palette="openpose",
                shape="ellipsoid",
                bone_smooth_window=int(bone_smooth_window),
            )
        elif mode_key == "scail":
            # SCAIL rig: open cylinders capped flush by joint spheres (sphere
            # radius defaults to cylinder radius for a seamless silhouette).
            cap_stick_radius = float(mesh_style.get("stick_radius_m", 0.022))
            cap_marker_radius = float(mesh_style.get("marker_radius_m", 0.0))
            if cap_marker_radius <= 0.0:
                cap_marker_radius = cap_stick_radius
            glb_bytes = build_glb_openpose(
                pose_data,
                fps=float(fps),
                camera_translation=str(camera_translation),
                track_index=int(track_index),
                marker_radius_m=cap_marker_radius,
                stick_radius_m=cap_stick_radius,
                include_hands=bool(mesh_style.get("include_hands", False)),
                hand_marker_radius_m=float(mesh_style.get("hand_marker_radius_m", 0.005)),
                hand_stick_radius_m=float(mesh_style.get("hand_stick_radius_m", 0.003)),
                face_style=str(mesh_style.get("face_style", "disabled")),
                palette="scail",
                shape="capsule",
                smooth_shade=True,
                # SCAIL material: slightly glossy (0.3) + double-sided so the
                # inside of the open cylinders shades sensibly at grazing angles.
                material_roughness=float(mesh_style.get("material_roughness", 0.3)),
                material_double_sided=True,
                bone_smooth_window=int(bone_smooth_window),
            )
        else:
            raise ValueError(f"BuildPoseGLB: unknown mesh_style {mode_key!r}")

        return IO.NodeOutput(Types.File3D(BytesIO(glb_bytes), file_format="glb"))


class SavePoseBVH(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SavePoseBVH",
            description="Save pose data as BVH mocap file",
            display_name="Save Pose BVH",
            category="3d",
            is_output_node=True,
            inputs=[
                IO.MultiType.Input(
                    "pose_data", types=[MHRPoseData, KimodoPoseData],
                    tooltip=(
                        "MHR pose data from SAM3DBody_Predict, or external-rig "
                        "pose data from Kimodo."
                    ),
                ),
                SAM3DBodyModel.Input("sam3d_body_model"),
                IO.String.Input("filename_prefix", default="3d/ComfyUI"),
                IO.Float.Input(
                    "fps", default=24.0, min=1.0, max=240.0, step=1.0,
                    tooltip="Animation frame rate (BVH `Frame Time`).",
                ),
                camera_translation_input(),
                IO.Combo.Input(
                    "units",
                    options=["cm", "m"],
                    default="cm",
                    tooltip="BVH OFFSET/position units. 'cm' is the mocap standard.",
                ),
                IO.Int.Input(
                    "track_index", default=0, min=0, max=15,
                    tooltip="Track to export. BVH carries one skeleton; export multi-person clips one at a time.",
                ),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            outputs=[],
        )

    @classmethod
    def execute(cls, pose_data, sam3d_body_model, filename_prefix="3d/ComfyUI",
                fps=24.0, camera_translation="off", units="cm",
                track_index=0) -> IO.NodeOutput:
        bvh_bytes = build_bvh(
            pose_data, sam3d_body_model,
            fps=float(fps),
            camera_translation=str(camera_translation),
            track_index=int(track_index),
            units=str(units),
        )

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory(),
            )
        f = f"{filename}_{counter:05}_.bvh"
        out_path = os.path.join(full_output_folder, f)
        with open(out_path, "wb") as fh:
            fh.write(bvh_bytes)

        return IO.NodeOutput(ui={"3d": [{
            "filename": f,
            "subfolder": subfolder,
            "type": "output",
        }]})


class Save3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [SaveGLB, BuildPoseGLB, SavePoseBVH]


async def comfy_entrypoint() -> Save3DExtension:
    return Save3DExtension()

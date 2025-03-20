import torch
import os
import json
import struct
import numpy as np
from comfy.ldm.modules.diffusionmodules.mmdit import get_1d_sincos_pos_embed_from_grid_torch
import folder_paths
import comfy.model_management
from comfy.cli_args import args


class EmptyLatentHunyuan3Dv2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"resolution": ("INT", {"default": 3072, "min": 1, "max": 8192}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent/3d"

    def generate(self, resolution, batch_size):
        latent = torch.zeros([batch_size, 64, resolution], device=comfy.model_management.intermediate_device())
        return ({"samples": latent, "type": "hunyuan3dv2"}, )


class Hunyuan3Dv2Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_vision_output": ("CLIP_VISION_OUTPUT",),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")

    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, clip_vision_output):
        embeds = clip_vision_output.last_hidden_state
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return (positive, negative)


class Hunyuan3Dv2ConditioningMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {},
                "optional": {"front": ("CLIP_VISION_OUTPUT",),
                             "left": ("CLIP_VISION_OUTPUT",),
                             "back": ("CLIP_VISION_OUTPUT",),
                             "right": ("CLIP_VISION_OUTPUT",), }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")

    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, front=None, left=None, back=None, right=None):
        all_embeds = [front, left, back, right]
        out = []
        pos_embeds = None
        for i, e in enumerate(all_embeds):
            if e is not None:
                if pos_embeds is None:
                    pos_embeds = get_1d_sincos_pos_embed_from_grid_torch(e.last_hidden_state.shape[-1], torch.arange(4))
                out.append(e.last_hidden_state + pos_embeds[i].reshape(1, 1, -1))

        embeds = torch.cat(out, dim=1)
        positive = [[embeds, {}]]
        negative = [[torch.zeros_like(embeds), {}]]
        return (positive, negative)


class VOXEL:
    def __init__(self, data):
        self.data = data


class VAEDecodeHunyuan3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT", ),
                             "vae": ("VAE", ),
                             "num_chunks": ("INT", {"default": 8000, "min": 1000, "max": 500000}),
                             "octree_resolution": ("INT", {"default": 256, "min": 16, "max": 512}),
                             }}
    RETURN_TYPES = ("VOXEL",)
    FUNCTION = "decode"

    CATEGORY = "latent/3d"

    def decode(self, vae, samples, num_chunks, octree_resolution):
        voxels = VOXEL(vae.decode(samples["samples"], vae_options={"num_chunks": num_chunks, "octree_resolution": octree_resolution}))
        return (voxels, )


def voxel_to_mesh(voxels, threshold=0.5, device=None):
    if device is None:
        device = torch.device("cpu")
    voxels = voxels.to(device)

    binary = (voxels > threshold).float()
    padded = torch.nn.functional.pad(binary, (1, 1, 1, 1, 1, 1), 'constant', 0)

    D, H, W = binary.shape

    neighbors = torch.tensor([
        [0, 0, 1],
        [0, 0, -1],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 0],
        [-1, 0, 0]
    ], device=device)

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device),
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    voxel_indices = torch.stack([z.flatten(), y.flatten(), x.flatten()], dim=1)

    solid_mask = binary.flatten() > 0
    solid_indices = voxel_indices[solid_mask]

    corner_offsets = [
        torch.tensor([
            [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]
        ], device=device),
        torch.tensor([
            [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]
        ], device=device),
        torch.tensor([
            [0, 1, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]
        ], device=device)
    ]

    all_vertices = []
    all_indices = []

    vertex_count = 0

    for face_idx, offset in enumerate(neighbors):
        neighbor_indices = solid_indices + offset

        padded_indices = neighbor_indices + 1

        is_exposed = padded[
            padded_indices[:, 0],
            padded_indices[:, 1],
            padded_indices[:, 2]
        ] == 0

        if not is_exposed.any():
            continue

        exposed_indices = solid_indices[is_exposed]

        corners = corner_offsets[face_idx].unsqueeze(0)

        face_vertices = exposed_indices.unsqueeze(1) + corners

        all_vertices.append(face_vertices.reshape(-1, 3))

        num_faces = exposed_indices.shape[0]
        face_indices = torch.arange(
            vertex_count,
            vertex_count + 4 * num_faces,
            device=device
        ).reshape(-1, 4)

        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 1], face_indices[:, 2]], dim=1))
        all_indices.append(torch.stack([face_indices[:, 0], face_indices[:, 2], face_indices[:, 3]], dim=1))

        vertex_count += 4 * num_faces

    if len(all_vertices) > 0:
        vertices = torch.cat(all_vertices, dim=0)
        faces = torch.cat(all_indices, dim=0)
    else:
        vertices = torch.zeros((1, 3))
        faces = torch.zeros((1, 3))

    v_min = 0
    v_max = max(voxels.shape)

    vertices = vertices - (v_min + v_max) / 2

    scale = (v_max - v_min) / 2
    if scale > 0:
        vertices = vertices / scale

    vertices = torch.fliplr(vertices)
    return vertices, faces


class MESH:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces


class VoxelToMeshBasic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"voxel": ("VOXEL", ),
                             "threshold": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0, "step": 0.01}),
                             }}
    RETURN_TYPES = ("MESH",)
    FUNCTION = "decode"

    CATEGORY = "3d"

    def decode(self, voxel, threshold):
        vertices = []
        faces = []
        for x in voxel.data:
            v, f = voxel_to_mesh(x, threshold=threshold, device=None)
            vertices.append(v)
            faces.append(f)

        return (MESH(torch.stack(vertices), torch.stack(faces)), )


def save_glb(vertices, faces, filepath, metadata=None):
    """
    Save PyTorch tensor vertices and faces as a GLB file without external dependencies.

    Parameters:
    vertices: torch.Tensor of shape (N, 3) - The vertex coordinates
    faces: torch.Tensor of shape (M, 4) or (M, 3) - The face indices (quad or triangle faces)
    filepath: str - Output filepath (should end with .glb)
    """

    # Convert tensors to numpy arrays
    vertices_np = vertices.cpu().numpy().astype(np.float32)
    faces_np = faces.cpu().numpy().astype(np.uint32)

    vertices_buffer = vertices_np.tobytes()
    indices_buffer = faces_np.tobytes()

    def pad_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b'\x00' * padding_length

    vertices_buffer_padded = pad_to_4_bytes(vertices_buffer)
    indices_buffer_padded = pad_to_4_bytes(indices_buffer)

    buffer_data = vertices_buffer_padded + indices_buffer_padded

    vertices_byte_length = len(vertices_buffer)
    vertices_byte_offset = 0
    indices_byte_length = len(indices_buffer)
    indices_byte_offset = len(vertices_buffer_padded)

    gltf = {
        "asset": {"version": "2.0", "generator": "ComfyUI"},
        "buffers": [
            {
                "byteLength": len(buffer_data)
            }
        ],
        "bufferViews": [
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
        ],
        "accessors": [
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
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0
                        },
                        "indices": 1,
                        "mode": 4  # TRIANGLES
                    }
                ]
            }
        ],
        "nodes": [
            {
                "mesh": 0
            }
        ],
        "scenes": [
            {
                "nodes": [0]
            }
        ],
        "scene": 0
    }

    if metadata is not None:
        gltf["asset"]["extras"] = metadata

    # Convert the JSON to bytes
    gltf_json = json.dumps(gltf).encode('utf8')

    def pad_json_to_4_bytes(buffer):
        padding_length = (4 - (len(buffer) % 4)) % 4
        return buffer + b' ' * padding_length

    gltf_json_padded = pad_json_to_4_bytes(gltf_json)

    # Create the GLB header
    # Magic glTF
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


class SaveGLB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"mesh": ("MESH", ),
                             "filename_prefix": ("STRING", {"default": "mesh/ComfyUI"}), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}, }

    RETURN_TYPES = ()
    FUNCTION = "save"

    OUTPUT_NODE = True

    CATEGORY = "3d"

    def save(self, mesh, filename_prefix, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        results = []

        metadata = {}
        if not args.disable_metadata:
            if prompt is not None:
                metadata["prompt"] = json.dumps(prompt)
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        for i in range(mesh.vertices.shape[0]):
            f = f"{filename}_{counter:05}_.glb"
            save_glb(mesh.vertices[i], mesh.faces[i], os.path.join(full_output_folder, f), metadata)
            results.append({
                "filename": f,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1
        return {"ui": {"3d": results}}


NODE_CLASS_MAPPINGS = {
    "EmptyLatentHunyuan3Dv2": EmptyLatentHunyuan3Dv2,
    "Hunyuan3Dv2Conditioning": Hunyuan3Dv2Conditioning,
    "Hunyuan3Dv2ConditioningMultiView": Hunyuan3Dv2ConditioningMultiView,
    "VAEDecodeHunyuan3D": VAEDecodeHunyuan3D,
    "VoxelToMeshBasic": VoxelToMeshBasic,
    "SaveGLB": SaveGLB,
}

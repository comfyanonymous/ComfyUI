from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
from comfy.ldm.trellis2.vae import SparseTensor
import comfy.model_management
import logging
from PIL import Image
import numpy as np
import torch
import scipy
import copy

shape_slat_normalization = {
    "mean": torch.tensor([
        0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
        -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
        0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
        -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665
    ])[None],
    "std": torch.tensor([
        5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
        5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
        4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
        5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691
    ])[None]
}

tex_slat_normalization = {
    "mean": torch.tensor([
        3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075,
        0.029398, -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149,
        -1.209521, 2.069067, 0.544735, 2.569128, -0.323407, 2.293000, -1.925608, -1.217717,
        1.213905, 0.971588, -0.023631, 0.106750, 2.021786, 0.250524, -0.662387, -0.768862
    ])[None],
    "std": torch.tensor([
        2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822,
        2.969419, 2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588,
        2.295916, 3.234985, 3.233086, 2.260140, 2.874801, 2.810596, 3.292720, 2.674999,
        2.680878, 2.372054, 2.451546, 2.353556, 2.995195, 2.379849, 2.786195, 2.775190
    ])[None]
}

def shape_norm(shape_latent, coords):
    std = shape_slat_normalization["std"].to(shape_latent)
    mean = shape_slat_normalization["mean"].to(shape_latent)
    samples = SparseTensor(feats = shape_latent, coords=coords)
    samples = samples * std + mean
    return samples

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

    final_colors = (v_colors * 0.5 + 0.5).clamp(0, 1).unsqueeze(0)

    out_mesh = copy.deepcopy(mesh)
    out_mesh.colors = final_colors

    return out_mesh

class VaeDecodeShapeTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeShapeTrellis",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("resolution", options=["512", "1024"], default="1024")
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                IO.AnyType.Output("shape_subs"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):

        resolution = int(resolution)
        patcher = vae.patcher
        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(patcher)

        vae = vae.first_stage_model
        coords = samples["coords"]

        samples = samples["samples"]
        samples = samples.squeeze(-1).transpose(1, 2).reshape(-1, 32).to(device)
        samples = shape_norm(samples, coords)

        mesh, subs = vae.decode_shape_slat(samples, resolution)
        faces = torch.stack([m.faces for m in mesh])
        verts = torch.stack([m.vertices for m in mesh])
        mesh = Types.MESH(vertices=verts, faces=faces)
        return IO.NodeOutput(mesh, subs)

class VaeDecodeTextureTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeTextureTrellis",
            category="latent/3d",
            inputs=[
                IO.Mesh.Input("shape_mesh"),
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.AnyType.Input("shape_subs"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
            ]
        )

    @classmethod
    def execute(cls, shape_mesh, samples, vae, shape_subs):

        resolution = 1024
        patcher = vae.patcher
        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(patcher)

        vae = vae.first_stage_model
        coords = samples["coords"]

        samples = samples["samples"]
        samples = samples.squeeze(-1).transpose(1, 2).reshape(-1, 32).to(device)
        std = tex_slat_normalization["std"].to(samples)
        mean = tex_slat_normalization["mean"].to(samples)
        samples = SparseTensor(feats = samples, coords=coords)
        samples = samples * std + mean

        voxel = vae.decode_tex_slat(samples, shape_subs)
        color_feats = voxel.feats[:, :3]
        voxel_coords = voxel.coords[:, 1:]

        out_mesh = paint_mesh_with_voxels(shape_mesh, voxel_coords, color_feats, resolution=resolution)
        return IO.NodeOutput(out_mesh)

class VaeDecodeStructureTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeStructureTrellis2",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("resolution", options=["32", "64"], default="32")
            ],
            outputs=[
                IO.Voxel.Output("structure_output"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):
        resolution = int(resolution)
        vae = vae.first_stage_model
        decoder = vae.struct_dec
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.vae_offload_device()
        decoder = decoder.to(load_device)
        samples = samples["samples"]
        samples = samples.to(load_device)
        decoded = decoder(samples)>0
        decoder.to(offload_device)
        current_res = decoded.shape[2]

        if current_res != resolution:
            ratio = current_res // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        out = Types.VOXEL(decoded.squeeze(1).float())
        return IO.NodeOutput(out)

class Trellis2UpsampleCascade(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2UpsampleCascade",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("shape_latent_512"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("target_resolution", options=["1024", "1536"], default="1024"),
                IO.Int.Input("max_tokens", default=49152, min=1024, max=100000)
            ],
            outputs=[
                IO.AnyType.Output("hr_coords"),
            ]
        )

    @classmethod
    def execute(cls, shape_latent_512, vae, target_resolution, max_tokens):
        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(vae.patcher)

        feats = shape_latent_512["samples"].squeeze(-1).transpose(1, 2).reshape(-1, 32).to(device)
        coords_512 = shape_latent_512["coords"].to(device)

        slat = shape_norm(feats, coords_512)

        decoder = vae.first_stage_model.shape_dec

        slat.feats = slat.feats.to(next(decoder.parameters()).dtype)
        hr_coords = decoder.upsample(slat, upsample_times=4)

        lr_resolution = 512
        hr_resolution = int(target_resolution)

        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            final_coords = quant_coords.unique(dim=0)
            num_tokens = final_coords.shape[0]

            if num_tokens < max_tokens or hr_resolution <= 1024:
                break
            hr_resolution -= 128

        return IO.NodeOutput(final_coords,)

dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
dino_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def run_conditioning(model, cropped_img_tensor, include_1024=True):
    model_internal = model.model
    device = comfy.model_management.intermediate_device()
    torch_device = comfy.model_management.get_torch_device()

    img_t = cropped_img_tensor.to(torch_device)

    def prepare_tensor(img, size):
        resized = torch.nn.functional.interpolate(img, size=(size, size), mode='bicubic', align_corners=False).clamp(0.0, 1.0)
        return (resized - dino_mean.to(torch_device)) / dino_std.to(torch_device)

    model_internal.image_size = 512
    input_512 = prepare_tensor(img_t, 512)
    cond_512 = model_internal(input_512)[0]

    cond_1024 = None
    if include_1024:
        model_internal.image_size = 1024
        input_1024 = prepare_tensor(img_t, 1024)
        cond_1024 = model_internal(input_1024)[0]

    conditioning = {
        'cond_512': cond_512.to(device),
        'neg_cond': torch.zeros_like(cond_512).to(device),
    }
    if cond_1024 is not None:
        conditioning['cond_1024'] = cond_1024.to(device)

    return conditioning

class Trellis2Conditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2Conditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.Combo.Input("background_color", options=["black", "gray", "white"], default="black")
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, image, mask, background_color) -> IO.NodeOutput:

        if image.ndim == 4:
            image = image[0]
        if mask.ndim == 3:
            mask = mask[0]

        img_np = (image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        mask_np = (mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_np)
        pil_mask = Image.fromarray(mask_np)

        max_size = max(pil_img.size)
        scale = min(1.0, 1024 / max_size)
        if scale < 1.0:
            new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)

        rgba_np = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
        rgba_np[:, :, :3] = np.array(pil_img)
        rgba_np[:, :, 3] = np.array(pil_mask)

        alpha = rgba_np[:, :, 3]
        bbox_coords = np.argwhere(alpha > 0.8 * 255)

        if len(bbox_coords) > 0:
            y_min, x_min = np.min(bbox_coords[:, 0]), np.min(bbox_coords[:, 1])
            y_max, x_max = np.max(bbox_coords[:, 0]), np.max(bbox_coords[:, 1])

            center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
            size = max(y_max - y_min, x_max - x_min)

            crop_x1 = int(center_x - size // 2)
            crop_y1 = int(center_y - size // 2)
            crop_x2 = int(center_x + size // 2)
            crop_y2 = int(center_y + size // 2)

            rgba_pil = Image.fromarray(rgba_np, 'RGBA')
            cropped_rgba = rgba_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped_np = np.array(cropped_rgba).astype(np.float32) / 255.0
        else:
            logging.warning("Mask for the image is empty. Trellis2 requires an image with a mask for the best mesh quality.")
            cropped_np = rgba_np.astype(np.float32) / 255.0

        bg_colors = {"black": [0.0, 0.0, 0.0], "gray":[0.5, 0.5, 0.5], "white":[1.0, 1.0, 1.0]}
        bg_rgb = np.array(bg_colors.get(background_color, [0.0, 0.0, 0.0]), dtype=np.float32)

        fg = cropped_np[:, :, :3]
        alpha_float = cropped_np[:, :, 3:4]
        composite_np = fg * alpha_float + bg_rgb * (1.0 - alpha_float)

        # to match trellis2 code (quantize -> dequantize)
        composite_uint8 = (composite_np * 255.0).round().clip(0, 255).astype(np.uint8)

        cropped_img_tensor = torch.from_numpy(composite_uint8).float() / 255.0
        cropped_img_tensor = cropped_img_tensor.movedim(-1, 0).unsqueeze(0)

        conditioning = run_conditioning(clip_vision_model, cropped_img_tensor, include_1024=True)

        embeds = conditioning["cond_1024"]
        positive = [[conditioning["cond_512"], {"embeds": embeds}]]
        negative = [[conditioning["neg_cond"], {"embeds": torch.zeros_like(embeds)}]]
        return IO.NodeOutput(positive, negative)

class EmptyShapeLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyShapeLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.AnyType.Input("structure_or_coords"),
                IO.Model.Input("model")
            ],
            outputs=[
                IO.Latent.Output(),
                IO.Model.Output()
            ]
        )

    @classmethod
    def execute(cls, structure_or_coords, model):
        # to accept the upscaled coords
        is_512_pass = False

        if hasattr(structure_or_coords, "data") and structure_or_coords.data.ndim == 4:
            decoded = structure_or_coords.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
            is_512_pass = True

        elif isinstance(structure_or_coords, torch.Tensor) and structure_or_coords.ndim == 2:
            coords = structure_or_coords.int()
            is_512_pass = False

        else:
            raise ValueError(f"Invalid input to EmptyShapeLatent: {type(structure_or_coords)}")
        in_channels = 32
        # image like format
        latent = torch.randn(1, in_channels, coords.shape[0], 1)
        model = model.clone()
        model.model_options = model.model_options.copy()
        if "transformer_options" in model.model_options:
            model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
        else:
            model.model_options["transformer_options"] = {}

        model.model_options["transformer_options"]["coords"] = coords
        if is_512_pass:
            model.model_options["transformer_options"]["generation_mode"] = "shape_generation_512"
        else:
            model.model_options["transformer_options"]["generation_mode"] = "shape_generation"
        return IO.NodeOutput({"samples": latent, "coords": coords, "type": "trellis2"}, model)

class EmptyTextureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTextureLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Voxel.Input("structure_or_coords"),
                IO.Latent.Input("shape_latent"),
                IO.Model.Input("model")
            ],
            outputs=[
                IO.Latent.Output(),
                IO.Model.Output()
            ]
        )

    @classmethod
    def execute(cls, structure_or_coords, shape_latent, model):
        channels = 32
        if hasattr(structure_or_coords, "data") and structure_or_coords.data.ndim == 4:
            decoded = structure_or_coords.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()

        elif isinstance(structure_or_coords, torch.Tensor) and structure_or_coords.ndim == 2:
            coords = structure_or_coords.int()

        shape_latent = shape_latent["samples"]
        if shape_latent.ndim == 4:
            shape_latent = shape_latent.squeeze(-1).transpose(1, 2).reshape(-1, channels)
        shape_latent = shape_norm(shape_latent, coords)

        latent = torch.randn(1, channels, coords.shape[0], 1)
        model = model.clone()
        model.model_options = model.model_options.copy()
        if "transformer_options" in model.model_options:
            model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
        else:
            model.model_options["transformer_options"] = {}

        model.model_options["transformer_options"]["coords"] = coords
        model.model_options["transformer_options"]["generation_mode"] = "texture_generation"
        model.model_options["transformer_options"]["shape_slat"] = shape_latent
        return IO.NodeOutput({"samples": latent, "coords": coords, "type": "trellis2"}, model)


class EmptyStructureLatentTrellis2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyStructureLatentTrellis2",
            category="latent/3d",
            inputs=[
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    @classmethod
    def execute(cls, batch_size):
        in_channels = 8
        resolution = 16
        latent = torch.randn(batch_size, in_channels, resolution, resolution, resolution)
        return IO.NodeOutput({"samples": latent, "type": "trellis2"})

def simplify_fn(vertices, faces, target=100000):
    is_batched = vertices.ndim == 3
    if is_batched:
        v_list, f_list = [], []
        for i in range(vertices.shape[0]):
            v_i, f_i = simplify_fn(vertices[i], faces[i], target)
            v_list.append(v_i)
            f_list.append(f_i)
        return torch.stack(v_list), torch.stack(f_list)

    if faces.shape[0] <= target:
        return vertices, faces

    device = vertices.device
    target_v = target / 2.0

    min_v = vertices.min(dim=0)[0]
    max_v = vertices.max(dim=0)[0]
    extent = max_v - min_v

    volume = (extent[0] * extent[1] * extent[2]).clamp(min=1e-8)
    cell_size = (volume / target_v) ** (1/3.0)

    quantized = ((vertices - min_v) / cell_size).round().long()
    unique_coords, inverse_indices = torch.unique(quantized, dim=0, return_inverse=True)
    num_cells = unique_coords.shape[0]

    new_vertices = torch.zeros((num_cells, 3), dtype=vertices.dtype, device=device)
    counts = torch.zeros((num_cells, 1), dtype=vertices.dtype, device=device)

    new_vertices.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), vertices)
    counts.scatter_add_(0, inverse_indices.unsqueeze(1), torch.ones_like(vertices[:, :1]))

    new_vertices = new_vertices / counts.clamp(min=1)

    new_faces = inverse_indices[faces]

    valid_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                 (new_faces[:, 1] != new_faces[:, 2]) & \
                 (new_faces[:, 2] != new_faces[:, 0])
    new_faces = new_faces[valid_mask]

    unique_face_indices, inv_face = torch.unique(new_faces.reshape(-1), return_inverse=True)
    final_vertices = new_vertices[unique_face_indices]
    final_faces = inv_face.reshape(-1, 3)

    return final_vertices, final_faces

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
        f_list =[]
        for i in range(faces.shape[0]):
            f_inv = faces[i][:,[0, 2, 1]]
            f_list.append(torch.cat([faces[i], f_inv], dim=0))
        return vertices, torch.stack(f_list)

    faces_inv = faces[:, [0, 2, 1]]
    faces_double = torch.cat([faces, faces_inv], dim=0)
    return vertices, faces_double

class PostProcessMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="PostProcessMesh",
            category="latent/3d",
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.Int.Input("simplify", default=1_000_000, min=0, max=50_000_000),
                IO.Float.Input("fill_holes_perimeter", default=0.03, min=0.0, step=0.0001)
            ],
            outputs=[
                IO.Mesh.Output("output_mesh"),
            ]
        )

    @classmethod
    def execute(cls, mesh, simplify, fill_holes_perimeter):
        mesh = copy.deepcopy(mesh)
        verts, faces = mesh.vertices, mesh.faces

        if fill_holes_perimeter > 0:
            verts, faces = fill_holes_fn(verts, faces, max_perimeter=fill_holes_perimeter)

        if simplify > 0 and faces.shape[0] > simplify:
            verts, faces = simplify_fn(verts, faces, target=simplify)

        verts, faces = make_double_sided(verts, faces)

        mesh.vertices = verts
        mesh.faces = faces
        return IO.NodeOutput(mesh)

class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            EmptyShapeLatentTrellis2,
            EmptyStructureLatentTrellis2,
            EmptyTextureLatentTrellis2,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis,
            VaeDecodeStructureTrellis2,
            Trellis2UpsampleCascade,
            PostProcessMesh
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()

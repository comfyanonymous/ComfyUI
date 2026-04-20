from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types
from comfy.ldm.trellis2.vae import SparseTensor
import comfy.model_management
from PIL import Image
import numpy as np
import torch
import scipy
import copy


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
        mesh.colors = packed_colors
        mesh.color_counts = color_counts

    return mesh


def get_mesh_batch_item(mesh, index):
    if hasattr(mesh, "vertex_counts"):
        vertex_count = int(mesh.vertex_counts[index].item())
        face_count = int(mesh.face_counts[index].item())
        vertices = mesh.vertices[index, :vertex_count]
        faces = mesh.faces[index, :face_count]
        colors = None
        if hasattr(mesh, "colors") and mesh.colors is not None:
            if hasattr(mesh, "color_counts"):
                color_count = int(mesh.color_counts[index].item())
                colors = mesh.colors[index, :color_count]
            else:
                colors = mesh.colors[index, :vertex_count]
        return vertices, faces, colors

    colors = None
    if hasattr(mesh, "colors") and mesh.colors is not None:
        colors = mesh.colors[index]
    return mesh.vertices[index], mesh.faces[index], colors

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


def infer_batched_coord_layout(coords):
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected Trellis2 coords with shape [N, 4], got {tuple(coords.shape)}")

    if coords.shape[0] == 0:
        raise ValueError("Trellis2 coords can't be empty")

    batch_ids = coords[:, 0].to(torch.int64)
    batch_size = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=batch_size)

    if (counts == 0).any():
        raise ValueError(f"Non-contiguous Trellis2 batch ids in coords: {batch_ids.unique(sorted=True).tolist()}")

    max_tokens = int(counts.max().item())
    return batch_size, counts, max_tokens


def flatten_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return samples.reshape(-1, samples.shape[-1]), coords

    feat_list = []
    coord_list = []
    for i in range(coord_counts.shape[0]):
        count = int(coord_counts[i].item())
        coords_i = coords[coords[:, 0] == i]
        if coords_i.shape[0] != count:
            raise ValueError(f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}")
        feat_list.append(samples[i, :count])
        coord_list.append(coords_i)

    return torch.cat(feat_list, dim=0), torch.cat(coord_list, dim=0)


def split_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return [(samples.reshape(-1, samples.shape[-1]), coords)]

    items = []
    for i in range(coord_counts.shape[0]):
        count = int(coord_counts[i].item())
        coords_i = coords[coords[:, 0] == i]
        if coords_i.shape[0] != count:
            raise ValueError(f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}")
        items.append((samples[i, :count], coords_i))
    return items


def paint_mesh_with_voxels(mesh, voxel_coords, voxel_colors, resolution):
    """
    Generic function to paint a mesh using nearest-neighbor colors from a sparse voxel field.
    """
    device = comfy.model_management.vae_offload_device()

    origin = torch.tensor([-0.5, -0.5, -0.5], device=device)
    # TODO: generic independent node? if so: figure how pass the resolution parameter
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

    out_mesh = copy.copy(mesh)
    out_mesh.colors = final_colors

    return out_mesh


def paint_mesh_default_colors(mesh):
    out_mesh = copy.copy(mesh)
    vertex_count = mesh.vertices.shape[1]
    out_mesh.colors = mesh.vertices.new_zeros((1, vertex_count, 3))
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
        coord_counts = samples.get("coord_counts")

        samples = samples["samples"]
        if coord_counts is None:
            samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
            samples = shape_norm(samples.to(device), coords.to(device))
            mesh, subs = vae.decode_shape_slat(samples, resolution)
        else:
            split_items = split_batched_sparse_latent(samples, coords, coord_counts)
            mesh = []
            subs_per_sample = []
            for feats_i, coords_i in split_items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                sample_i = shape_norm(feats_i.to(device), coords_i)
                mesh_i, subs_i = vae.decode_shape_slat(sample_i, resolution)
                mesh.append(mesh_i[0])
                subs_per_sample.append(subs_i)

            subs = []
            for stage_index in range(len(subs_per_sample[0])):
                stage_tensors = [sample_subs[stage_index] for sample_subs in subs_per_sample]
                feats_list = [stage_tensor.feats for stage_tensor in stage_tensors]
                coords_list = [stage_tensor.coords for stage_tensor in stage_tensors]
                subs.append(SparseTensor.from_tensor_list(feats_list, coords_list))

        face_list = [m.faces for m in mesh]
        vert_list = [m.vertices for m in mesh]
        if all(v.shape == vert_list[0].shape for v in vert_list) and all(f.shape == face_list[0].shape for f in face_list):
            mesh = Types.MESH(vertices=torch.stack(vert_list), faces=torch.stack(face_list))
        else:
            mesh = pack_variable_mesh_batch(vert_list, face_list)
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
        coord_counts = samples.get("coord_counts")

        samples = samples["samples"]
        samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
        samples = samples.to(device)
        std = tex_slat_normalization["std"].to(samples)
        mean = tex_slat_normalization["mean"].to(samples)
        samples = SparseTensor(feats = samples, coords=coords.to(device))
        samples = samples * std + mean

        voxel = vae.decode_tex_slat(samples, shape_subs)
        color_feats = voxel.feats[:, :3]
        voxel_coords = voxel.coords[:, 1:]
        voxel_batch_idx = voxel.coords[:, 0]

        mesh_batch_size = shape_mesh.vertices.shape[0]
        if mesh_batch_size > 1:
            out_verts, out_faces, out_colors = [], [], []
            for i in range(mesh_batch_size):
                sel = voxel_batch_idx == i
                item_coords = voxel_coords[sel]
                item_colors = color_feats[sel]
                item_vertices, item_faces, _ = get_mesh_batch_item(shape_mesh, i)
                item_mesh = Types.MESH(vertices=item_vertices.unsqueeze(0), faces=item_faces.unsqueeze(0))
                if item_coords.shape[0] == 0:
                    painted = paint_mesh_default_colors(item_mesh)
                else:
                    painted = paint_mesh_with_voxels(item_mesh, item_coords, item_colors, resolution=resolution)
                out_verts.append(painted.vertices.squeeze(0))
                out_faces.append(painted.faces.squeeze(0))
                out_colors.append(painted.colors.squeeze(0))
            out_mesh = pack_variable_mesh_batch(out_verts, out_faces, out_colors)
        else:
            if voxel_coords.shape[0] == 0:
                out_mesh = paint_mesh_default_colors(shape_mesh)
            else:
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
        if samples.shape[0] > 1:
            decoded_items = []
            for i in range(samples.shape[0]):
                decoded_items.append(decoder(samples[i:i + 1]) > 0)
            decoded = torch.cat(decoded_items, dim=0)
        else:
            decoded = decoder(samples) > 0
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

        coord_counts = shape_latent_512.get("coord_counts")
        decoder = vae.first_stage_model.shape_dec
        lr_resolution = 512
        target_resolution = int(target_resolution)

        if coord_counts is None:
            feats, coords_512 = flatten_batched_sparse_latent(
                shape_latent_512["samples"],
                shape_latent_512["coords"],
                coord_counts,
            )
            feats = feats.to(device)
            coords_512 = coords_512.to(device)
            slat = shape_norm(feats, coords_512)
            slat.feats = slat.feats.to(next(decoder.parameters()).dtype)
            hr_coords = decoder.upsample(slat, upsample_times=4)

            hr_resolution = target_resolution
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

        items = split_batched_sparse_latent(
            shape_latent_512["samples"],
            shape_latent_512["coords"],
            coord_counts,
        )
        decoder_dtype = next(decoder.parameters()).dtype

        final_coords_list = []
        output_resolutions = []
        output_coord_counts = []
        for batch_index, (feats_i, coords_i) in enumerate(items):
            feats_i = feats_i.to(device)
            coords_i = coords_i.to(device).clone()
            coords_i[:, 0] = 0
            slat_i = shape_norm(feats_i, coords_i)
            slat_i.feats = slat_i.feats.to(decoder_dtype)
            hr_coords_i = decoder.upsample(slat_i, upsample_times=4)

            hr_resolution = target_resolution
            while True:
                quant_coords_i = torch.cat([
                    hr_coords_i[:, :1],
                    ((hr_coords_i[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
                ], dim=1)
                final_coords_i = quant_coords_i.unique(dim=0)
                num_tokens = final_coords_i.shape[0]

                if num_tokens < max_tokens or hr_resolution <= 1024:
                    break
                hr_resolution -= 128

            final_coords_i = final_coords_i.clone()
            final_coords_i[:, 0] = batch_index
            final_coords_list.append(final_coords_i)
            output_resolutions.append(int(hr_resolution))
            output_coord_counts.append(int(final_coords_i.shape[0]))

        return IO.NodeOutput({
            "coords": torch.cat(final_coords_list, dim=0),
            "coord_counts": torch.tensor(output_coord_counts, dtype=torch.int64),
            "resolutions": torch.tensor(output_resolutions, dtype=torch.int64),
        },)

dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def run_conditioning(model, cropped_img_tensor, include_1024=True):
    model_internal = model.model
    device = comfy.model_management.intermediate_device()
    torch_device = comfy.model_management.get_torch_device()
    had_image_size = hasattr(model_internal, "image_size")
    original_image_size = getattr(model_internal, "image_size", None)

    def prepare_tensor(pil_img, size):
        resized_pil = pil_img.resize((size, size), Image.Resampling.LANCZOS)
        img_np = np.array(resized_pil).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch_device)
        return (img_t - dino_mean.to(torch_device)) / dino_std.to(torch_device)

    cond_1024 = None
    try:
        model_internal.image_size = 512
        input_512 = prepare_tensor(cropped_img_tensor, 512)
        cond_512 = model_internal(input_512, skip_norm_elementwise=True)[0]

        if include_1024:
            model_internal.image_size = 1024
            input_1024 = prepare_tensor(cropped_img_tensor, 1024)
            cond_1024 = model_internal(input_1024, skip_norm_elementwise=True)[0]
    finally:
        if not had_image_size:
            delattr(model_internal, "image_size")
        else:
            model_internal.image_size = original_image_size

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
        # Normalize to batched form so per-image conditioning loop below is uniform.
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        batch_size = image.shape[0]
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(f"Trellis2Conditioning mask batch {mask.shape[0]} does not match image batch {batch_size}")

        cond_512_list = []
        cond_1024_list = []

        for b in range(batch_size):
            item_image = image[b]
            item_mask = mask[b]

            img_np = (item_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (item_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

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

                rgba_pil = Image.fromarray(rgba_np)
                cropped_rgba = rgba_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                cropped_np = np.array(cropped_rgba).astype(np.float32) / 255.0
            else:
                import logging
                logging.warning("Mask for the image is empty. Trellis2 requires an image with a mask for the best mesh quality.")
                cropped_np = rgba_np.astype(np.float32) / 255.0

            bg_colors = {"black":[0.0, 0.0, 0.0], "gray":[0.5, 0.5, 0.5], "white":[1.0, 1.0, 1.0]}
            bg_rgb = np.array(bg_colors.get(background_color, [0.0, 0.0, 0.0]), dtype=np.float32)

            fg = cropped_np[:, :, :3]
            alpha_float = cropped_np[:, :, 3:4]
            composite_np = fg * alpha_float + bg_rgb * (1.0 - alpha_float)

            # to match trellis2 code (quantize -> dequantize)
            composite_uint8 = (composite_np * 255.0).round().clip(0, 255).astype(np.uint8)

            cropped_pil = Image.fromarray(composite_uint8)

            item_conditioning = run_conditioning(clip_vision_model, cropped_pil, include_1024=True)
            cond_512_list.append(item_conditioning["cond_512"])
            cond_1024_list.append(item_conditioning["cond_1024"])

        cond_512_batched = torch.cat(cond_512_list, dim=0)
        cond_1024_batched = torch.cat(cond_1024_list, dim=0)
        neg_cond_batched = torch.zeros_like(cond_512_batched)
        neg_embeds_batched = torch.zeros_like(cond_1024_batched)

        positive = [[cond_512_batched, {"embeds": cond_1024_batched}]]
        negative = [[neg_cond_batched, {"embeds": neg_embeds_batched}]]
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
        coord_counts = None
        coord_resolutions = None

        if hasattr(structure_or_coords, "data") and structure_or_coords.data.ndim == 4:
            decoded = structure_or_coords.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
            is_512_pass = True

        elif isinstance(structure_or_coords, dict):
            coords = structure_or_coords["coords"].int()
            coord_counts = structure_or_coords.get("coord_counts")
            coord_resolutions = structure_or_coords.get("resolutions")
            is_512_pass = False

        elif isinstance(structure_or_coords, torch.Tensor) and structure_or_coords.ndim == 2:
            coords = structure_or_coords.int()
            is_512_pass = False

        else:
            raise ValueError(f"Invalid input to EmptyShapeLatent: {type(structure_or_coords)}")
        in_channels = 32
        batch_size, inferred_coord_counts, max_tokens = infer_batched_coord_layout(coords)
        if coord_counts is not None:
            coord_counts = coord_counts.to(dtype=torch.int64, device=coords.device)
            if coord_counts.shape != inferred_coord_counts.shape or not torch.equal(coord_counts, inferred_coord_counts):
                raise ValueError(
                    f"Trellis2 coord_counts metadata {coord_counts.tolist()} does not match coords layout {inferred_coord_counts.tolist()}"
                )
        else:
            coord_counts = inferred_coord_counts
        if batch_size == 1:
            coord_counts = None
            latent = torch.randn(1, in_channels, coords.shape[0], 1)
        else:
            latent = torch.zeros(batch_size, in_channels, max_tokens, 1)
            base_state = torch.random.get_rng_state()
            for i in range(batch_size):
                count = int(coord_counts[i].item())
                generator = torch.Generator(device="cpu")
                generator.set_state(base_state.clone())
                latent_i = torch.randn(1, in_channels, count, 1, generator=generator)
                latent[i, :, :count] = latent_i[0]
        if coord_counts is not None:
            latent.trellis_coord_counts = coord_counts.clone()
        model = model.clone()
        model.model_options = model.model_options.copy()
        if "transformer_options" in model.model_options:
            model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
        else:
            model.model_options["transformer_options"] = {}

        model.model_options["transformer_options"]["coords"] = coords
        if coord_counts is not None:
            model.model_options["transformer_options"]["coord_counts"] = coord_counts
        if coord_resolutions is not None:
            model.model_options["transformer_options"]["coord_resolutions"] = coord_resolutions
        if is_512_pass:
            model.model_options["transformer_options"]["generation_mode"] = "shape_generation_512"
        else:
            model.model_options["transformer_options"]["generation_mode"] = "shape_generation"
        output = {"samples": latent, "coords": coords, "type": "trellis2"}
        if coord_counts is not None:
            output["coord_counts"] = coord_counts
            if coord_resolutions is not None:
                output["coord_resolutions"] = coord_resolutions
            output["batch_index"] = [0] * batch_size
        return IO.NodeOutput(output, model)

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
        coord_counts = None
        if hasattr(structure_or_coords, "data") and structure_or_coords.data.ndim == 4:
            decoded = structure_or_coords.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()

        elif isinstance(structure_or_coords, dict):
            coords = structure_or_coords["coords"].int()
            coord_counts = structure_or_coords.get("coord_counts")

        elif isinstance(structure_or_coords, torch.Tensor) and structure_or_coords.ndim == 2:
            coords = structure_or_coords.int()

        shape_latent = shape_latent["samples"]
        batch_size, inferred_coord_counts, max_tokens = infer_batched_coord_layout(coords)
        if coord_counts is not None:
            coord_counts = coord_counts.to(dtype=torch.int64, device=coords.device)
            if coord_counts.shape != inferred_coord_counts.shape or not torch.equal(coord_counts, inferred_coord_counts):
                raise ValueError(
                    f"Trellis2 coord_counts metadata {coord_counts.tolist()} does not match coords layout {inferred_coord_counts.tolist()}"
                )
        else:
            coord_counts = inferred_coord_counts
        if shape_latent.ndim == 4:
            if shape_latent.shape[0] != batch_size:
                raise ValueError(
                    f"shape_latent batch {shape_latent.shape[0]} doesn't match coords batch {batch_size}"
                )
            shape_latent = shape_latent.squeeze(-1).transpose(1, 2)
            if shape_latent.shape[1] < max_tokens:
                raise ValueError(
                    f"shape_latent tokens {shape_latent.shape[1]} can't cover coords max tokens {max_tokens}"
                )

        if batch_size == 1:
            coord_counts = None
            latent = torch.randn(1, channels, coords.shape[0], 1)
        else:
            latent = torch.zeros(batch_size, channels, max_tokens, 1)
            base_state = torch.random.get_rng_state()
            for i in range(batch_size):
                count = int(coord_counts[i].item())
                generator = torch.Generator(device="cpu")
                generator.set_state(base_state.clone())
                latent_i = torch.randn(1, channels, count, 1, generator=generator)
                latent[i, :, :count] = latent_i[0]
        if coord_counts is not None:
            latent.trellis_coord_counts = coord_counts.clone()
        model = model.clone()
        model.model_options = model.model_options.copy()
        if "transformer_options" in model.model_options:
            model.model_options["transformer_options"] = model.model_options["transformer_options"].copy()
        else:
            model.model_options["transformer_options"] = {}

        model.model_options["transformer_options"]["coords"] = coords
        if coord_counts is not None:
            model.model_options["transformer_options"]["coord_counts"] = coord_counts
        model.model_options["transformer_options"]["generation_mode"] = "texture_generation"
        model.model_options["transformer_options"]["shape_slat"] = shape_latent
        output = {"samples": latent, "coords": coords, "type": "trellis2"}
        if coord_counts is not None:
            output["coord_counts"] = coord_counts
            output["batch_index"] = [0] * batch_size
        return IO.NodeOutput(output, model)


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
        latent = torch.randn(1, in_channels, resolution, resolution, resolution).repeat(batch_size, 1, 1, 1, 1)
        output = {"samples": latent, "type": "trellis2"}
        if batch_size > 1:
            output["batch_index"] = [0] * batch_size
        return IO.NodeOutput(output)

def simplify_fn(vertices, faces, colors=None, target=100000):
    if vertices.ndim == 3:
        v_list, f_list, c_list = [], [], []
        for i in range(vertices.shape[0]):
            c_in = colors[i] if colors is not None else None
            v_i, f_i, c_i = simplify_fn(vertices[i], faces[i], c_in, target)
            v_list.append(v_i)
            f_list.append(f_i)
            if c_i is not None:
                c_list.append(c_i)

        c_out = torch.stack(c_list) if len(c_list) > 0 else None
        return torch.stack(v_list), torch.stack(f_list), c_out

    if faces.shape[0] <= target:
        return vertices, faces, colors

    device = vertices.device
    target_v = max(target / 4.0, 1.0)

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

    new_colors = None
    if colors is not None:
        new_colors = torch.zeros((num_cells, colors.shape[1]), dtype=colors.dtype, device=device)
        new_colors.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, colors.shape[1]), colors)
        new_colors = new_colors / counts.clamp(min=1)

    new_faces = inverse_indices[faces]
    valid_mask = (new_faces[:, 0] != new_faces[:, 1]) & \
                 (new_faces[:, 1] != new_faces[:, 2]) & \
                 (new_faces[:, 2] != new_faces[:, 0])
    new_faces = new_faces[valid_mask]

    unique_face_indices, inv_face = torch.unique(new_faces.reshape(-1), return_inverse=True)
    final_vertices = new_vertices[unique_face_indices]
    final_faces = inv_face.reshape(-1, 3)

    # assign colors
    final_colors = new_colors[unique_face_indices] if new_colors is not None else None

    return final_vertices, final_faces, final_colors

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
        if hasattr(mesh, "vertex_counts"):
            out_verts, out_faces, out_colors = [], [], []
            for i in range(mesh.vertices.shape[0]):
                v_i, f_i, c_i = get_mesh_batch_item(mesh, i)
                actual_face_count = f_i.shape[0]
                if fill_holes_perimeter > 0:
                    v_i, f_i = fill_holes_fn(v_i, f_i, max_perimeter=fill_holes_perimeter)
                if simplify > 0 and actual_face_count > simplify:
                    v_i, f_i, c_i = simplify_fn(v_i, f_i, target=simplify, colors=c_i)
                v_i, f_i = make_double_sided(v_i, f_i)
                out_verts.append(v_i)
                out_faces.append(f_i)
                if c_i is not None:
                    out_colors.append(c_i)
            out_mesh = pack_variable_mesh_batch(out_verts, out_faces, out_colors if len(out_colors) == len(out_verts) else None)
            return IO.NodeOutput(out_mesh)
        verts, faces = mesh.vertices, mesh.faces
        colors = None
        if hasattr(mesh, "colors"):
            colors = mesh.colors

        actual_face_count = faces.shape[1] if faces.ndim == 3 else faces.shape[0]
        if fill_holes_perimeter > 0:
            verts, faces = fill_holes_fn(verts, faces, max_perimeter=fill_holes_perimeter)

        if simplify > 0 and actual_face_count > simplify:
            verts, faces, colors = simplify_fn(verts, faces, target=simplify, colors=colors)

        verts, faces = make_double_sided(verts, faces)

        mesh = type(mesh)(vertices=verts, faces=faces)
        mesh.vertices = verts
        mesh.faces = faces
        if colors is not None:
            mesh.colors = colors
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

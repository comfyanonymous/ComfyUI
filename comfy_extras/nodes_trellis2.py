from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types, io
from comfy.ldm.trellis2.vae import SparseTensor
from comfy.ldm.trellis2.model import _build_proj_transform_matrix, _project_points_to_image
from comfy.ldm.trellis2.naf.model import build_naf_from_state_dict
from comfy_extras.nodes_mesh_postprocess import pack_variable_mesh_batch
import comfy.model_management
import comfy.utils
import folder_paths
from PIL import Image
import logging
import numpy as np
import math
import torch

ShapeSubdivides = io.Custom("SHAPE_SUBDIVIDES")
Pixal3DProjPack = io.Custom("PIXAL3D_PROJ_PACK")
NAFModel = io.Custom("NAF_MODEL")


# Pixal3D trains in a 90°-X-rotated grid frame (F_p). We un-rotate decoder outputs for
# user-facing previews/meshes, then re-rotate before feeding coords back to the shape DiT.

def _pixal3d_unrotate_voxel_data(data: torch.Tensor) -> torch.Tensor:
    if data.ndim == 4:
        return data.flip(-1).permute(0, 1, 3, 2).contiguous()
    if data.ndim == 5:
        return data.flip(-1).permute(0, 1, 2, 4, 3).contiguous()
    raise ValueError(f"unexpected voxel shape {tuple(data.shape)}")


def _pixal3d_rerotate_voxel_data(data: torch.Tensor) -> torch.Tensor:
    if data.ndim == 4:
        return data.permute(0, 1, 3, 2).flip(-1).contiguous()
    if data.ndim == 5:
        return data.permute(0, 1, 2, 4, 3).flip(-1).contiguous()
    raise ValueError(f"unexpected voxel shape {tuple(data.shape)}")


def _pixal3d_unrotate_vertices(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.numel() == 0:
        return vertices
    x, y, z = vertices.unbind(-1)
    return torch.stack([-x, y, -z], dim=-1).contiguous()


def _pixal3d_unrotate_sparse_coords(coords: torch.Tensor, resolution: int) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    R1 = resolution - 1
    if coords.shape[-1] == 4:
        b, i, j, k = coords.unbind(-1)
        return torch.stack([b, R1 - i, j, R1 - k], dim=-1).contiguous()
    if coords.shape[-1] == 3:
        i, j, k = coords.unbind(-1)
        return torch.stack([R1 - i, j, R1 - k], dim=-1).contiguous()
    raise ValueError(f"unexpected coord shape {tuple(coords.shape)}")

def prepare_trellis_vae_for_decode(vae, sample_shape):
    memory_required = vae.memory_used_decode(sample_shape, vae.vae_dtype)
    if len(sample_shape) == 5:
        memory_required *= max(1, int(sample_shape[4]))
    memory_required = max(1, int(memory_required))
    device = comfy.model_management.get_torch_device()
    comfy.model_management.load_models_gpu(
        [vae.patcher],
        memory_required=memory_required,
        force_full_load=getattr(vae, "disable_offload", False),
    )
    free_memory = vae.patcher.get_free_memory(device)
    batch_number = max(1, int(free_memory / memory_required))
    return batch_number

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
    if (batch_ids < 0).any():
        raise ValueError(f"Trellis2 batch ids must be non-negative, got {batch_ids.unique(sorted=True).tolist()}")
    batch_size = int(batch_ids.max().item()) + 1
    counts = torch.bincount(batch_ids, minlength=batch_size)

    if (counts == 0).any():
        raise ValueError(f"Non-contiguous Trellis2 batch ids in coords: {batch_ids.unique(sorted=True).tolist()}")

    max_tokens = int(counts.max().item())
    return batch_size, counts, max_tokens


def split_batched_coords(coords, coord_counts):
    if coord_counts.ndim != 1:
        raise ValueError(f"Trellis2 coord_counts must be 1D, got shape {tuple(coord_counts.shape)}")
    if (coord_counts < 0).any():
        raise ValueError(f"Trellis2 coord_counts must be non-negative, got {coord_counts.tolist()}")
    if int(coord_counts.sum().item()) != coords.shape[0]:
        raise ValueError(
            f"Trellis2 coord_counts total {int(coord_counts.sum().item())} does not match coords rows {coords.shape[0]}"
        )

    batch_ids = coords[:, 0].to(torch.int64)
    order = torch.argsort(batch_ids, stable=True)
    sorted_coords = coords.index_select(0, order)
    sorted_batch_ids = batch_ids.index_select(0, order)

    offsets = coord_counts.cumsum(0) - coord_counts
    items = []
    for i in range(coord_counts.shape[0]):
        count = int(coord_counts[i].item())
        start = int(offsets[i].item())
        coords_i = sorted_coords[start:start + count]
        ids_i = sorted_batch_ids[start:start + count]
        if coords_i.shape[0] != count or not torch.all(ids_i == i):
            raise ValueError(f"Trellis2 coords rows for batch {i} expected {count}, got {coords_i.shape[0]}")
        items.append(coords_i)
    return items

def flatten_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return samples.reshape(-1, samples.shape[-1]), coords

    coords_items = split_batched_coords(coords, coord_counts)
    feat_list = []
    coord_list = []
    for i, coords_i in enumerate(coords_items):
        count = int(coord_counts[i].item())
        feat_list.append(samples[i, :count])
        coord_list.append(coords_i)

    return torch.cat(feat_list, dim=0), torch.cat(coord_list, dim=0)


def split_batched_sparse_latent(samples, coords, coord_counts):
    samples = samples.squeeze(-1).transpose(1, 2)
    if coord_counts is None:
        return [(samples.reshape(-1, samples.shape[-1]), coords)]

    coords_items = split_batched_coords(coords, coord_counts)
    items = []
    for i, coords_i in enumerate(coords_items):
        count = int(coord_counts[i].item())
        items.append((samples[i, :count], coords_i))
    return items

class VaeDecodeShapeTrellis(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VaeDecodeShapeTrellis",
            category="latent/3d",
            inputs=[
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
            ],
            outputs=[
                IO.Mesh.Output("mesh"),
                ShapeSubdivides.Output(display_name = "shape_subdivides"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae):

        resolution = int(vae.first_stage_model.resolution.item())
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        prepare_trellis_vae_for_decode(vae, sample_tensor.shape)
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")
        pixal3d_mode = samples.get("model_options", {}).get("proj_feat_pack") is not None

        samples = samples["samples"]
        if coord_counts is None:
            samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
            samples = shape_norm(samples.to(device), coords.to(device))
            mesh, subs = trellis_vae.decode_shape_slat(samples, resolution)
        else:
            split_items = split_batched_sparse_latent(samples, coords, coord_counts)
            mesh = []
            subs_per_sample = []
            for feats_i, coords_i in split_items:
                coords_i = coords_i.to(device).clone()
                coords_i[:, 0] = 0
                sample_i = shape_norm(feats_i.to(device), coords_i)
                mesh_i, subs_i = trellis_vae.decode_shape_slat(sample_i, resolution)
                mesh.append(mesh_i[0])
                subs_per_sample.append(subs_i)

            subs = []
            for stage_index in range(len(subs_per_sample[0])):
                stage_tensors = [sample_subs[stage_index] for sample_subs in subs_per_sample]
                feats_list = [stage_tensor.feats for stage_tensor in stage_tensors]
                coords_list = [stage_tensor.coords for stage_tensor in stage_tensors]
                subs.append(SparseTensor.from_tensor_list(feats_list, coords_list))

        if pixal3d_mode:
            for m in mesh:
                m.vertices = _pixal3d_unrotate_vertices(m.vertices)

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
                IO.Latent.Input("samples"),
                IO.Vae.Input("vae"),
                ShapeSubdivides.Input("shape_subdivides",
                                 tooltip=(
                                    "Shape information used to guide higher-detail reconstruction during decoding. "
                                    "Helps preserve structure consistency at higher resolutions."
                                 )),
            ],
            outputs=[
                IO.Voxel.Output("voxel_colors"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, shape_subdivides):
        sample_tensor = samples["samples"]
        device = comfy.model_management.get_torch_device()
        coords = samples["coords"]
        prepare_trellis_vae_for_decode(vae, sample_tensor.shape)
        trellis_vae = vae.first_stage_model
        coord_counts = samples.get("coord_counts")
        pixal3d_mode = samples.get("model_options", {}).get("proj_feat_pack") is not None

        samples = samples["samples"]
        samples, coords = flatten_batched_sparse_latent(samples, coords, coord_counts)
        samples = samples.to(device)
        std = tex_slat_normalization["std"].to(samples)
        mean = tex_slat_normalization["mean"].to(samples)
        samples = SparseTensor(feats = samples, coords=coords.to(device))
        samples = samples * std + mean

        voxel = trellis_vae.decode_tex_slat(samples, shape_subdivides)
        color_feats = voxel.feats[:, :3]
        voxel_coords = voxel.coords#[:, 1:]

        if voxel_coords.numel() > 0 and voxel_coords.shape[-1] >= 3:
            spatial = voxel_coords[:, -3:] if voxel_coords.shape[-1] == 4 else voxel_coords
            max_idx = int(spatial.max().item()) + 1
            tex_resolution = next((r for r in (256, 512, 1024, 1536, 2048) if r >= max_idx), max_idx)
        else:
            tex_resolution = 1024

        if pixal3d_mode:
            voxel_coords = _pixal3d_unrotate_sparse_coords(voxel_coords, resolution=tex_resolution)

        voxel = Types.VOXEL(voxel_coords, color_feats, tex_resolution)
        return IO.NodeOutput(voxel)

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
                IO.Voxel.Output("voxel"),
            ]
        )

    @classmethod
    def execute(cls, samples, vae, resolution):
        resolution = int(resolution)
        sample_tensor = samples["samples"]
        sample_tensor = sample_tensor[:, :8]
        batch_number = prepare_trellis_vae_for_decode(vae, sample_tensor.shape)
        decoder = vae.first_stage_model.struct_dec
        load_device = comfy.model_management.get_torch_device()
        decoded_batches = []
        for start in range(0, sample_tensor.shape[0], batch_number):
            sample_chunk = sample_tensor[start:start + batch_number].to(load_device)
            decoded_batches.append(decoder(sample_chunk) > 0)
        decoded = torch.cat(decoded_batches, dim=0)
        current_res = decoded.shape[2]

        if current_res != resolution:
            ratio = current_res // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        voxel_data = decoded.squeeze(1).float()
        if samples.get("model_options", {}).get("proj_feat_pack") is not None:
            voxel_data = _pixal3d_unrotate_voxel_data(voxel_data)
        out = Types.VOXEL(voxel_data)
        return IO.NodeOutput(out)

class Trellis2UpsampleCascade(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2UpsampleCascade",
            category="latent/3d",
            display_name="Trellis2 Upsample Cascade",
            description="Upsamples low-resolution Trellis2 shape latents into higher resolution coordinates while respecting the maximum token budget.",
            inputs=[
                IO.Latent.Input("shape_latent"),
                IO.Vae.Input("vae"),
                IO.Combo.Input("target_resolution", options=["1024", "1536"], default="1024", tooltip="Controls output detail level for upsampling."),
                IO.Int.Input("max_tokens", default=49152, min=1024, max=100000,
                             tooltip=(
                                "Maximum number of output elements (coordinates) allowed after upsampling. "
                                "Used to limit memory usage and control mesh density."
                            ))
            ],
            outputs=[
                IO.Voxel.Output(
                "high_res_voxel",
                tooltip=(
                    "High-resolution sparse coordinates produced after cascade upsampling. "
                    "Represents the refined 3D structure at target resolution."
                )
            )
            ]
        )

    @classmethod
    def execute(cls, shape_latent, vae, target_resolution, max_tokens):
        shape_latent_512 = shape_latent
        device = comfy.model_management.get_torch_device()
        prepare_trellis_vae_for_decode(vae, shape_latent_512["samples"].shape)

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

        sample_hr_coords = []
        for feats_i, coords_i in items:
            feats_i = feats_i.to(device)
            coords_i = coords_i.to(device).clone()
            coords_i[:, 0] = 0
            slat_i = shape_norm(feats_i, coords_i)
            slat_i.feats = slat_i.feats.to(decoder_dtype)
            sample_hr_coords.append(decoder.upsample(slat_i, upsample_times=4))

        hr_resolution = target_resolution
        while True:
            exceeds_limit = False
            for hr_coords_i in sample_hr_coords:
                quant_coords_i = torch.cat([
                    hr_coords_i[:, :1],
                    ((hr_coords_i[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
                ], dim=1)
                if quant_coords_i.unique(dim=0).shape[0] >= max_tokens:
                    exceeds_limit = True
                    break
            if not exceeds_limit or hr_resolution <= 1024:
                break
            hr_resolution -= 128

        final_coords_list = []
        output_coord_counts = []
        for sample_offset, hr_coords_i in enumerate(sample_hr_coords):
            quant_coords_i = torch.cat([
                hr_coords_i[:, :1],
                ((hr_coords_i[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            final_coords_i = quant_coords_i.unique(dim=0)
            final_coords_i = final_coords_i.clone()
            final_coords_i[:, 0] = sample_offset
            final_coords_list.append(final_coords_i)
            output_coord_counts.append(int(final_coords_i.shape[0]))

        coords = torch.cat(final_coords_list, dim=0)
        output = Types.VOXEL(coords)
        output.coord_counts = torch.tensor(output_coord_counts, dtype=torch.int64)
        output.resolutions = torch.full((len(final_coords_list),), int(hr_resolution), dtype=torch.int64)
        output.upsampled = True

        return IO.NodeOutput(output,)

dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def run_conditioning(model, cropped_img_tensor, include_1024=True):
    model_internal = model.model
    device = comfy.model_management.intermediate_device()
    torch_device = comfy.model_management.get_torch_device()

    def prepare_tensor(pil_img, size):
        resized_pil = pil_img.resize((size, size), Image.Resampling.LANCZOS)
        img_np = np.array(resized_pil).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch_device)
        return (img_t - dino_mean.to(torch_device)) / dino_std.to(torch_device)

    model_internal.image_size = 512
    input_512 = prepare_tensor(cropped_img_tensor, 512)
    cond_512 = model_internal(input_512, skip_norm_elementwise=True)[0]

    cond_1024 = None
    if include_1024:
        model_internal.image_size = 1024
        input_1024 = prepare_tensor(cropped_img_tensor, 1024)
        cond_1024 = model_internal(input_1024, skip_norm_elementwise=True)[0]

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
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    @classmethod
    def execute(cls, clip_vision_model, image, mask) -> IO.NodeOutput:
        # Normalize to batched form so per-image conditioning loop below is uniform.
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim == 4:
            if image.shape[1] in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]:
                image = image.permute(0, 2, 3, 1)

        # normalize mask to standard [B, H, W] (handling 2D, 3D, and 4D variants)
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = mask.squeeze(1)
            elif mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            else:
                mask = mask[:, :, :, 0] # take first channel as fallback

        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask.squeeze(-1).unsqueeze(0)
        elif mask.ndim == 2:
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
            item_mask = mask[b] if mask.size(0) > 1 else mask[0]

            img_np = (item_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            mask_np = (item_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # Ensure img_np is either 2D (grayscale) or 3D (RGB/RGBA)
            if img_np.ndim == 3 and img_np.shape[-1] == 1:
                img_np = img_np.squeeze(-1)

            mask_np = mask_np.squeeze()

            # detect inverted mask
            border_pixels = np.concatenate([
                mask_np[0, :], mask_np[-1, :], mask_np[:, 0], mask_np[:, -1]
            ])
            if np.mean(border_pixels) > 127:
                mask_np = 255 - mask_np

            mask_np[mask_np < 35] = 0

            border_shave = 4
            mask_np[:border_shave, :] = 0
            mask_np[-border_shave:, :] = 0
            mask_np[:, :border_shave] = 0
            mask_np[:, -border_shave:] = 0

            pil_img = Image.fromarray(img_np)
            pil_mask = Image.fromarray(mask_np)

            max_size = max(pil_img.size)
            scale = min(1.0, 1024 / max_size)
            if scale < 1.0:
                new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)

            rgba_np = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
            rgba_np[:, :, :3] = np.array(pil_img.convert("RGB"))
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
                logging.warning("Mask for the image is empty. Trellis2 requires an image with a mask for the best mesh quality.")
                cropped_np = rgba_np.astype(np.float32) / 255.0

            bg_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            fg = cropped_np[:, :, :3]
            alpha_float = cropped_np[:, :, 3:4]
            composite_np = fg * alpha_float + bg_rgb * (1.0 - alpha_float)

            # Keep the image as 4-channel RGBA to force TRELLIS to bypass its internal background remover
            rgb_uint8 = (composite_np * 255.0).round().clip(0, 255).astype(np.uint8)
            alpha_uint8 = (alpha_float.squeeze(-1) * 255.0).round().clip(0, 255).astype(np.uint8)

            rgba_composite = np.zeros((cropped_np.shape[0], cropped_np.shape[1], 4), dtype=np.uint8)
            rgba_composite[:, :, :3] = rgb_uint8
            rgba_composite[:, :, 3] = alpha_uint8

            cropped_pil = Image.fromarray(rgba_composite, mode="RGBA")

            # Convert to RGB to ensure the CLIP/DINO model receives a 3-channel image
            item_conditioning = run_conditioning(clip_vision_model, cropped_pil.convert("RGB"), include_1024=True)
            cond_512_list.append(item_conditioning["cond_512"])
            cond_1024_list.append(item_conditioning["cond_1024"])

        cond_512_batched = torch.cat(cond_512_list, dim=0)
        cond_1024_batched = torch.cat(cond_1024_list, dim=0)
        neg_cond_batched = torch.zeros_like(cond_512_batched)
        neg_embeds_batched = torch.zeros_like(cond_1024_batched)

        positive = [[cond_512_batched, {"embeds": cond_1024_batched}]]
        negative = [[neg_cond_batched, {"embeds": neg_embeds_batched}]]
        return IO.NodeOutput(positive, negative)

class EmptyTrellis2ShapeLatent(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2ShapeLatent",
            category="latent/3d",
            inputs=[
                IO.Voxel.Input(
                    "voxel",
                    tooltip=(
                        "Shape structure input. Accepts either a voxel structure "
                        "or upsampled voxel coordinates from a previous cascade stage."
                    )
            ),
                Pixal3DProjPack.Input(
                    "proj_feat_pack",
                    optional=True,
                    tooltip="Pixal3D pixel-aligned projection pack from Pixal3DConditioning. Leave empty for vanilla Trellis2.",
                ),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, proj_feat_pack=None):
        is_512_pass = False
        coord_resolution = None
        upsampled = hasattr(voxel, "upsampled")
        if upsampled:
            if hasattr(voxel, "resolutions") and voxel.resolutions is not None:
                coord_resolution = int(voxel.resolutions[0].item()) // 16
            voxel = voxel.data

        if not upsampled:
            voxel_data = voxel.data
            if proj_feat_pack is not None:
                voxel_data = _pixal3d_rerotate_voxel_data(voxel_data)
            decoded = voxel_data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
            is_512_pass = True
            coord_resolution = int(decoded.shape[-1])

        else:
            coords = voxel.int()

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)
        in_channels = 32
        # image like format
        latent = torch.zeros(batch_size, in_channels, max_tokens, 1)

        if is_512_pass:
            generation_mode = "shape_generation_512"
        else:
            generation_mode = "shape_generation"
        model_options = {"generation_mode": generation_mode, "coords": coords, "coord_counts": counts}
        if coord_resolution is not None:
            model_options["coord_resolution"] = coord_resolution
        if proj_feat_pack is not None:
            model_options["proj_feat_pack"] = proj_feat_pack
        return IO.NodeOutput({"samples": latent, "coords": coords, "coord_counts": counts, "type": "trellis2",
                              "model_options": model_options})

class EmptyTrellis2LatentTexture(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2LatentTexture",
            category="latent/3d",
            inputs=[
                IO.Voxel.Input(
                    "voxel",
                    tooltip=(
                        "Shape structure input. Accepts either a voxel structure "
                        "or upsampled voxel coordinates from a previous cascade stage."
                    )
                ),
                IO.Latent.Input("shape_latent"),
                Pixal3DProjPack.Input(
                    "proj_feat_pack",
                    optional=True,
                    tooltip="Pixal3D pixel-aligned projection pack from Pixal3DConditioning. Leave empty for vanilla Trellis2.",
                ),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, shape_latent, proj_feat_pack=None):
        channels = 32
        coord_resolution = None
        upsampled = hasattr(voxel, "upsampled")
        if upsampled:
            if hasattr(voxel, "resolutions") and voxel.resolutions is not None:
                coord_resolution = int(voxel.resolutions[0].item()) // 16
            voxel = voxel.data

        if not upsampled:
            voxel_data = voxel.data
            if proj_feat_pack is not None:
                voxel_data = _pixal3d_rerotate_voxel_data(voxel_data)
            decoded = voxel_data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
            coord_resolution = int(decoded.shape[-1])
        else:
            coords = voxel.int()

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)

        shape_latent = shape_latent["samples"]
        if shape_latent.ndim == 4:
            shape_latent = shape_latent.squeeze(-1).transpose(1, 2).reshape(-1, channels)

        latent = torch.zeros(batch_size, channels, max_tokens, 1)
        model_options = {"generation_mode": "texture_generation", "coords": coords, "coord_counts": counts, "shape_slat": shape_latent}
        if coord_resolution is not None:
            model_options["coord_resolution"] = coord_resolution
        if proj_feat_pack is not None:
            model_options["proj_feat_pack"] = proj_feat_pack
        return IO.NodeOutput({"samples": latent, "type": "trellis2", "coords": coords, "coord_counts": counts,
                              "model_options": model_options})


class EmptyTrellis2LatentStructure(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2LatentStructure",
            category="latent/3d",
            inputs=[
                IO.Int.Input("batch_size", default=1, min=1, max=4096, tooltip="The number of latent images in the batch."),
                Pixal3DProjPack.Input(
                    "proj_feat_pack",
                    optional=True,
                    tooltip="Pixal3D pixel-aligned projection pack. Leave empty for vanilla Trellis2.",
                ),
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )
    @classmethod
    def execute(cls, batch_size, proj_feat_pack=None):
        # Trellis2.forward slices x[:, :8] and pads out to 32; KSampler residual math
        # needs the empty latent to match latent_format (32-channel).
        in_channels = 32
        resolution = 16
        latent = torch.zeros(batch_size, in_channels, resolution, resolution, resolution)
        output = {
            "samples": latent,
            "type": "trellis2",
        }
        if proj_feat_pack is not None:
            output["model_options"] = {"proj_feat_pack": proj_feat_pack}
        return IO.NodeOutput(output)

def _dinov3_patches_to_2d(tokens, image_size, patch_size=16):
    h_p = w_p = image_size // patch_size
    n_patches = h_p * w_p
    n_reg = tokens.shape[1] - 1 - n_patches
    if n_reg < 0 or tokens.shape[1] != 1 + n_reg + n_patches:
        raise ValueError(
            f"_dinov3_patches_to_2d: got {tokens.shape[1]} tokens, expected "
            f"1 (CLS) + N_reg + {h_p}*{w_p}={n_patches} patches at image_size={image_size}, "
            f"patch_size={patch_size}. Inferred N_reg={n_reg} which is invalid."
        )
    start = 1 + n_reg
    patches = tokens[:, start:start + n_patches]
    return patches.transpose(1, 2).reshape(tokens.shape[0], -1, h_p, w_p).contiguous()


def _fov_from_moge_intrinsics(moge_intrinsics: torch.Tensor) -> float:
    fx = moge_intrinsics[..., 0, 0].float()
    fov = 2.0 * torch.atan(0.5 / fx.clamp(min=1e-4))
    return float(fov.mean().item())


def _run_dinov3_with_patches(model, cropped_pil, image_size):
    # Pixal3D's cross-attn was trained against CLS + registers only (~5 tokens), not the
    # full patch grid. The patch grid goes to the proj branch via patches_2d.
    model_internal = model.model
    torch_device = comfy.model_management.get_torch_device()
    resized = cropped_pil.resize((image_size, image_size), Image.Resampling.LANCZOS)
    img_np = np.array(resized).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(torch_device)
    img_t = (img_t - dino_mean.to(torch_device)) / dino_std.to(torch_device)
    model_internal.image_size = image_size
    tokens = model_internal(img_t, skip_norm_elementwise=True)[0]
    patches = _dinov3_patches_to_2d(tokens, image_size)
    h_p = w_p = image_size // 16
    n_reg = tokens.shape[1] - 1 - h_p * w_p
    global_tokens = tokens[:, :1 + n_reg]
    return {"tokens": global_tokens, "patches_2d": patches}


def _crop_image_with_mask(item_image, item_mask, max_image_size=1024):
    img_np = (item_image.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mask_np = (item_mask.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    pil_mask = Image.fromarray(mask_np)
    max_size = max(pil_img.size)
    scale = min(1.0, max_image_size / max_size)
    if scale < 1.0:
        new_w, new_h = int(pil_img.width * scale), int(pil_img.height * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
    scene_size = (pil_img.width, pil_img.height)
    rgba_np = np.zeros((pil_img.height, pil_img.width, 4), dtype=np.uint8)
    rgba_np[:, :, :3] = np.array(pil_img)
    rgba_np[:, :, 3] = np.array(pil_mask)
    alpha = rgba_np[:, :, 3]
    bbox_coords = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_coords) > 0:
        y_min, x_min = np.min(bbox_coords[:, 0]), np.min(bbox_coords[:, 1])
        y_max, x_max = np.max(bbox_coords[:, 0]), np.max(bbox_coords[:, 1])
        center_y, center_x = (y_min + y_max) / 2.0, (x_min + x_max) / 2.0
        # Upstream pads the bbox by 10% — encoders were trained with that breathing room.
        size = max(y_max - y_min, x_max - x_min)
        size = int(size * 1.1)
        half = size // 2
        crop_x1 = int(center_x - half)
        crop_y1 = int(center_y - half)
        crop_x2 = crop_x1 + 2 * half
        crop_y2 = crop_y1 + 2 * half
        crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
        rgba_pil = Image.fromarray(rgba_np)
        cropped_rgba = rgba_pil.crop(crop_bbox)
        cropped_np = np.array(cropped_rgba).astype(np.float32) / 255.0
    else:
        logging.warning("Mask for the image is empty. Pixal3D requires a clean foreground mask.")
        cropped_np = rgba_np.astype(np.float32) / 255.0
        crop_bbox = (0, 0, scene_size[0], scene_size[1])
    fg = cropped_np[:, :, :3]
    alpha_float = cropped_np[:, :, 3:4]
    composite_np = fg * alpha_float
    composite_uint8 = (composite_np * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(composite_uint8), crop_bbox, scene_size


class Pixal3DConditioning(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Pixal3DConditioning",
            category="conditioning/video_models",
            inputs=[
                IO.ClipVision.Input("clip_vision_model", tooltip="DINOv3 ViT-L/16 ClipVision."),
                IO.Image.Input("image"),
                IO.Mask.Input("mask"),
                IO.Float.Input(
                    "camera_angle_x", default=0.2, min=0.0175, max=2.9671, step=0.001,
                    tooltip="Horizontal FOV in radians (upstream demo default 0.2). "
                            "Overridden by moge_geometry if connected.",
                ),
                IO.Float.Input(
                    "mesh_scale", default=1.0, min=0.1, max=4.0, step=0.01,
                    tooltip="Mesh scale; 1.0 means unit cube.",
                ),
                IO.Float.Input(
                    "distance_override", default=0.0, min=0.0, max=10.0, step=0.001,
                    tooltip="Override camera distance directly. 0 = auto-derive from FOV.",
                ),
                io.Custom("MOGE_GEOMETRY").Input(
                    "moge_geometry",
                    optional=True,
                    tooltip="If connected, camera_angle_x is recovered from MoGe.",
                ),
                NAFModel.Input(
                    "naf_model",
                    optional=True,
                    tooltip="Optional NAF feature upsampler. Required for shape/texture stages "
                            "to match upstream's trained feature distribution.",
                ),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
                Pixal3DProjPack.Output(display_name="proj_feat_pack"),
            ],
        )

    @classmethod
    def execute(cls, clip_vision_model, image, mask, camera_angle_x, mesh_scale,
                distance_override=0.0,
                moge_geometry=None, naf_model=None) -> IO.NodeOutput:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        batch_size = image.shape[0]
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(f"Pixal3DConditioning mask batch {mask.shape[0]} != image batch {batch_size}")

        if moge_geometry is not None and "intrinsics" in moge_geometry:
            camera_angle_x = _fov_from_moge_intrinsics(moge_geometry["intrinsics"])

        device = comfy.model_management.intermediate_device()

        cond_512_list, cond_1024_list = [], []
        patches_512_list, patches_1024_list = [], []
        cropped_pil_list = []
        crop_bbox_list, scene_size_list = [], []

        torch_device = comfy.model_management.get_torch_device()
        for b in range(batch_size):
            item_image = image[b]
            item_mask = mask[b] if mask.size(0) > 1 else mask[0]
            cropped_pil, crop_bbox, scene_size = _crop_image_with_mask(
                item_image, item_mask, max_image_size=1024)
            crop_bbox_list.append(crop_bbox)
            scene_size_list.append(scene_size)
            cropped_pil_list.append(cropped_pil)

            cond_512 = _run_dinov3_with_patches(clip_vision_model, cropped_pil, 512)
            cond_1024 = _run_dinov3_with_patches(clip_vision_model, cropped_pil, 1024)
            cond_512_list.append(cond_512["tokens"].to(device))
            cond_1024_list.append(cond_1024["tokens"].to(device))
            patches_512_list.append(cond_512["patches_2d"].to(device))
            patches_1024_list.append(cond_1024["patches_2d"].to(device))

        global_512 = torch.cat(cond_512_list, dim=0)
        global_1024 = torch.cat(cond_1024_list, dim=0)

        fm_512_dino = torch.cat(patches_512_list, dim=0)
        fm_1024_dino = torch.cat(patches_1024_list, dim=0)

        # Upstream samples the LR DINO grid AND the NAF HR grid separately at projected
        # 3D points, then cats sampled features along channels. Back-projection (in model.py)
        # mirrors that — here we just stash LR + optional HR per stage.
        # NAF targets per stage: shape_512=512, shape_1024=512, tex_1024=1024.
        def _naf_hr(lr_feat, image_pil_list, image_size, naf_target):
            if naf_model is None or naf_target is None:
                return None
            # Run NAF in the input feature dtype (typically fp16 since DINO/ClipVision
            # loads that way). The previous .float() cast doubled NAF memory by forcing
            # full fp32 — at tex_1024/target=1024 that's ~10 GB on its own. Model
            # weights need to match input dtype since PyTorch conv ops error out on
            # mixed fp16-input/fp32-weight.
            target_dtype = lr_feat.dtype
            if next(naf_model.parameters()).dtype != target_dtype:
                naf_model.to(dtype=target_dtype)
            imgs = torch.stack([
                torch.from_numpy(
                    np.array(p.resize((image_size, image_size), Image.Resampling.LANCZOS))
                       .astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                for p in image_pil_list
            ], dim=0).to(torch_device).to(target_dtype)
            
            hr = naf_model(imgs, lr_feat.to(torch_device).to(target_dtype), naf_target)
            return hr.to(device)

        hr_shape_512  = _naf_hr(fm_512_dino,  cropped_pil_list, 512,  (512, 512))
        hr_shape_1024 = _naf_hr(fm_1024_dino, cropped_pil_list, 1024, (512, 512))
        hr_tex_1024   = _naf_hr(fm_1024_dino, cropped_pil_list, 1024, (1024, 1024))

        # distance_from_fov: grid_point (-1, 0, 0) projects to pixel (0, image_resolution-1).
        camera_angle_x = float(camera_angle_x)
        if distance_override > 0:
            distance = float(distance_override)
        else:
            distance = 0.5 / math.tan(camera_angle_x / 2.0) / float(mesh_scale)
        cam_angle_t = torch.tensor([camera_angle_x] * batch_size, device=device, dtype=torch.float32)
        dist_t = torch.tensor([distance] * batch_size, device=device, dtype=torch.float32)
        scale_t = torch.tensor([float(mesh_scale)] * batch_size, device=device, dtype=torch.float32)
        T = _build_proj_transform_matrix(dist_t, batch_size, device=device, dtype=torch.float32)

        proj_pack = {
            "stages": {
                "ss":         {"feature_map": fm_512_dino,    "feature_map_hr": None,         "image_resolution": 512},
                "shape_512":  {"feature_map": fm_512_dino,    "feature_map_hr": hr_shape_512, "image_resolution": 512},
                "shape_1024": {"feature_map": fm_1024_dino,   "feature_map_hr": hr_shape_1024,"image_resolution": 1024},
                "tex_1024":   {"feature_map": fm_1024_dino,   "feature_map_hr": hr_tex_1024,  "image_resolution": 1024},
            },
            "transform_matrix": T,
            "camera_angle_x": cam_angle_t,
            "mesh_scale": scale_t,
            "distance": dist_t,
            "patch_size": 16,
            "crop_bboxes": crop_bbox_list,
            "scene_sizes": scene_size_list,
        }

        # global_512 → SS/shape_512 cross-attn; global_1024 → shape_1024/tex_1024
        # (Trellis2.forward swaps context↔embeds for non-structure HR stages).
        neg_global = torch.zeros_like(global_512)
        neg_embeds = torch.zeros_like(global_1024)
        positive = [[global_512, {"embeds": global_1024}]]
        negative = [[neg_global, {"embeds": neg_embeds}]]
        return IO.NodeOutput(positive, negative, proj_pack)


def _project_vertices_to_image_uv(vertices_world, transform_matrix, camera_angle_x, image_resolution):
    points = vertices_world.unsqueeze(0).float()
    T = transform_matrix.unsqueeze(0).float() if transform_matrix.ndim == 2 else transform_matrix.float()
    cam = camera_angle_x.unsqueeze(0) if camera_angle_x.ndim == 0 else camera_angle_x
    uv_pix, depth, valid = _project_points_to_image(points, T, cam.float(), image_resolution)
    uv = uv_pix.squeeze(0) / image_resolution
    return uv, depth.squeeze(0), valid.squeeze(0)


def _crop_uv_to_scene_pixels(uv_crop, crop_bbox, scene_image_size):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    px = uv_crop[:, 0] * crop_w + crop_x1
    py = uv_crop[:, 1] * crop_h + crop_y1
    W, H = scene_image_size
    return torch.stack([px.clamp(0, W - 1), py.clamp(0, H - 1)], dim=-1)


class Pixal3DAlignObject(IO.ComfyNode):
    """Pixal3D paper §3.3 Global Alignment for a single object.

    Solves (scale, translation) aligning the mesh to MoGe's per-pixel point map. Requires
    MoGe to have been computed on the same resized scene image as Pixal3DConditioning."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Pixal3DAlignObject",
            category="latent/3d",
            inputs=[
                IO.Mesh.Input("mesh"),
                Pixal3DProjPack.Input("proj_feat_pack", tooltip="The proj pack produced by Pixal3DConditioning for this object."),
                io.Custom("MOGE_GEOMETRY").Input("moge_geometry", tooltip="MoGe geometry computed on the original scene image."),
                IO.Mask.Input(
                    "object_mask",
                    optional=True,
                    tooltip="Optional per-object scene-space mask. If connected, only vertices whose projected pixel falls inside the mask contribute to the alignment solve.",
                ),
                IO.Int.Input(
                    "batch_index",
                    default=0, min=0, max=1024,
                    tooltip="Which batch slot of the proj_feat_pack/MoGe geometry corresponds to this object.",
                ),
            ],
            outputs=[
                IO.Mesh.Output("aligned_mesh"),
                IO.Float.Output(display_name="scale"),
            ],
        )

    @classmethod
    def execute(cls, mesh, proj_feat_pack, moge_geometry, object_mask=None, batch_index=0) -> IO.NodeOutput:
        vertices = mesh.vertices
        faces = mesh.faces
        if vertices.ndim == 3:
            vertices_one = vertices[0]
            faces_one = faces[0]
        else:
            vertices_one = vertices
            faces_one = faces

        T = proj_feat_pack["transform_matrix"][batch_index:batch_index + 1]
        cam_angle = proj_feat_pack["camera_angle_x"][batch_index:batch_index + 1]
        mesh_scale = proj_feat_pack["mesh_scale"][batch_index]
        image_resolution = int(proj_feat_pack.get("image_resolution", 1024))
        crop_bbox = proj_feat_pack["crop_bboxes"][batch_index]
        pack_scene_size = proj_feat_pack.get("scene_sizes", [None] * (batch_index + 1))[batch_index]
        moge_points = moge_geometry["points"]
        moge_mask = moge_geometry["mask"]
        if moge_points.ndim != 4:
            raise ValueError(f"MoGe points expected [B, H, W, 3]; got {tuple(moge_points.shape)}")
        scene_H, scene_W = moge_points.shape[1], moge_points.shape[2]
        if pack_scene_size is not None and pack_scene_size != (scene_W, scene_H):
            raise ValueError(
                f"Pixal3DAlignObject: MoGe geometry was computed on a {scene_W}x{scene_H} image, "
                f"but the proj_feat_pack's bbox lives in a {pack_scene_size[0]}x{pack_scene_size[1]} "
                "image. Run MoGe on the same resized scene image Pixal3DConditioning used."
            )

        # Compose VaeDecodeShapeTrellis's R_y(180°) inverse with R_proj to map user mesh
        # space to ProjGrid world: (X, Y, Z) -> (-X, Z, Y).
        v = vertices_one.float()
        verts_world = torch.stack([-v[..., 0], v[..., 2], v[..., 1]], dim=-1)
        verts_world = verts_world / float(mesh_scale.item())
        uv_crop, _depth, valid = _project_vertices_to_image_uv(
            verts_world, T[0], cam_angle[0], image_resolution)
        scene_pixels = _crop_uv_to_scene_pixels(uv_crop, crop_bbox, (scene_W, scene_H))
        in_scene = ((scene_pixels[:, 0] >= 0) & (scene_pixels[:, 0] < scene_W) &
                    (scene_pixels[:, 1] >= 0) & (scene_pixels[:, 1] < scene_H))
        sx = scene_pixels[:, 0].long().clamp(0, scene_W - 1)
        sy = scene_pixels[:, 1].long().clamp(0, scene_H - 1)
        moge_per_vertex = moge_points[batch_index, sy, sx]
        moge_mask_per_vertex = moge_mask[batch_index, sy, sx]
        keep = valid & in_scene & moge_mask_per_vertex
        if object_mask is not None:
            om = object_mask if object_mask.ndim == 2 else object_mask[batch_index]
            keep = keep & (om[sy, sx] > 0.5)

        finite = torch.isfinite(moge_per_vertex).all(dim=-1)
        keep = keep & finite

        kept = int(keep.sum().item())
        if kept < 8:
            scale = 1.0
            aligned = vertices_one
        else:
            P = vertices_one[keep].float()
            Q = moge_per_vertex[keep].float()
            p_mean = P.mean(dim=0, keepdim=True)
            q_mean = Q.mean(dim=0, keepdim=True)
            P_c = P - p_mean
            Q_c = Q - q_mean
            num = (P_c * Q_c).sum()
            den = (P_c * P_c).sum().clamp(min=1e-8)
            scale = float((num / den).item())
            if not (scale > 0):
                # Negative scale would mirror the mesh; treat as a camera-convention mismatch.
                logging.warning(
                    f"Pixal3DAlignObject: computed scale={scale:.4f} <= 0; "
                    "refusing to apply mirroring. Check camera convention alignment.")
                scale = 1.0
                aligned = vertices_one
            else:
                t = q_mean - scale * p_mean
                aligned = scale * vertices_one + t

        if vertices.ndim == 3:
            aligned = aligned.unsqueeze(0)
            out_mesh = Types.MESH(vertices=aligned, faces=faces)
        else:
            out_mesh = Types.MESH(vertices=aligned, faces=faces_one)
        return IO.NodeOutput(out_mesh, float(scale))


class LoadNAFModel(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadNAFModel",
            display_name="Load NAF Model",
            category="loaders",
            inputs=[
                IO.Combo.Input(
                    "naf_name",
                    options=folder_paths.get_filename_list("upscale_models"),
                    tooltip="NAF safetensors checkpoint (e.g. naf_release.safetensors).",
                ),
            ],
            outputs=[NAFModel.Output(display_name="naf_model")],
        )

    @classmethod
    def execute(cls, naf_name) -> IO.NodeOutput:
        path = folder_paths.get_full_path_or_raise("upscale_models", naf_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        model = build_naf_from_state_dict(sd)
        device = comfy.model_management.get_torch_device()
        model = model.to(device).eval()
        return IO.NodeOutput(model)


class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
            Pixal3DConditioning,
            Pixal3DAlignObject,
            LoadNAFModel,
            EmptyTrellis2ShapeLatent,
            EmptyTrellis2LatentStructure,
            EmptyTrellis2LatentTexture,
            VaeDecodeTextureTrellis,
            VaeDecodeShapeTrellis,
            VaeDecodeStructureTrellis2,
            Trellis2UpsampleCascade,
        ]


async def comfy_entrypoint() -> Trellis2Extension:
    return Trellis2Extension()

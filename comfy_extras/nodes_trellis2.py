from typing_extensions import override
from comfy_api.latest import ComfyExtension, IO, Types, io
from comfy.ldm.trellis2.vae import SparseTensor
from comfy_extras.nodes_mesh_postprocess import pack_variable_mesh_batch
import comfy.model_management
from PIL import Image
import numpy as np
import torch

ShapeSubdivides = io.Custom("SHAPE_SUBDIVIDES")

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

        voxel = Types.VOXEL(voxel_coords, color_feats, 1024)
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
        out = Types.VOXEL(decoded.squeeze(1).float())
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
    def execute(cls, clip_vision_model, image, mask) -> IO.NodeOutput:
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
            item_mask = mask[b] if mask.size(0) > 1 else mask[0]

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

            bg_rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)

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
            )
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel):
        # to accept the upscaled coords
        is_512_pass = False
        upsampled = hasattr(voxel, "upsampled")
        if upsampled:
            voxel = voxel.data

        if not upsampled:
            decoded = voxel.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
            is_512_pass = True

        else:
            coords = voxel.int()
            is_512_pass = False

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)
        in_channels = 32
        # image like format
        latent = torch.zeros(batch_size, in_channels, max_tokens, 1)

        if is_512_pass:
            generation_mode = "shape_generation_512"
        else:
            generation_mode = "shape_generation"
        return IO.NodeOutput({"samples": latent, "coords": coords, "coord_counts": counts, "type": "trellis2",
                              "model_options": {"generation_mode": generation_mode, "coords": coords, "coord_counts": counts}})

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
            ],
            outputs=[
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, voxel, shape_latent):
        channels = 32
        upsampled = hasattr(voxel, "upsampled")
        if upsampled:
            voxel = voxel.data

        if not upsampled:
            decoded = voxel.data.unsqueeze(1)
            coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
        else:
            coords = voxel.int()

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)

        shape_latent = shape_latent["samples"]
        if shape_latent.ndim == 4:
            shape_latent = shape_latent.squeeze(-1).transpose(1, 2).reshape(-1, channels)

        latent = torch.zeros(batch_size, channels, max_tokens, 1)
        return IO.NodeOutput({"samples": latent, "type": "trellis2", "coords": coords, "coord_counts": counts,
                                     "model_options": {"generation_mode": "texture_generation",
                                                       "coords": coords, "coord_counts": counts, "shape_slat": shape_latent}})


class EmptyTrellis2LatentStructure(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="EmptyTrellis2LatentStructure",
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
        latent = torch.zeros(batch_size, in_channels, resolution, resolution, resolution)
        output = {
            "samples": latent,
            "type": "trellis2",
        }
        return IO.NodeOutput(output)

class Trellis2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            Trellis2Conditioning,
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
